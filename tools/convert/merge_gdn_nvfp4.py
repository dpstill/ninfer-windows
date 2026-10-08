"""Merge Minima NVFP4 GDN in_proj variants into an existing NVFP4 artifact.

Post-processes a qwen3_8_27b_nvfp4 .ninfer artifact: every baseline object is
re-streamed byte-for-byte, and for each GDN (linear attention) layer one NVFP4
in_proj parent (qkv and z packed into one [16384, 5120] object) is imported
bit-exact from a Minima checkpoint, with one FP32 activation-divisor
auxiliary per parameter. Dual bindings let the runtime select the FP8
(baseline) or NVFP4 (Minima) GDN projection per layer; an unused variant is
never materialized at load time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, ResourceSpec, TensorObject, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import MethodInput, PrepareRequest, import_encoded
from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.logical import select_rows
from tools.convert.sources.safetensors import SafetensorsSource

NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
FP8_LAYOUT = "row_scale_v1"
AUX_FORMAT = "fp32"
AUX_LAYOUT = "contiguous_le_v1"

Q_ROWS = 2048
K_ROWS = 2048
V_ROWS = 6144
Z_ROWS = 6144
K = 5120
QKV_ROWS = Q_ROWS + K_ROWS + V_ROWS
PARENT_SHAPE = (QKV_ROWS + Z_ROWS, K)
ROLES = ("query", "key", "value", "z")


def gdn_layers(bindings: dict) -> list[int]:
    """Layers whose text component declares a GDN query binding."""
    layers = set()
    for name in bindings:
        if name.startswith("text/layers/") and name.endswith("/gdn/query"):
            layers.add(int(name.split("/")[2]))
    if not layers:
        raise ValueError("baseline artifact declares no GDN layers")
    return sorted(layers)


def _baseline_specs(directory) -> list:
    specs = []
    for obj in directory.objects:
        if isinstance(obj, TensorObject):
            specs.append(TensorSpec(obj.id, obj.shape, obj.format, obj.layout))
        elif isinstance(obj, ResourceObject):
            from tools.artifact.schema import ResourceSpec

            specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
        else:
            raise TypeError(f"unknown object kind {type(obj).__name__}")
    return specs


def _import_layer(
    writer: ArtifactWriter,
    store: SafetensorsSource,
    layer: int,
    parent_id: str,
    aux_ids: dict,
    device: str,
) -> None:
    prefix = f"model.layers.{layer}.linear_attn"
    qkv = matrix_source(store, f"{prefix}.in_proj_qkv.weight", (QKV_ROWS, K))
    z = matrix_source(store, f"{prefix}.in_proj_z.weight", (Z_ROWS, K))
    spans = {
        "query": (0, Q_ROWS),
        "key": (Q_ROWS, Q_ROWS + K_ROWS),
        "value": (Q_ROWS + K_ROWS, QKV_ROWS),
    }
    sources = {role: select_rows(qkv, ((begin, end),)) for role, (begin, end) in spans.items()}
    sources["z"] = z
    inputs = tuple(
        MethodInput(f"gdn/{role}", source, ((f"gdn/{role}", "mixer_input"),))
        for role, source in sources.items()
    )
    policies = {(f"gdn/{role}", "mixer_input"): "AllowA4" for role in ROLES}
    request = PrepareRequest(
        TensorSpec(parent_id, PARENT_SHAPE, "nvfp4", NVFP4_LAYOUT),
        inputs,
        policies,
        {},
        device=device,
    )
    prepared = import_encoded(request)
    prepared.produce(TensorOutput(writer, parent_id))
    for role in ROLES:
        value = prepared.auxiliaries[(f"gdn/{role}", "mixer_input", "activation_input_divisor")]
        writer.write_object(aux_ids[(layer, role)], value.data)


def merge(
    baseline: str | Path,
    minima: str | Path,
    out: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    """Write the merged artifact and report the new object ids."""
    base = Artifact(baseline)
    directory = base.directory
    layers = gdn_layers(directory.bindings)

    weight_max = max(
        (int(obj.id.split("/")[1]) for obj in directory.objects if obj.id.startswith("weight/")),
        default=-1,
    )
    aux_max = max(
        (int(obj.id.split("/")[1]) for obj in directory.objects if obj.id.startswith("auxiliary/")),
        default=-1,
    )
    parent_ids = {
        layer: f"weight/{weight_max + 1 + index:06d}" for index, layer in enumerate(layers)
    }
    aux_ids = {
        (layer, role): f"auxiliary/{aux_max + 1 + index * len(ROLES) + j:06d}"
        for index, layer in enumerate(layers)
        for j, role in enumerate(ROLES)
    }

    specs = _baseline_specs(directory)
    for layer in layers:
        specs.append(TensorSpec(parent_ids[layer], PARENT_SHAPE, "nvfp4", NVFP4_LAYOUT))
    for layer in layers:
        for role in ROLES:
            specs.append(
                TensorSpec(aux_ids[(layer, role)], (), AUX_FORMAT, AUX_LAYOUT)
            )

    bindings = dict(directory.bindings)
    for layer in layers:
        for role in ROLES:
            fp8 = directory.bindings[f"text/layers/{layer}/gdn/{role}"]
            parts = fp8.get("parts")
            if parts is None or len(parts) != 1:
                raise ValueError(
                    f"text/layers/{layer}/gdn/{role}: expected a one-part baseline binding"
                )
            object_id = parts[0]["object"]
            obj = base.by_id[object_id]
            if obj.format != "fp8_e4m3fn_row_bf16" or tuple(obj.shape) != PARENT_SHAPE:
                raise ValueError(
                    f"text/layers/{layer}/gdn/{role}: baseline parent {object_id} is not the "
                    f"FP8 {PARENT_SHAPE} GDN parent"
                )
            bindings[f"text/layers/{layer}/gdn/{role}_nvfp4"] = {
                "parts": [{"object": parent_ids[layer], "range": list(parts[0]["range"])}]
            }

    new_uses = [
        {
            "parameter": f"text/layers/{layer}/gdn/{role}_nvfp4",
            "input": f"text/layers/{layer}/mixer_input",
            "activation_policy": "AllowA4",
            "auxiliaries": {"activation_input_divisor": {"object": aux_ids[(layer, role)]}},
        }
        for layer in layers
        for role in ROLES
    ]

    provenance = dict(directory.provenance)
    provenance["gdn_nvfp4_merge"] = {"minima": str(minima), "layers": layers}

    with ArtifactWriter(
        out,
        specs,
        components=directory.components,
        bindings=bindings,
        uses=[*directory.uses, *new_uses],
        metadata=directory.metadata,
        provenance=provenance,
    ) as writer:
        for obj in directory.objects:
            writer.write_object(obj.id, base.iter_object(obj.id))
        with SafetensorsSource(minima) as store:
            for layer in layers:
                _import_layer(writer, store, layer, parent_ids[layer], aux_ids, device)
                print(f"merged layer {layer} ({parent_ids[layer]})")
    base.close()
    return {
        "layers": layers,
        "parents": parent_ids,
        "auxiliaries": aux_ids,
        "payload_bytes": writer.directory.payload_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--minima", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    report = merge(args.baseline, args.minima, args.out)
    with Artifact(args.out) as merged:
        print(f"objects: {len(merged.directory.objects)}")
        print(f"payload bytes: {merged.payload_bytes}")
    print(
        f"merged {len(report['layers'])} GDN layers into {args.out} "
        f"in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
