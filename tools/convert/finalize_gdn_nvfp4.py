"""Finalize a mixed GDN QKVZ artifact: FP8 baseline plus Minima NVFP4 layers.

Post-processes a qwen3_8_27b_nvfp4 .ninfer artifact: every baseline object is
re-streamed byte-for-byte except the replaced GDN in_proj FP8 parents, which
are dropped. For each selected GDN (linear attention) layer one NVFP4 in_proj
parent (qkv and z packed into one [16384, 5120] object) is imported bit-exact
from a Minima checkpoint, with one FP32 activation-divisor auxiliary per
parameter, and the layer's gdn/{role} bindings and Uses are rebound from the
FP8 parent to the NVFP4 parent with AllowA4. All other objects, bindings and
Uses are unchanged, so the runtime takes the FP8 or NVFP4 GDN projection per
layer from the stored profile without a layer-selection option.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.merge_gdn_nvfp4 import (
    AUX_FORMAT,
    AUX_LAYOUT,
    NVFP4_LAYOUT,
    PARENT_SHAPE,
    ROLES,
    _baseline_specs,
    _import_layer,
    gdn_layers,
)
from tools.convert.sources.safetensors import SafetensorsSource


def resolve_layers(value: str | Sequence[int], gdn: list[int]) -> list[int]:
    """Resolve and validate the NVFP4 layer profile against the GDN layers."""
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
        if any(not token for token in tokens):
            raise ValueError("layer list must not contain empty entries")
        candidates = []
        for token in tokens:
            try:
                candidates.append(int(token))
            except ValueError:
                raise ValueError(f"invalid layer number {token!r}") from None
    else:
        candidates = [int(layer) for layer in value]
    gdn_set = set(gdn)
    layers: list[int] = []
    for layer in candidates:
        if layer not in gdn_set:
            raise ValueError(f"layer {layer} is not a GDN (linear attention) layer")
        if layer in layers:
            raise ValueError(f"duplicate layer {layer} in the NVFP4 profile")
        layers.append(layer)
    if not layers:
        raise ValueError("the NVFP4 layer list must not be empty")
    return sorted(layers)


def _binding_objects(binding: dict) -> list[str]:
    if "parts" in binding:
        return [part["object"] for part in binding["parts"]]
    return [binding["object"]]


def unreferenced_tensor_objects(directory) -> list[str]:
    """Tensor objects referenced by no binding, Use auxiliary, or component resource."""
    references: set[str] = set()
    for binding in directory.bindings.values():
        for object_id in _binding_objects(binding):
            references.add(object_id)
    for use in directory.uses:
        for binding in use.get("auxiliaries", {}).values():
            for object_id in _binding_objects(binding):
                references.add(object_id)
    for component in directory.components.values():
        for object_id in component.get("resources", {}).values():
            references.add(object_id)
    return [
        obj.id
        for obj in directory.objects
        if isinstance(obj, TensorObject) and obj.id not in references
    ]


def verify_final(artifact_path: str | Path, report: dict) -> dict:
    """Reopen the finalized artifact and check its directory contract."""
    with Artifact(artifact_path) as artifact:
        directory = artifact.directory
        unreferenced = unreferenced_tensor_objects(directory)
        if unreferenced:
            raise ArtifactError(f"unreferenced tensor objects: {unreferenced}")
        if len(directory.objects) != report["objects"]:
            raise ArtifactError(
                f"expected {report['objects']} objects, got {len(directory.objects)}"
            )
        for obj in directory.objects:
            artifact.object(obj.id)
        for layer in report["nvfp4_layers"]:
            for role in ROLES:
                name = f"text/layers/{layer}/gdn/{role}"
                part = directory.bindings[name]["parts"][0]
                obj = artifact.by_id[part["object"]]
                if (obj.format, obj.shape) != ("nvfp4", PARENT_SHAPE):
                    raise ArtifactError(
                        f"{name}: parent {part['object']} is not the NVFP4 GDN parent"
                    )
                use = next(u for u in directory.uses if u["parameter"] == name)
                if use["activation_policy"] != "AllowA4":
                    raise ArtifactError(f"{name}: expected the AllowA4 activation policy")
                aux = artifact.by_id[
                    use["auxiliaries"]["activation_input_divisor"]["object"]
                ]
                if (aux.format, aux.shape) != (AUX_FORMAT, ()):
                    raise ArtifactError(
                        f"{name}: divisor auxiliary {aux.id} is not an FP32 scalar"
                    )
        for layer in report["fp8_layers"]:
            for role in ROLES:
                name = f"text/layers/{layer}/gdn/{role}"
                part = directory.bindings[name]["parts"][0]
                obj = artifact.by_id[part["object"]]
                if (obj.format, obj.shape) != ("fp8_e4m3fn_row_bf16", PARENT_SHAPE):
                    raise ArtifactError(
                        f"{name}: parent {part['object']} is not the FP8 GDN parent"
                    )
                for use in directory.uses:
                    if use["parameter"] == name and "auxiliaries" in use:
                        raise ArtifactError(f"{name}: FP8 use gained auxiliaries")
    return {"unreferenced_tensor_objects": 0}


def finalize(
    baseline: str | Path,
    minima: str | Path,
    out: str | Path,
    nvfp4_layers: str | Sequence[int],
    *,
    device: str = "cpu",
) -> dict:
    """Write the finalized mixed artifact and report the object profile."""
    base = Artifact(baseline)
    directory = base.directory
    gdn = gdn_layers(directory.bindings)
    layers = resolve_layers(nvfp4_layers, gdn)
    fp8_layers = [layer for layer in gdn if layer not in set(layers)]

    dropped: dict[str, int] = {}
    for layer in layers:
        for role in ROLES:
            name = f"text/layers/{layer}/gdn/{role}"
            binding = directory.bindings[name]
            parts = binding.get("parts")
            if parts is None or len(parts) != 1:
                raise ValueError(f"{name}: expected a one-part baseline binding")
            object_id = parts[0]["object"]
            obj = base.by_id[object_id]
            if obj.format != "fp8_e4m3fn_row_bf16" or tuple(obj.shape) != PARENT_SHAPE:
                raise ValueError(
                    f"{name}: baseline parent {object_id} is not the "
                    f"FP8 {PARENT_SHAPE} GDN parent"
                )
            dropped[object_id] = layer

    referencers: dict[str, set[str]] = {}
    for name, binding in directory.bindings.items():
        for object_id in _binding_objects(binding):
            referencers.setdefault(object_id, set()).add(name)
    aux_referencers: dict[str, set[str]] = {}
    for use in directory.uses:
        for role, binding in use.get("auxiliaries", {}).items():
            for object_id in _binding_objects(binding):
                aux_referencers.setdefault(object_id, set()).add(
                    f"{use['parameter']}@{use['input']}/{role}"
                )
    for object_id, layer in dropped.items():
        expected = {f"text/layers/{layer}/gdn/{role}" for role in ROLES}
        actual = referencers.get(object_id, set())
        if actual != expected:
            raise ValueError(
                f"{object_id}: expected references {sorted(expected)}, "
                f"got {sorted(actual)}"
            )
        if object_id in aux_referencers:
            raise ValueError(
                f"{object_id}: referenced by auxiliary "
                f"{sorted(aux_referencers[object_id])}"
            )

    weight_max = max(
        (
            int(obj.id.split("/")[1])
            for obj in directory.objects
            if obj.id.startswith("weight/")
        ),
        default=-1,
    )
    aux_max = max(
        (
            int(obj.id.split("/")[1])
            for obj in directory.objects
            if obj.id.startswith("auxiliary/")
        ),
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

    dropped_ids = set(dropped)
    specs = [spec for spec in _baseline_specs(directory) if spec.id not in dropped_ids]
    for layer in layers:
        specs.append(TensorSpec(parent_ids[layer], PARENT_SHAPE, "nvfp4", NVFP4_LAYOUT))
    for layer in layers:
        for role in ROLES:
            specs.append(TensorSpec(aux_ids[(layer, role)], (), AUX_FORMAT, AUX_LAYOUT))

    bindings = dict(directory.bindings)
    for layer in layers:
        for role in ROLES:
            name = f"text/layers/{layer}/gdn/{role}"
            parts = directory.bindings[name]["parts"]
            bindings[name] = {
                "parts": [{"object": parent_ids[layer], "range": list(parts[0]["range"])}]
            }

    rebound = {
        f"text/layers/{layer}/gdn/{role}": (layer, role)
        for layer in layers
        for role in ROLES
    }
    uses = []
    for use in directory.uses:
        hit = rebound.get(use["parameter"])
        if hit is None:
            uses.append(dict(use))
            continue
        layer, role = hit
        uses.append(
            {
                "parameter": use["parameter"],
                "input": use["input"],
                "activation_policy": "AllowA4",
                "auxiliaries": {
                    "activation_input_divisor": {"object": aux_ids[(layer, role)]}
                },
            }
        )
    missing = set(rebound) - {use["parameter"] for use in uses}
    if missing:
        raise ValueError(f"no Use for rebound parameters {sorted(missing)}")

    provenance = dict(directory.provenance)
    provenance["gdn_nvfp4_finalize"] = {
        "baseline": str(baseline),
        "minima": str(minima),
        "fp8_layers": fp8_layers,
        "nvfp4_layers": layers,
    }

    with ArtifactWriter(
        out,
        specs,
        components=directory.components,
        bindings=bindings,
        uses=uses,
        metadata=directory.metadata,
        provenance=provenance,
    ) as writer:
        for obj in directory.objects:
            if obj.id in dropped_ids:
                continue
            writer.write_object(obj.id, base.iter_object(obj.id))
        with SafetensorsSource(minima) as store:
            for layer in layers:
                _import_layer(writer, store, layer, parent_ids[layer], aux_ids, device)
                print(f"finalized layer {layer} ({parent_ids[layer]})")
    base.close()

    report = {
        "fp8_layers": fp8_layers,
        "nvfp4_layers": layers,
        "parents": parent_ids,
        "auxiliaries": aux_ids,
        "dropped": sorted(dropped),
        "objects": len(specs),
        "payload_bytes": writer.directory.payload_bytes,
    }
    report.update(verify_final(out, report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--minima", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--nvfp4-layers",
        required=True,
        help="comma-separated GDN layer numbers imported from the Minima checkpoint",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    started = time.time()
    report = finalize(
        args.baseline, args.minima, args.out, args.nvfp4_layers, device=args.device
    )
    with Artifact(args.out) as artifact:
        print(f"objects: {len(artifact.directory.objects)}")
        print(f"payload bytes: {artifact.payload_bytes}")
    print(
        f"finalized {len(report['nvfp4_layers'])} NVFP4 GDN layers "
        f"({len(report['fp8_layers'])} FP8 kept) into {args.out} "
        f"in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
