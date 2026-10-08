"""Contract tests for the mixed GDN QKVZ finalizer (tools.convert.finalize_gdn_nvfp4).

The synthetic test builds a small baseline artifact with the real GDN parent
geometry, finalizes one GDN layer through the production import chain, and
checks that retained baseline objects are re-streamed byte-for-byte, the
replaced FP8 parent is dropped without leaving unreferenced tensor objects,
the rebound bindings/uses are structurally correct, and the new NVFP4 parent
is bit-exact against the checkpoint words with a round-trip decode. The
real-artifact test verifies the built mixed .ninfer the same way.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, swizzle_nvfp4_scales
from tools.artifact.layouts import block_scale_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.finalize_gdn_nvfp4 import finalize, unreferenced_tensor_objects
from tools.convert.merge_gdn_nvfp4 import (
    K,
    K_ROWS,
    PARENT_SHAPE,
    QKV_ROWS,
    Q_ROWS,
    ROLES,
    Z_ROWS,
    gdn_layers,
)
from tools.convert.sources.safetensors import SafetensorsSource

MINIMA_CHECKPOINT = Path(r"D:\Ai\minima-ai\mnma_qwen3.8_27b_nvfp4\model.safetensors")
BASELINE = Path(r"D:\Ai\neroued\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4.ninfer")
FINAL = Path(
    r"D:\Ai\dpstill\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4_gdn_mixed.ninfer"
)
LAYERS = (0, 2)
NVFP4_LAYER = 2
FP8_LAYER = 0

E2M1 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _logical_values(
    codes: torch.Tensor, scales: torch.Tensor, divisor: bytes, k: int
) -> torch.Tensor:
    """Naive FP32 decode of NVFP4 words: E2M1 codes * E4M3 scales / divisor."""
    table = torch.tensor(E2M1)
    e2m1 = torch.stack(
        (table[(codes & 15).long()], table[(codes >> 4).long()]), dim=-1
    ).reshape(codes.shape[0], k)
    e4m3 = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return e2m1 * e4m3 / struct.unpack("<f", divisor)[0]


def _pattern(object_id: str, length: int) -> bytes:
    base = object_id.encode("utf-8")
    return (base * (length // len(base) + 1))[:length]


def _write_baseline(path: Path) -> None:
    specs = [
        TensorSpec("weight/000000", PARENT_SHAPE, "fp8_e4m3fn_row_bf16", "row_scale_v1"),
        TensorSpec("weight/000001", PARENT_SHAPE, "fp8_e4m3fn_row_bf16", "row_scale_v1"),
        TensorSpec("weight/000002", (64, K), "bf16", "contiguous_le_v1"),
    ]
    parent = {0: "weight/000000", 2: "weight/000001"}
    ranges = [
        [0, Q_ROWS * K],
        [Q_ROWS * K, (Q_ROWS + K_ROWS) * K],
        [(Q_ROWS + K_ROWS) * K, QKV_ROWS * K],
        [QKV_ROWS * K, (QKV_ROWS + Z_ROWS) * K],
    ]
    bindings = {}
    uses = []
    for layer in LAYERS:
        for role, rng in zip(ROLES, ranges):
            bindings[f"text/layers/{layer}/gdn/{role}"] = {
                "parts": [{"object": parent[layer], "range": rng}]
            }
            uses.append(
                {
                    "parameter": f"text/layers/{layer}/gdn/{role}",
                    "input": f"text/layers/{layer}/mixer_input",
                    "activation_policy": "AllowA8",
                }
            )
    bindings["text/layers/1/attention/query"] = {"object": "weight/000002"}
    uses.append(
        {
            "parameter": "text/layers/1/attention/query",
            "input": "text/layers/1/attn_input",
            "activation_policy": "A16Only",
        }
    )
    with ArtifactWriter(
        path,
        specs,
        components={"text": {"config": {"num_hidden_layers": 3}}},
        bindings=bindings,
        uses=uses,
        metadata={"name": "synthetic"},
    ) as writer:
        for obj in writer.objects:
            writer.write_object(obj.id, _pattern(obj.id, obj.bytes))


def _write_minima(path: Path) -> dict:
    tensors = {}
    expected = {}
    for layer in LAYERS:
        qkv_packed = torch.zeros((QKV_ROWS, K // 2), dtype=torch.uint8)
        qkv_packed[:Q_ROWS] = 0x12 + 16 * layer
        qkv_packed[Q_ROWS : Q_ROWS + K_ROWS] = 0x21 + 16 * layer
        qkv_packed[Q_ROWS + K_ROWS :] = 0x30 + 16 * layer
        z_packed = torch.full((Z_ROWS, K // 2), 0x44 + 16 * layer, dtype=torch.uint8)
        qkv_scales = torch.zeros((QKV_ROWS, K // 16), dtype=torch.uint8)
        qkv_scales[:Q_ROWS] = 0x41 + layer
        qkv_scales[Q_ROWS : Q_ROWS + K_ROWS] = 0x42 + layer
        qkv_scales[Q_ROWS + K_ROWS :] = 0x43 + layer
        z_scales = torch.full((Z_ROWS, K // 16), 0x48 + layer, dtype=torch.uint8)
        prefix = f"model.layers.{layer}.linear_attn"
        weight_divisor = 6176.0 + layer
        input_divisor_qkv = 53.5 + layer
        input_divisor_z = 54.5 + layer
        tensors.update(
            {
                f"{prefix}.in_proj_qkv.weight_packed": qkv_packed,
                f"{prefix}.in_proj_qkv.weight_scale": qkv_scales.view(torch.float8_e4m3fn),
                f"{prefix}.in_proj_qkv.weight_global_scale": torch.tensor(
                    [weight_divisor]
                ),
                f"{prefix}.in_proj_qkv.input_global_scale": torch.tensor(
                    [input_divisor_qkv]
                ),
                f"{prefix}.in_proj_z.weight_packed": z_packed,
                f"{prefix}.in_proj_z.weight_scale": z_scales.view(torch.float8_e4m3fn),
                f"{prefix}.in_proj_z.weight_global_scale": torch.tensor(
                    [weight_divisor]
                ),
                f"{prefix}.in_proj_z.input_global_scale": torch.tensor(
                    [input_divisor_z]
                ),
            }
        )
        expected[layer] = {
            "codes": torch.cat([qkv_packed, z_packed]),
            "scales": torch.cat([qkv_scales, z_scales]),
            "divisor": struct.pack("<f", weight_divisor),
            "input_divisors": {
                "query": struct.pack("<f", input_divisor_qkv),
                "key": struct.pack("<f", input_divisor_qkv),
                "value": struct.pack("<f", input_divisor_qkv),
                "z": struct.pack("<f", input_divisor_z),
            },
        }
    save_file(tensors, str(path))
    return expected


def test_finalize_synthetic_mixed_profile(tmp_path):
    baseline = tmp_path / "baseline.ninfer"
    minima = tmp_path / "model.safetensors"
    final_path = tmp_path / "mixed.ninfer"
    _write_baseline(baseline)
    expected = _write_minima(minima)

    report = finalize(baseline, minima, final_path, str(NVFP4_LAYER))
    assert report["fp8_layers"] == [FP8_LAYER]
    assert report["nvfp4_layers"] == [NVFP4_LAYER]
    assert report["dropped"] == ["weight/000001"]
    assert report["parents"] == {NVFP4_LAYER: "weight/000003"}
    assert report["unreferenced_tensor_objects"] == 0

    base = Artifact(baseline)
    final = Artifact(final_path)
    directory = final.directory

    # Retained baseline objects are re-streamed byte-for-byte; the replaced
    # FP8 parent is dropped, the new NVFP4 parent and auxiliaries exist.
    assert len(directory.objects) == 3 - 1 + 1 + 4
    for obj in base.directory.objects:
        if obj.id in report["dropped"]:
            assert obj.id not in final.by_id
        else:
            assert final.read_object(obj.id) == base.read_object(obj.id), obj.id

    # Rebound binding: the same row ranges on the new NVFP4 parent.
    for role in ROLES:
        fp8 = base.directory.bindings[f"text/layers/{NVFP4_LAYER}/gdn/{role}"]
        nv = directory.bindings[f"text/layers/{NVFP4_LAYER}/gdn/{role}"]
        assert nv["parts"][0]["range"] == fp8["parts"][0]["range"]
        assert nv["parts"][0]["object"] == report["parents"][NVFP4_LAYER]

    # The FP8 layer keeps its baseline binding and Use.
    for role in ROLES:
        name = f"text/layers/{FP8_LAYER}/gdn/{role}"
        assert directory.bindings[name] == base.directory.bindings[name]
    for use in directory.uses:
        if use["parameter"].startswith(f"text/layers/{FP8_LAYER}/gdn/"):
            assert use["activation_policy"] == "AllowA8"
            assert "auxiliaries" not in use

    # Rebound Use: AllowA4 with the Minima input-divisor auxiliary.
    for role in ROLES:
        use = next(
            u
            for u in directory.uses
            if u["parameter"] == f"text/layers/{NVFP4_LAYER}/gdn/{role}"
        )
        assert use["activation_policy"] == "AllowA4"
        assert use["input"] == f"text/layers/{NVFP4_LAYER}/mixer_input"
        aux_id = use["auxiliaries"]["activation_input_divisor"]["object"]
        assert final.read_object(aux_id) == expected[NVFP4_LAYER]["input_divisors"][role]

    # The new NVFP4 parent is bit-exact against the checkpoint words.
    geometry = block_scale_geometry("nvfp4", PARENT_SHAPE)
    payload = final.read_object(report["parents"][NVFP4_LAYER])
    assert len(payload) == geometry.payload_bytes
    exp = expected[NVFP4_LAYER]
    assert payload[: geometry.code_plane_bytes] == exp["codes"].numpy().tobytes()
    assert (
        payload[
            geometry.scale_plane_offset : geometry.scale_plane_offset
            + geometry.scale_plane_bytes
        ]
        == swizzle_nvfp4_scales(exp["scales"], PARENT_SHAPE).numpy().tobytes()
    )
    assert (
        payload[geometry.weight_divisor_offset : geometry.weight_divisor_offset + 4]
        == exp["divisor"]
    )
    codes, scales, divisor = decode_nvfp4_words(payload, PARENT_SHAPE)
    assert torch.equal(codes, exp["codes"])
    assert torch.equal(scales, exp["scales"])
    # Row order: Q [0,2048), K [2048,4096), V [4096,10240), Z [10240,16384).
    for begin in (0, Q_ROWS, Q_ROWS + 2048, QKV_ROWS, PARENT_SHAPE[0] - 128):
        tile = _logical_values(
            codes[begin : begin + 128],
            scales[begin : begin + 128],
            divisor.numpy().tobytes(),
            K,
        )
        source = _logical_values(
            exp["codes"][begin : begin + 128],
            exp["scales"][begin : begin + 128],
            exp["divisor"],
            K,
        )
        assert torch.equal(tile, source)

    # No unreferenced tensor objects and the profile is recorded.
    assert unreferenced_tensor_objects(directory) == []
    assert directory.provenance["gdn_nvfp4_finalize"]["fp8_layers"] == [FP8_LAYER]
    assert (
        directory.provenance["gdn_nvfp4_finalize"]["nvfp4_layers"] == [NVFP4_LAYER]
    )
    base.close()
    final.close()


def test_finalize_rejects_invalid_profiles(tmp_path):
    baseline = tmp_path / "baseline.ninfer"
    minima = tmp_path / "model.safetensors"
    out = tmp_path / "mixed.ninfer"
    _write_baseline(baseline)
    _write_minima(minima)

    # Layer 1 is the attention layer, not a GDN layer.
    with pytest.raises(ValueError, match="not a GDN"):
        finalize(baseline, minima, out, "1")
    with pytest.raises(ValueError, match="duplicate"):
        finalize(baseline, minima, out, "2,2")
    with pytest.raises(ValueError, match="empty"):
        finalize(baseline, minima, out, "0,")
    assert not out.exists()


@pytest.mark.skipif(
    not (BASELINE.is_file() and FINAL.is_file() and MINIMA_CHECKPOINT.is_file()),
    reason="real baseline, mixed artifact, or Minima checkpoint missing",
)
def test_mixed_real_artifact_contract():
    base = Artifact(BASELINE)
    final = Artifact(FINAL)
    gdn = gdn_layers(base.directory.bindings)
    assert len(gdn) == 48
    directory = final.directory
    profile = directory.provenance["gdn_nvfp4_finalize"]
    fp8_layers = profile["fp8_layers"]
    nvfp4_layers = profile["nvfp4_layers"]
    assert sorted(fp8_layers + nvfp4_layers) == gdn
    assert len(nvfp4_layers) == 37

    # 37 dropped FP8 parents, 37 new NVFP4 parents, 148 divisor auxiliaries.
    assert len(directory.objects) == len(base.directory.objects) - 37 + 37 + 37 * 4

    dropped_ids = set()
    for layer in nvfp4_layers:
        for role in ROLES:
            dropped_ids.add(
                base.directory.bindings[f"text/layers/{layer}/gdn/{role}"][
                    "parts"
                ][0]["object"]
            )
    assert len(dropped_ids) == 37
    for object_id in dropped_ids:
        assert object_id not in final.by_id

    # Every retained object is re-streamed byte-for-byte.
    for obj in base.directory.objects:
        if obj.id in final.by_id:
            assert final.read_object(obj.id) == base.read_object(obj.id), obj.id
        else:
            assert (
                isinstance(obj, TensorObject)
                and obj.format == "fp8_e4m3fn_row_bf16"
                and obj.id in dropped_ids
            ), obj.id
    assert unreferenced_tensor_objects(directory) == []

    parent_ids = set()
    for layer in nvfp4_layers:
        for role in ROLES:
            name = f"text/layers/{layer}/gdn/{role}"
            fp8 = base.directory.bindings[name]
            nv = directory.bindings[name]
            assert nv["parts"][0]["range"] == fp8["parts"][0]["range"]
            parent_ids.add(nv["parts"][0]["object"])
    assert len(parent_ids) == 37
    for parent_id in parent_ids:
        obj = final.by_id[parent_id]
        assert (obj.format, obj.layout, obj.shape) == (
            "nvfp4",
            "block_scale_k16_m128x4_v1",
            PARENT_SHAPE,
        )
    for layer in nvfp4_layers:
        for role in ROLES:
            use = next(
                u
                for u in directory.uses
                if u["parameter"] == f"text/layers/{layer}/gdn/{role}"
            )
            assert use["activation_policy"] == "AllowA4"
            aux = final.by_id[use["auxiliaries"]["activation_input_divisor"]["object"]]
            assert (aux.format, aux.shape) == ("fp32", ())
    for layer in fp8_layers:
        for role in ROLES:
            name = f"text/layers/{layer}/gdn/{role}"
            assert directory.bindings[name] == base.directory.bindings[name]
            for use in directory.uses:
                if use["parameter"] == name:
                    assert "auxiliaries" not in use

    # Decode two real parents and check them against the Minima source words.
    geometry = block_scale_geometry("nvfp4", PARENT_SHAPE)
    with SafetensorsSource(MINIMA_CHECKPOINT) as store:
        for layer in (nvfp4_layers[0], nvfp4_layers[-1]):
            prefix = f"model.layers.{layer}.linear_attn"
            qkv_packed = store.read_flat(
                f"{prefix}.in_proj_qkv.weight_packed"
            ).reshape(QKV_ROWS, K // 2)
            z_packed = store.read_flat(f"{prefix}.in_proj_z.weight_packed").reshape(
                Z_ROWS, K // 2
            )
            qkv_scales = (
                store.read_flat(f"{prefix}.in_proj_qkv.weight_scale")
                .view(torch.uint8)
                .reshape(QKV_ROWS, K // 16)
            )
            z_scales = (
                store.read_flat(f"{prefix}.in_proj_z.weight_scale")
                .view(torch.uint8)
                .reshape(Z_ROWS, K // 16)
            )
            weight_divisor = store.read_flat(
                f"{prefix}.in_proj_qkv.weight_global_scale"
            ).view(torch.uint8).numpy().tobytes()
            assert weight_divisor == store.read_flat(
                f"{prefix}.in_proj_z.weight_global_scale"
            ).view(torch.uint8).numpy().tobytes()

            parent_id = directory.bindings[
                f"text/layers/{layer}/gdn/query"
            ]["parts"][0]["object"]
            payload = final.read_object(parent_id)
            assert payload[: geometry.code_plane_bytes] == torch.cat(
                [qkv_packed, z_packed]
            ).numpy().tobytes()
            assert (
                payload[
                    geometry.scale_plane_offset : geometry.scale_plane_offset
                    + geometry.scale_plane_bytes
                ]
                == swizzle_nvfp4_scales(
                    torch.cat([qkv_scales, z_scales]), PARENT_SHAPE
                ).numpy().tobytes()
            )
            assert (
                payload[
                    geometry.weight_divisor_offset : geometry.weight_divisor_offset + 4
                ]
                == weight_divisor
            )
            codes, scales, divisor = decode_nvfp4_words(payload, PARENT_SHAPE)
            for begin in (0, Q_ROWS, QKV_ROWS, PARENT_SHAPE[0] - 128):
                tile = _logical_values(
                    codes[begin : begin + 128],
                    scales[begin : begin + 128],
                    divisor.numpy().tobytes(),
                    K,
                )
                source = _logical_values(
                    torch.cat([qkv_packed, z_packed])[begin : begin + 128],
                    torch.cat([qkv_scales, z_scales])[begin : begin + 128],
                    weight_divisor,
                    K,
                )
                assert torch.equal(tile, source)
    base.close()
    final.close()
