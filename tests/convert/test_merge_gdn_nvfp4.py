"""Contract tests for the GDN NVFP4 merge builder (tools.convert.merge_gdn_nvfp4).

The synthetic test builds a small baseline artifact with the real GDN parent
geometry, merges a synthetic Minima checkpoint through the production import
chain, and checks that baseline objects are re-streamed byte-for-byte, the
dual bindings/uses/auxiliaries are structurally correct, and the new NVFP4
parents are bit-exact against the checkpoint words with a round-trip decode.
The real-artifact test verifies the built merged .ninfer the same way,
including a full byte comparison of the re-streamed baseline payload.
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
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.merge_gdn_nvfp4 import (
    K,
    K_ROWS,
    PARENT_SHAPE,
    QKV_ROWS,
    Q_ROWS,
    ROLES,
    Z_ROWS,
    gdn_layers,
    merge,
)
from tools.convert.sources.safetensors import SafetensorsSource

MINIMA_CHECKPOINT = Path(r"D:\Ai\minima-ai\mnma_qwen3.8_27b_nvfp4\model.safetensors")
BASELINE = Path(r"D:\Ai\neroued\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4.ninfer")
MERGED = Path(r"D:\Ai\neroued\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4_gdn.ninfer")
LAYERS = (0, 2)

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


def test_merge_synthetic_dual_variant(tmp_path):
    baseline = tmp_path / "baseline.ninfer"
    minima = tmp_path / "model.safetensors"
    merged_path = tmp_path / "merged.ninfer"
    _write_baseline(baseline)
    expected = _write_minima(minima)

    report = merge(baseline, minima, merged_path)
    assert report["layers"] == list(LAYERS)
    assert report["parents"] == {0: "weight/000003", 2: "weight/000004"}

    base = Artifact(baseline)
    merged = Artifact(merged_path)
    directory = merged.directory

    # Every baseline object is re-streamed byte-for-byte.
    assert len(directory.objects) == 3 + 2 + 8
    for obj in base.directory.objects:
        assert merged.read_object(obj.id) == base.read_object(obj.id), obj.id

    # Dual bindings: same row ranges on the new NVFP4 parent, baseline kept.
    for layer in LAYERS:
        for role in ROLES:
            fp8 = base.directory.bindings[f"text/layers/{layer}/gdn/{role}"]
            nv = directory.bindings[f"text/layers/{layer}/gdn/{role}_nvfp4"]
            assert nv["parts"][0]["range"] == fp8["parts"][0]["range"]
            assert nv["parts"][0]["object"] == report["parents"][layer]
            assert nv["parts"][0]["object"] != fp8["parts"][0]["object"]

    # Uses: baseline entries kept, one AllowA4 use per new parameter.
    baseline_uses = [dict(u) for u in base.directory.uses]
    assert [u for u in directory.uses if u not in baseline_uses and "_nvfp4" not in u["parameter"]] == []
    for layer in LAYERS:
        for role in ROLES:
            use = next(
                u
                for u in directory.uses
                if u["parameter"] == f"text/layers/{layer}/gdn/{role}_nvfp4"
            )
            assert use["activation_policy"] == "AllowA4"
            assert use["input"] == f"text/layers/{layer}/mixer_input"
            aux_id = use["auxiliaries"]["activation_input_divisor"]["object"]
            assert merged.read_object(aux_id) == expected[layer]["input_divisors"][role]

    # New NVFP4 parents are bit-exact against the checkpoint words.
    geometry = block_scale_geometry("nvfp4", PARENT_SHAPE)
    for layer in LAYERS:
        payload = merged.read_object(report["parents"][layer])
        assert len(payload) == geometry.payload_bytes
        exp = expected[layer]
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
    base.close()
    merged.close()


@pytest.mark.skipif(
    not (BASELINE.is_file() and MERGED.is_file() and MINIMA_CHECKPOINT.is_file()),
    reason="real baseline, merged artifact, or Minima checkpoint missing",
)
def test_merged_real_artifact_contract():
    base = Artifact(BASELINE)
    merged = Artifact(MERGED)
    layers = gdn_layers(base.directory.bindings)
    assert len(layers) == 48
    directory = merged.directory

    # 48 new NVFP4 parents and 192 activation-divisor auxiliaries.
    assert len(directory.objects) == len(base.directory.objects) + 48 + 48 * 4

    # The entire re-streamed baseline payload is byte-identical.
    total = base.payload_bytes
    for chunk_a, chunk_b in zip(base.iter_range(0, total), merged.iter_range(0, total)):
        assert chunk_a == chunk_b

    parent_ids = set()
    for layer in layers:
        for role in ROLES:
            fp8 = base.directory.bindings[f"text/layers/{layer}/gdn/{role}"]
            nv = directory.bindings[f"text/layers/{layer}/gdn/{role}_nvfp4"]
            assert nv["parts"][0]["range"] == fp8["parts"][0]["range"]
            parent_ids.add(nv["parts"][0]["object"])
    assert len(parent_ids) == 48
    for parent_id in parent_ids:
        obj = merged.by_id[parent_id]
        assert (obj.format, obj.layout, obj.shape) == (
            "nvfp4",
            "block_scale_k16_m128x4_v1",
            PARENT_SHAPE,
        )
    for layer in layers:
        for role in ROLES:
            use = next(
                u
                for u in directory.uses
                if u["parameter"] == f"text/layers/{layer}/gdn/{role}_nvfp4"
            )
            assert use["activation_policy"] == "AllowA4"
            aux = merged.by_id[use["auxiliaries"]["activation_input_divisor"]["object"]]
            assert (aux.format, aux.shape) == ("fp32", ())

    # Decode two real parents and check them against the Minima source words.
    geometry = block_scale_geometry("nvfp4", PARENT_SHAPE)
    with SafetensorsSource(MINIMA_CHECKPOINT) as store:
        for layer in (0, layers[-1]):
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
                f"text/layers/{layer}/gdn/query_nvfp4"
            ]["parts"][0]["object"]
            payload = merged.read_object(parent_id)
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
    merged.close()
