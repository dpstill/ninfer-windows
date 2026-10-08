"""Proof of import: GDN in_proj_qkv + in_proj_z into one NVFP4 parent, bit-exact.

Drives the production import chain (SafetensorsSource -> compressed_tensors
matrix_source -> select_rows -> PrepareRequest/import_encoded ->
TensorOutput/ArtifactWriter) and checks that the emitted parent payload is
byte-identical to the checkpoint words: packed E2M1 codes, E4M3 scales after
the registered NInfer swizzle, the stored FP32 weight divisor, and the
activation input divisor. A round-trip decode of the parent payload must
reproduce the source's logical values.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, swizzle_nvfp4_scales
from tools.artifact.layouts import block_scale_geometry
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import MethodInput, PrepareRequest, import_encoded
from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.logical import select_rows
from tools.convert.sources.safetensors import SafetensorsSource

MINIMA_CHECKPOINT = Path(r"D:\Ai\minima-ai\mnma_qwen3.8_27b_nvfp4\model.safetensors")
LAYER0 = "model.layers.0.linear_attn"
QKV_ROWS, Z_ROWS, K = 10240, 6144, 5120
Q_END, K_END = 2048, 4096

E2M1 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _logical_values(
    codes: torch.Tensor, scales: torch.Tensor, divisor: bytes, k: int
) -> torch.Tensor:
    """Naive FP32 decode of NVFP4 words: E2M1 codes * E4M3 scales / divisor."""
    table = torch.tensor(E2M1)
    # Torch 2.14+ CPU rejects a 2D uint8 tensor as an index into a 1D lookup
    # table; .long() supplies the integer index dtype.
    e2m1 = torch.stack(
        (table[(codes & 15).long()], table[(codes >> 4).long()]), dim=-1
    ).reshape(codes.shape[0], k)
    e4m3 = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return e2m1 * e4m3 / struct.unpack("<f", divisor)[0]


def _drive_import(
    q_source, k_source, v_source, z_source, parent_shape: tuple[int, int],
    artifact_path: Path,
):
    """Run import_encoded for one qkv/z parent through a real ArtifactWriter."""
    n, k = parent_shape
    spec = TensorSpec("weight/000000", (n, k), "nvfp4", "block_scale_k16_m128x4_v1")
    inputs = (
        MethodInput("gdn/query", q_source, (("gdn/query", "mixer_input"),)),
        MethodInput("gdn/key", k_source, (("gdn/key", "mixer_input"),)),
        MethodInput("gdn/value", v_source, (("gdn/value", "mixer_input"),)),
        MethodInput("gdn/z", z_source, (("gdn/z", "mixer_input"),)),
    )
    policies = {
        (name, "mixer_input"): "AllowA4"
        for name in ("gdn/query", "gdn/key", "gdn/value", "gdn/z")
    }
    request = PrepareRequest(spec, inputs, policies, {}, device="cpu")
    prepared = import_encoded(request)
    with ArtifactWriter(
        artifact_path,
        [spec],
        components={"text": {"config": {}}},
        bindings={},
    ) as writer:
        prepared.produce(TensorOutput(writer, spec.id))
        payload_offset = writer.payload_offset
    geometry = block_scale_geometry("nvfp4", (n, k))
    payload = artifact_path.read_bytes()[payload_offset:]
    assert len(payload) == geometry.payload_bytes
    return prepared, request, geometry, payload


def _assert_planes(
    payload: bytes,
    geometry,
    expected_codes: bytes,
    expected_scales: bytes,
    expected_divisor: bytes,
) -> None:
    assert payload[: geometry.code_plane_bytes] == expected_codes
    assert (
        payload[
            geometry.scale_plane_offset : geometry.scale_plane_offset
            + geometry.scale_plane_bytes
        ]
        == expected_scales
    )
    assert (
        payload[geometry.weight_divisor_offset : geometry.weight_divisor_offset + 4]
        == expected_divisor
    )


def _assert_activation_divisors(prepared, expected: bytes) -> None:
    assert len(prepared.auxiliaries) == 4
    for name in ("gdn/query", "gdn/key", "gdn/value", "gdn/z"):
        value = prepared.auxiliaries[(name, "mixer_input", "activation_input_divisor")]
        assert (value.format, value.shape, value.data) == ("fp32", (), expected)


def test_synthetic_gdn_qkvz_nvfp4_parent_import(tmp_path):
    k = 128
    q_rows = k_rows = v_rows = z_rows = 128
    qkv_rows = q_rows + k_rows + v_rows
    n = qkv_rows + z_rows
    weight_divisor = struct.pack("<f", 6176.0)
    input_divisor = struct.pack("<f", 53.5)

    qkv_packed = torch.zeros((qkv_rows, k // 2), dtype=torch.uint8)
    qkv_packed[:q_rows] = 0x12
    qkv_packed[q_rows : q_rows + k_rows] = 0x21
    qkv_packed[q_rows + k_rows :] = 0x30
    z_packed = torch.full((z_rows, k // 2), 0x04, dtype=torch.uint8)

    qkv_scales = torch.zeros((qkv_rows, k // 16), dtype=torch.uint8)
    qkv_scales[:q_rows] = 0x40
    qkv_scales[q_rows : q_rows + k_rows] = 0x42
    qkv_scales[q_rows + k_rows :] = 0x44
    z_scales = torch.full((z_rows, k // 16), 0x48, dtype=torch.uint8)

    save_file(
        {
            "gdn.in_proj_qkv.weight_packed": qkv_packed,
            "gdn.in_proj_qkv.weight_scale": qkv_scales.view(torch.float8_e4m3fn),
            "gdn.in_proj_qkv.weight_global_scale": torch.tensor([6176.0]),
            "gdn.in_proj_qkv.input_global_scale": torch.tensor([53.5]),
            "gdn.in_proj_z.weight_packed": z_packed,
            "gdn.in_proj_z.weight_scale": z_scales.view(torch.float8_e4m3fn),
            "gdn.in_proj_z.weight_global_scale": torch.tensor([6176.0]),
            "gdn.in_proj_z.input_global_scale": torch.tensor([53.5]),
        },
        str(tmp_path / "model.safetensors"),
    )

    with SafetensorsSource(tmp_path) as store:
        qkv = matrix_source(store, "gdn.in_proj_qkv.weight", (qkv_rows, k))
        q_source = select_rows(qkv, ((0, q_rows),))
        k_source = select_rows(qkv, ((q_rows, q_rows + k_rows),))
        v_source = select_rows(qkv, ((q_rows + k_rows, qkv_rows),))
        z_source = matrix_source(store, "gdn.in_proj_z.weight", (z_rows, k))
        prepared, request, geometry, payload = _drive_import(
            q_source, k_source, v_source, z_source, (n, k), tmp_path / "gdn.ninfer"
        )

    expected_codes = torch.cat([qkv_packed, z_packed]).numpy().tobytes()
    expected_scales = (
        swizzle_nvfp4_scales(torch.cat([qkv_scales, z_scales]), (n, k))
        .numpy()
        .tobytes()
    )
    _assert_planes(payload, geometry, expected_codes, expected_scales, weight_divisor)
    _assert_activation_divisors(prepared, input_divisor)

    codes, scales, divisor = decode_nvfp4_words(payload, (n, k))
    assert torch.equal(codes, torch.cat([qkv_packed, z_packed]))
    assert torch.equal(scales, torch.cat([qkv_scales, z_scales]))
    assert divisor.numpy().tobytes() == weight_divisor

    # Row order: Q [0,128), K [128,256), V [256,384), Z [384,512).
    assert codes[0, 0].item() == 0x12
    assert codes[q_rows, 0].item() == 0x21
    assert codes[q_rows + k_rows, 0].item() == 0x30
    assert codes[qkv_rows, 0].item() == 0x04

    logical = _logical_values(codes, scales, divisor.numpy().tobytes(), k)
    assert torch.equal(logical, request.values(0, n * k).reshape(n, k))


@pytest.mark.skipif(
    not MINIMA_CHECKPOINT.is_file(),
    reason="Minima checkpoint is not present on this machine",
)
def test_minima_layer0_gdn_qkvz_nvfp4_parent_import(tmp_path):
    n = QKV_ROWS + Z_ROWS

    with SafetensorsSource(MINIMA_CHECKPOINT) as store:
        qkv = matrix_source(
            store, f"{LAYER0}.in_proj_qkv.weight", (QKV_ROWS, K)
        )
        z = matrix_source(store, f"{LAYER0}.in_proj_z.weight", (Z_ROWS, K))
        q_source = select_rows(qkv, ((0, Q_END),))
        k_source = select_rows(qkv, ((Q_END, K_END),))
        v_source = select_rows(qkv, ((K_END, QKV_ROWS),))
        prepared, request, geometry, payload = _drive_import(
            q_source, k_source, v_source, z, (n, K), tmp_path / "gdn.ninfer"
        )

        # read_flat returns a flat 1D tensor by contract; restore the source
        # shape explicitly.
        qkv_packed = store.read_flat(
            f"{LAYER0}.in_proj_qkv.weight_packed"
        ).reshape(QKV_ROWS, K // 2)
        z_packed = store.read_flat(
            f"{LAYER0}.in_proj_z.weight_packed"
        ).reshape(Z_ROWS, K // 2)
        # read_flat returns a flat 1D tensor by contract; restore the source
        # shape explicitly.
        qkv_scales = (
            store.read_flat(f"{LAYER0}.in_proj_qkv.weight_scale")
            .view(torch.uint8)
            .reshape(QKV_ROWS, K // 16)
        )
        z_scales = (
            store.read_flat(f"{LAYER0}.in_proj_z.weight_scale")
            .view(torch.uint8)
            .reshape(Z_ROWS, K // 16)
        )
        weight_divisor = store.read_flat(
            f"{LAYER0}.in_proj_qkv.weight_global_scale"
        ).view(torch.uint8).numpy().tobytes()
        z_divisor = store.read_flat(
            f"{LAYER0}.in_proj_z.weight_global_scale"
        ).view(torch.uint8).numpy().tobytes()
        input_divisor = store.read_flat(
            f"{LAYER0}.in_proj_qkv.input_global_scale"
        ).view(torch.uint8).numpy().tobytes()

    # One parent requires one shared weight divisor word across qkv and z.
    assert weight_divisor == z_divisor

    _assert_planes(
        payload,
        geometry,
        torch.cat([qkv_packed, z_packed]).numpy().tobytes(),
        swizzle_nvfp4_scales(torch.cat([qkv_scales, z_scales]), (n, K)).numpy().tobytes(),
        weight_divisor,
    )
    _assert_activation_divisors(prepared, input_divisor)

    codes, scales, divisor = decode_nvfp4_words(payload, (n, K))
    expected_codes = torch.cat([qkv_packed, z_packed])
    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales, torch.cat([qkv_scales, z_scales]))
    assert divisor.numpy().tobytes() == weight_divisor

    # Row order: Q [0,2048), K [2048,4096), V [4096,10240), Z [10240,16384).
    assert torch.equal(codes[0], qkv_packed[0])
    assert torch.equal(codes[Q_END], qkv_packed[Q_END])
    assert torch.equal(codes[K_END], qkv_packed[K_END])
    assert torch.equal(codes[QKV_ROWS], z_packed[0])

    # Round-trip: decode the parent words and compare the source's logical
    # values at one 128-row tile from each section.
    for begin in (0, Q_END, K_END, QKV_ROWS, n - 128):
        tile = _logical_values(
            codes[begin : begin + 128],
            scales[begin : begin + 128],
            divisor.numpy().tobytes(),
            K,
        )
        assert torch.equal(
            tile, request.values(begin * K, (begin + 128) * K).reshape(128, K)
        )
