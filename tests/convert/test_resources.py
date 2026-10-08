from __future__ import annotations

import json

import pytest

from tools.convert.resources import (
    _normalize_tokenizer_config,
    load_resources,
    token_domain,
)


def test_final_resources_override_defaults_without_hash_pinning(tmp_path):
    (tmp_path / "tokenizer.json").write_text("invalid default")
    selected = tmp_path / "custom-tokenizer.json"
    selected.write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
        )
    )
    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id":2,"temperature":1.0}'
    )
    template = tmp_path / "custom.jinja"
    template.write_text("custom {{ messages }}")
    references, payloads, count, special = load_resources(
        tmp_path,
        vocab_size=4,
        overrides={"tokenizer.json": selected, "chat_template.jinja": template},
    )
    assert count == 3 and special == (2,)
    assert set(references) == {"text"}
    assert payloads[references["text"]["chat_template.jinja"]] == template.read_bytes()
    assert (
        payloads[references["text"]["generation_config.json"]]
        == (tmp_path / "generation_config.json").read_bytes()
    )


def test_special_tokens_merge_both_resources_with_consistent_flags():
    tokenizer = {
        "model": {"vocab": {"a": 0}},
        "added_tokens": [{"id": 1, "content": "<start>", "special": True}],
    }
    config = {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
    assert token_domain(tokenizer, config, 4) == (3, (1, 2))
    config["added_tokens_decoder"]["1"] = {"content": "<start>", "special": False}
    with pytest.raises(ValueError, match="special flag"):
        token_domain(tokenizer, config, 4)


def test_tokenizer_config_derives_added_tokens_decoder_when_missing():
    tokenizer = {
        "added_tokens": [
            {
                "id": 2,
                "content": "<end>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 3,
                "content": "<think>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": False,
            },
        ]
    }
    config = {"eos_token": "<end>"}

    normalized = _normalize_tokenizer_config(tokenizer, config)

    assert normalized is not config
    assert "added_tokens_decoder" not in config
    assert normalized["added_tokens_decoder"] == {
        "2": {
            "content": "<end>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        "3": {
            "content": "<think>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        },
    }

    # Existing valid metadata must be preserved, not regenerated.
    assert _normalize_tokenizer_config(tokenizer, normalized) is normalized


def test_tokenizer_config_without_added_tokens_gets_prefix_semantics():
    tokenizer = {
        "model": {
            "vocab": {
                "a": 0,
                "b": 1,
            }
        }
    }
    config = {}

    normalized = _normalize_tokenizer_config(tokenizer, config)

    assert normalized is not config
    assert config == {}
    assert normalized["add_bos_token"] is False
    assert normalized["add_prefix_space"] is False
    assert "added_tokens_decoder" not in normalized


def test_tokenizer_config_normalizes_qwen_pad_token():
    tokenizer = {
        "added_tokens": [
            {
                "id": 248044,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ]
    }
    config = {
        "add_prefix_space": False,
        "pad_token": "<|im_end|>",
    }

    normalized = _normalize_tokenizer_config(tokenizer, config)

    assert config["pad_token"] == "<|im_end|>"
    assert normalized["add_bos_token"] is False
    assert normalized["add_prefix_space"] is False
    assert normalized["pad_token"] == "<|endoftext|>"
    assert normalized["added_tokens_decoder"]["248044"]["content"] == "<|endoftext|>"
