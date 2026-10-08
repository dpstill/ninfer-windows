"""Select final resource bytes and derive the tokenizer's public token domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
VISION_RESOURCES = ("preprocessor_config.json", "video_preprocessor_config.json")


def token_domain(
    tokenizer: dict, config: dict, vocab_size: int
) -> tuple[int, tuple[int, ...]]:
    ids: dict[int, str] = {}
    special: dict[int, bool] = {}

    def add(index: object, content: object) -> None:
        if (
            type(index) is not int
            or not 0 <= index < vocab_size
            or not isinstance(content, str)
        ):
            raise ValueError(
                f"invalid tokenizer entry id={index!r}, content={content!r}"
            )
        if index in ids and ids[index] != content:
            raise ValueError(f"tokenizer resources disagree about token {index}")
        ids[index] = content

    def add_special(index: int, token: dict) -> None:
        flag = token.get("special", False)
        if type(flag) is not bool:
            raise ValueError(f"token {index}: special must be boolean")
        if index in special and special[index] != flag:
            raise ValueError(
                f"tokenizer resources disagree about token {index} special flag"
            )
        special[index] = flag

    model = tokenizer.get("model", {})
    vocabulary = model.get("vocab")
    if not isinstance(vocabulary, dict):
        raise ValueError("tokenizer model must provide a vocabulary mapping")
    for token, index in vocabulary.items():
        add(index, token)
    for token in tokenizer.get("added_tokens", []):
        add(token["id"], token["content"])
        add_special(token["id"], token)
    for raw_id, token in config.get("added_tokens_decoder", {}).items():
        index = int(raw_id)
        add(index, token["content"])
        add_special(index, token)
    if not ids or set(ids) != set(range(max(ids) + 1)):
        raise ValueError("this tokenizer requires a contiguous public token ID domain")
    return len(ids), tuple(sorted(index for index, flag in special.items() if flag))



_ADDED_TOKEN_FIELDS = (
    "content",
    "single_word",
    "lstrip",
    "rstrip",
    "normalized",
    "special",
)


def _normalize_tokenizer_config(tokenizer: dict, config: dict) -> dict:
    """Normalize Hugging Face tokenizer metadata for the Qwen3.5 frontend."""
    normalized = config

    def set_value(key: str, value) -> None:
        nonlocal normalized
        if normalized is config:
            normalized = dict(config)
        normalized[key] = value

    # NInfer Qwen3.5 requires no implicit BOS and no prefix-space insertion.
    # Some tokenizer exports omit these fields even though their tokenizer
    # semantics are equivalent to false.
    for field in ("add_bos_token", "add_prefix_space"):
        if field not in config:
            set_value(field, False)
            continue
        value = config[field]
        if type(value) is not bool:
            raise ValueError(f"tokenizer_config.json {field} must be boolean")
        if value:
            raise ValueError(
                f"tokenizer_config.json {field} must be false for Qwen3.5"
            )

    # NInfer Qwen3.5 frontend requires the official <|endoftext|> pad token.
    # Qwen3.8 metadata may export <|im_end|> here even though
    # generation_config.json uses pad_token_id 248044 (<|endoftext|>).
    if normalized.get("pad_token") != "<|endoftext|>":
        set_value("pad_token", "<|endoftext|>")

    if "added_tokens_decoder" in config:
        if not isinstance(config["added_tokens_decoder"], dict):
            raise ValueError(
                "tokenizer_config.json added_tokens_decoder must be an object"
            )
        return normalized

    if "added_tokens" not in tokenizer:
        return normalized

    added = tokenizer["added_tokens"]
    if not isinstance(added, list):
        raise ValueError("tokenizer.json added_tokens must be an array")

    decoder = {}
    for token in added:
        if not isinstance(token, dict):
            raise ValueError("tokenizer.json added token must be an object")
        if type(token.get("id")) is not int:
            raise ValueError("tokenizer.json added token id must be an integer")
        entry = {}
        for field in _ADDED_TOKEN_FIELDS:
            if field not in token:
                raise ValueError(
                    f"tokenizer.json added token is missing {field}"
                )
            entry[field] = token[field]
        decoder[str(token["id"])] = entry

    set_value("added_tokens_decoder", decoder)
    return normalized


def load_resources(
    model_dir: Path,
    *,
    vocab_size: int,
    vision_config: Mapping[str, int] | None = None,
    overrides: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, bytes], int, tuple[int, ...]]:
    overrides = {} if overrides is None else dict(overrides)
    roles = {"text": TEXT_RESOURCES}
    if vision_config is not None:
        roles["vision"] = VISION_RESOURCES
    allowed = {role for names in roles.values() for role in names}
    if overrides.keys() - allowed:
        raise ValueError(
            f"resource overrides have no selected consumer: {sorted(overrides.keys()-allowed)}"
        )
    references: dict[str, dict[str, str]] = {}
    payloads: dict[str, bytes] = {}
    parsed: dict[str, dict] = {}
    for component, names in roles.items():
        references[component] = {}
        for role in names:
            path = Path(overrides[role]) if role in overrides else model_dir / role
            data = path.read_bytes()
            if not data:
                raise ValueError(f"{path}: resource is empty")
            text = data.decode("utf-8")
            if role.endswith(".json"):
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: resource must contain a JSON object")
                parsed[role] = value
                if component == "vision":
                    for field, config_field in (
                        ("patch_size", "patch_size"),
                        ("temporal_patch_size", "temporal_patch_size"),
                        ("merge_size", "spatial_merge_size"),
                    ):
                        if (
                            type(value.get(field)) is not int
                            or value[field] != vision_config[config_field]
                        ):
                            raise ValueError(
                                f"{path}: {field} differs from Vision config"
                            )
            object_id = f"resource/{component}/{role}"
            references[component][role] = object_id
            payloads[object_id] = data
    tokenizer = parsed["tokenizer.json"]
    tokenizer_config = _normalize_tokenizer_config(
        tokenizer, parsed["tokenizer_config.json"]
    )

    if tokenizer_config is not parsed["tokenizer_config.json"]:
        parsed["tokenizer_config.json"] = tokenizer_config
        object_id = references["text"]["tokenizer_config.json"]
        payloads[object_id] = (
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")

    count, special = token_domain(
        tokenizer, tokenizer_config, vocab_size
    )
    return references, payloads, count, special
