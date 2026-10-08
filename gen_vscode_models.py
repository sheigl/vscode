#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyyaml",
# ]
# ///
"""
Generate the VS Code chat language models file (chatLanguageModels.json) for the
litellm custom endpoint.

This is the VS Code twin of gen_litellm_models.py. It reuses the exact same
reality-based capability logic (LiteLLM model list cross-checked against each
machine's llama-swap config) so the capabilities never drift between the two
files, then renders them in VS Code's chatLanguageModels.json shape.

Requires env vars:
    LITELLM_SERVER   e.g. https://server.tailc63ae8.ts.net:4444
    LITELLM_API_KEY  the litellm API key

Usage:
    uv run ~/.config/Code/User/gen_vscode_models.py [path/to/chatLanguageModels.json]

It imports gen_litellm_models from ~/.config/opencode/scripts (override with the
OPENCODE_SCRIPTS_DIR env var if that path differs).

The existing file is read so its top-level endpoint fields (name, vendor,
apiKey, apiType) and per-model `url` are preserved; only the `models` list is
replaced.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.parse import urlparse

# Reuse all capability logic from the opencode scripts script.
OPENCODE_SCRIPTS_DIR = os.environ.get(
    "OPENCODE_SCRIPTS_DIR",
    os.path.expanduser("~/.config/opencode/scripts"),
)
sys.path.insert(0, OPENCODE_SCRIPTS_DIR)

import gen_litellm_models as glm  # noqa: E402

DEFAULT_PATH = os.path.expanduser(
    os.path.join("~", ".config", "Code", "User", "chatLanguageModels.json")
)


def build_model_entry(model_id: str, caps: dict, defaults: dict, url: str) -> dict | None:
    """Render a single model in VS Code's chatLanguageModels.json shape."""
    if caps.get("embedding"):
        return None

    entry: dict = {
        "id": model_id,
        "name": model_id,
        "url": url,
        "toolCalling": bool(caps.get("tool_call", bool(defaults.get("tool_call", True)))),
        "vision": bool(caps.get("vision", False)),
    }

    context = caps.get("context")
    if context:
        output = caps.get("output") or int(defaults.get("max_output") or 8192)
        output = min(output, int(context))
        entry["maxInputTokens"] = max(int(context) - output, 1)
        entry["maxOutputTokens"] = output

    return entry


def resolve_url(existing: dict, server: str) -> str:
    """Prefer the `url` already in the file; fall back to the env server base."""
    models = existing.get("models") or []
    for m in models:
        if isinstance(m, dict) and m.get("url"):
            return str(m["url"])
    return server.rstrip("/") + "/v1"


def load_existing(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def build_endpoint(existing: dict, models: list[dict], server: str) -> dict:
    """Keep the endpoint's top-level fields, swap in the fresh models list."""
    url = resolve_url(existing, server)
    endpoint: dict = {
        "name": existing.get("name", "litellm"),
        "vendor": existing.get("vendor", "customendpoint"),
        "apiKey": existing.get("apiKey", "${input:chat.lm.secret.litellm}"),
        "apiType": existing.get("apiType", "chat-completions"),
        "models": models,
    }
    return endpoint


def main() -> int:
    server = os.environ.get("LITELLM_SERVER")
    api_key = os.environ.get("LITELLM_API_KEY")
    if not server or not api_key:
        print("Missing env vars. Need LITELLM_SERVER and LITELLM_API_KEY.", file=sys.stderr)
        return 1

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    mach_cfg = glm.load_machines_config()
    machines = mach_cfg.get("llama_swap", {}).get("machines", {})
    defaults = mach_cfg.get("defaults", {})
    overrides = mach_cfg.get("overrides", {})

    print(f"Fetching models from {server} ...")
    model_ids, by_name = glm.fetch_litellm_models(server, api_key)
    if not model_ids:
        print("No models returned by server.", file=sys.stderr)
        return 1

    hosts: dict[str, list[str]] = {}
    for mid in model_ids:
        host = urlparse((by_name.get(mid, {}) or {}).get("api_base") or "").hostname or ""
        hosts.setdefault(host, []).append(mid)

    entries_by_host: dict[str, dict] = {}
    for host in hosts:
        machine = machines.get(host)
        if not machine:
            continue
        text = glm.fetch_llamaswap_yaml(machine, host)
        if text is None:
            print(f"  (no llama-swap config for host {host})")
            continue
        entries_by_host[host] = glm.parse_llamaswap(text)

    existing_list = load_existing(path)
    existing = existing_list[0] if existing_list and isinstance(existing_list[0], dict) else {}
    url = resolve_url(existing, server)

    models: list[dict] = []
    skipped: list[str] = []
    for mid in sorted(model_ids):
        meta = by_name.get(mid, {})
        caps = glm.resolve_caps(mid, meta, entries_by_host)
        caps = glm.merge_caps(mid, meta, caps, machines, overrides)
        entry = build_model_entry(mid, caps, defaults, url)
        if entry is None:
            skipped.append(mid)
            continue
        models.append(entry)
        detail = (
            f"ctx={caps.get('context') or '?'}"
            f" vision={caps.get('vision', '?')}"
            f" reason={caps.get('reasoning', '?')}"
            f" tool={caps.get('tool_call', '?')}"
        )
        print(f"  - {mid:36s} {detail}")

    if skipped:
        print(f"\nSkipped non-chat models (embeddings/rerankers): {', '.join(skipped)}")

    endpoint = build_endpoint(existing, models, server)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([endpoint], fh, indent=2)
        fh.write("\n")

    print(f"\nWrote {len(models)} models to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
