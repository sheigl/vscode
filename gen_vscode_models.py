#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyyaml",
# ]
# ///
"""
Generate and apply the VS Code chat language models for the litellm custom
endpoint.

This is the VS Code twin of gen_litellm_models.py. It reuses the exact same
reality-based capability logic (LiteLLM model list cross-checked against each
machine's llama-swap config) so the capabilities never drift between the two
files, then renders them in VS Code's chatLanguageModels.json shape.

Two files, two steps:

    1. generate (default)  fetch from the server and write the models array to
       models.json, then commit and push it. models.json is the tracked,
       shareable artifact.
    2. apply                copy that models array into chatLanguageModels.json,
       leaving every other top-level field (name, vendor, apiKey, apiType, and
       anything else added by hand) alone. chatLanguageModels.json is untracked
       because its apiKey and url are machine-local.

Requires env vars (generate only):
    LITELLM_SERVER   e.g. https://server.tailc63ae8.ts.net:4444
    LITELLM_API_KEY  the litellm API key

Usage:
    uv run ~/.config/Code/User/gen_vscode_models.py [generate] [path/to/models.json]
    uv run ~/.config/Code/User/gen_vscode_models.py apply [path/to/chatLanguageModels.json]

Apply reads models.json from MODELS_JSON when that is set.

It imports gen_litellm_models from ~/.config/opencode/scripts (override with the
OPENCODE_SCRIPTS_DIR env var if that path differs).

Generate reads the url already written into either models file so a new
endpoint keeps pointing at the same server.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from urllib.parse import urlparse

# Reuse all capability logic from the opencode scripts script.
OPENCODE_SCRIPTS_DIR = os.environ.get(
    "OPENCODE_SCRIPTS_DIR",
    os.path.expanduser("~/.config/opencode/scripts"),
)

# Set by load_shared(); `apply` never touches it and never needs pyyaml.
glm = None

USER_DIR = os.path.expanduser(os.path.join("~", ".config", "Code", "User"))
DEFAULT_MODELS_PATH = os.path.join(USER_DIR, "models.json")
DEFAULT_TARGET_PATH = os.path.join(USER_DIR, "chatLanguageModels.json")


def load_shared():
    """Import gen_litellm_models on first use. Deferred so `apply` stays a pure
    file copy that cannot fail on an unrelated missing dependency."""
    global glm
    if glm is None:
        if OPENCODE_SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, OPENCODE_SCRIPTS_DIR)
        import gen_litellm_models as module  # noqa: PLC0415

        glm = module
    return glm


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


def resolve_url(paths: list[str], server: str) -> str:
    """Prefer the `url` already written into either models file; fall back to
    the env server base."""
    for path in paths:
        for m in load_models(path):
            if isinstance(m, dict) and m.get("url"):
                return str(m["url"])
    return server.rstrip("/") + "/v1"


def load_models(path: str) -> list:
    """Read a models array out of either file shape: a bare array (models.json)
    or an endpoint object whose `models` key holds it (chatLanguageModels.json).
    The two are told apart by that key, since a bare array also starts with a
    dict."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        nested = data[0].get("models")
        if isinstance(nested, list):
            return nested
    elif isinstance(data, dict):
        nested = data.get("models")
        return nested if isinstance(nested, list) else []
    return data if isinstance(data, list) else []


def load_endpoint(path: str) -> dict:
    """Read the endpoint object out of chatLanguageModels.json, or {} if the
    file is missing or unreadable (so apply can create it)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def build_endpoint(existing: dict, models: list[dict]) -> dict:
    """Keep the endpoint's top-level fields, swap in the models array."""
    endpoint: dict = {
        "name": existing.get("name", "litellm"),
        "vendor": existing.get("vendor", "customendpoint"),
        "apiKey": existing.get("apiKey", "${input:chat.lm.secret.litellm}"),
        "apiType": existing.get("apiType", "chat-completions"),
        "models": models,
    }
    # chatLanguageModels.json is yours now that it is untracked, so anything you
    # added by hand has to survive an apply. Only `models` is ours to replace.
    for key, value in existing.items():
        if key != "models" and key not in endpoint:
            endpoint[key] = value
    return endpoint


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def apply_models(models_path: str, target_path: str) -> int:
    """Copy models.json's array into chatLanguageModels.json, nothing else."""
    if not os.path.exists(models_path):
        print(
            f"{models_path} does not exist. Generate it first:\n"
            f"  uv run {os.path.basename(sys.argv[0])}",
            file=sys.stderr,
        )
        return 1

    models = load_models(models_path)
    if not models:
        print(
            f"{models_path} has no readable models array (empty or invalid JSON).",
            file=sys.stderr,
        )
        return 1
    nameless = [i for i, m in enumerate(models) if not isinstance(m, dict) or not m.get("id")]
    if nameless:
        where = f"entry {nameless[0]}" if len(nameless) == 1 else f"entries {nameless}"
        print(f"{models_path} is not a usable models array: {where} has no \"id\".", file=sys.stderr)
        return 1

    existed = os.path.exists(target_path)
    write_json(target_path, [build_endpoint(load_endpoint(target_path), models)])

    verb = "Updated" if existed else "Created"
    print(f"{verb} {target_path} with {len(models)} models from {models_path}")
    return 0


# Every shared-module call main() makes, with sample arguments that must keep
# binding. A required parameter added upstream breaks these, and check_shared_api
# names it instead of letting a TypeError surface from an unrelated line.
SHARED_CALLS: dict[str, tuple] = {
    "load_machines_config": (),
    "fetch_litellm_models": ("", ""),
    "fetch_llamaswap_yaml": ({}, ""),
    "machine_for_host": ({}, ""),
    "parse_llamaswap": ("", {}),
    "resolve_caps": ("", {}, {}),
    "merge_caps": ("", {}, None, {}, {}),
}


def drift(name: str, detail: str) -> SystemExit:
    return SystemExit(
        f"API drift: {name}() in {OPENCODE_SCRIPTS_DIR}/gen_litellm_models.py {detail}.\n"
        "Mirror the upstream change in this script before running again."
    )


def check_shared_api() -> None:
    """Fail early and legibly when gen_litellm_models.py moves under us.

    This script imports that module by path rather than by package, so nothing
    here notices when upstream renames or re-signs something until the call
    fails mid-run.
    """
    for name, sample_args in SHARED_CALLS.items():
        fn = getattr(glm, name, None)
        if fn is None:
            raise drift(name, "is gone")
        try:
            inspect.signature(fn).bind(*sample_args)
        except TypeError as exc:
            raise drift(name, f"no longer accepts these arguments ({exc})") from exc


def _git(*args: str, cwd: str) -> bool:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{result.stderr.strip()}", file=sys.stderr)
    return result.returncode == 0


def commit_and_push(target: str) -> bool:
    repo = os.path.dirname(os.path.abspath(target))
    if not _git("rev-parse", "--is-inside-work-tree", cwd=repo):
        print("Not a git repo, skipping commit/push.", file=sys.stderr)
        return False
    rel = os.path.relpath(target, repo)
    if not _git("add", "--", rel, cwd=repo):
        return False
    if not _git("diff", "--cached", "--quiet", cwd=repo):
        if not _git("commit", "-m", "update litellm models from server", cwd=repo):
            return False
    return _git("push", cwd=repo)


def parse_args(argv: list[str]) -> tuple[str, str | None]:
    """`apply` is the only subcommand; the optional path means models.json in
    generate mode and chatLanguageModels.json in apply mode."""
    args = list(argv)
    mode = "generate"
    if args and args[0] in ("apply", "generate"):
        mode = args.pop(0)
    if len(args) > 1:
        prog = os.path.basename(sys.argv[0])
        print(
            f"Unexpected argument: {args[1]}\n"
            f"Usage: {prog} [generate] [path/to/models.json]\n"
            f"       {prog} apply [path/to/chatLanguageModels.json]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return mode, (args[0] if args else None)


def main() -> int:
    mode, path = parse_args(sys.argv[1:])
    if mode == "apply":
        return apply_models(
            os.environ.get("MODELS_JSON") or DEFAULT_MODELS_PATH,
            path or DEFAULT_TARGET_PATH,
        )

    load_shared()
    check_shared_api()

    server = os.environ.get("LITELLM_SERVER")
    api_key = os.environ.get("LITELLM_API_KEY")
    if not server or not api_key:
        print("Missing env vars. Need LITELLM_SERVER and LITELLM_API_KEY.", file=sys.stderr)
        return 1

    models_path = path or DEFAULT_MODELS_PATH

    mach_cfg = glm.load_machines_config()
    machines = mach_cfg.get("llama_swap", {}).get("machines", {})
    defaults = mach_cfg.get("defaults", {})
    overrides = mach_cfg.get("overrides", {})

    print(f"Fetching models from {server} ...")
    fetched = glm.fetch_litellm_models(server, api_key)
    if not isinstance(fetched, tuple) or len(fetched) != 3:
        shape = type(fetched).__name__
        if isinstance(fetched, tuple):
            shape += f" of length {len(fetched)}"
        raise drift("fetch_litellm_models", f"returned a {shape}, expected a 3-tuple")
    model_ids, by_name, ambiguous = fetched
    if not model_ids:
        print("No models returned by server.", file=sys.stderr)
        return 1

    hosts: dict[str, list[str]] = {}
    for mid in model_ids:
        host = urlparse((by_name.get(mid, {}) or {}).get("api_base") or "").hostname or ""
        hosts.setdefault(host, []).append(mid)

    entries_by_host: dict[str, dict] = {}
    for host in hosts:
        machine = glm.machine_for_host(machines, host)
        if not machine:
            print(f"  (no machine in scripts/config.json for host {host})")
            continue
        text = glm.fetch_llamaswap_yaml(machine, host)
        if text is None:
            print(f"  (no llama-swap config for host {host})")
            continue
        # machine is what lets parse_llamaswap resolve docker mounts, launcher
        # scripts and chat templates; without it those caps are silently skipped.
        entries_by_host[host] = glm.parse_llamaswap(text, machine)

    # url already in use by either file wins over the env base, so a rename of
    # the endpoint does not silently repoint every model.
    url = resolve_url([models_path, DEFAULT_TARGET_PATH], server)

    models: list[dict] = []
    skipped: list[str] = []
    no_caps: list[str] = []
    multi_host: list[str] = []
    seen: set[str] = set()
    for mid in sorted(model_ids):
        if mid in seen:
            continue
        seen.add(mid)
        meta = by_name.get(mid, {})
        # a name litellm load-balances across two boxes has no single truth for
        # its flags; fall back to litellm metadata instead of blaming one host.
        if mid in ambiguous:
            multi_host.append(mid)
            caps = None
        else:
            caps = glm.resolve_caps(mid, meta, entries_by_host)
        caps = glm.merge_caps(mid, meta, caps, machines, overrides)
        entry = build_model_entry(mid, caps, defaults, url)
        if entry is None:
            skipped.append(mid)
            continue
        if not caps:
            no_caps.append(mid)
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
    if multi_host:
        print("\nServed by more than one machine (no flags attributed):")
        for mid in multi_host:
            print(f"  - {mid}")
        print("  Give one of these a real value in scripts/config.json 'overrides'.")
    if no_caps:
        print("\nNo capability data found (name-only output):")
        for mid in no_caps:
            print(f"  - {mid}")
        print("Add real values for those in scripts/config.json under 'overrides'.")

    write_json(models_path, models)
    print(f"\nWrote {len(models)} models to {models_path}")

    if not commit_and_push(models_path):
        return 1
    print("Committed and pushed.")
    print(f"Apply them locally with: uv run {os.path.basename(sys.argv[0])} apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
