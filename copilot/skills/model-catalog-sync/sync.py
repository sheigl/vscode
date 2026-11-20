#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Sync the model catalog: llama-swap -> the llm_router gateway -> OpenCode + VS Code.

The chain, in order:

  1. llama-swap configs on ai.home / desktop2.home / server.home  (source of truth)
  2. POST /api/v1/providers/{id}/models/preview then /models/sync on the gateway
  3. gen_litellm_models.py            -> ~/.config/opencode/opencode.jsonc, commit+push
  4. gen_vscode_models.py generate    -> ~/.config/Code/User/models.json, commit+push
     gen_vscode_models.py apply       -> machine-local chatLanguageModels.json
  5. gen_copilot.py generate          -> ~/.config/Code/User/copilot, commit+push
     gen_copilot.py apply             -> ~/.copilot agents/skills/mcp-config.json
  6. every other machine: pull both repos, apply, and copy this skill's scripts

Run this after editing any llama-swap config. It stops at the first failure, is
safe to re-run (every step is idempotent) and prints what changed.

    sync.py                 run the chain
    sync.py --dry-run       report drift only: compare each llama-swap endpoint
                            with the gateway, then preview both clients to /tmp and
                            diff them; nothing is written or committed
    sync.py --no-vscode     skip step 4

Requires LITELLM_SERVER and LITELLM_API_KEY in the environment.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

OPENCODE_DIR = Path(os.environ.get("MODEL_SYNC_OPENCODE_DIR", "~/.config/opencode")).expanduser()
VSCODE_DIR = Path(os.environ.get("MODEL_SYNC_VSCODE_DIR", "~/.config/Code/User")).expanduser()
SKILL_DIR = Path(os.environ.get("MODEL_SYNC_SKILL_DIR", "~/.agents/skills/model-catalog-sync")).expanduser()
WATCH = [w for w in os.environ.get("MODEL_SYNC_WATCH", "flash-next").split(",") if w]


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"\nFAILED: {msg}", file=sys.stderr)
    raise SystemExit(1)


def need_env() -> None:
    missing = [v for v in ("LITELLM_SERVER", "LITELLM_API_KEY") if not os.environ.get(v)]
    if missing:
        fail(f"{', '.join(missing)} not set (both generators need them)")


def http_json(url: str, timeout: int = 15, auth: bool = True):
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}"} if auth else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def litellm_models(timeout: int = 15) -> list[dict]:
    return http_json(os.environ["LITELLM_SERVER"].rstrip("/") + "/v1/models", timeout).get("data", [])


def router_get(path: str, timeout: int = 60):
    return http_json(os.environ["LITELLM_SERVER"].rstrip("/") + path, timeout)


def router_post(path: str, payload: dict, timeout: int = 900):
    """A management call that changes the gateway; a 409 is a conflict, not a retry."""
    url = os.environ["LITELLM_SERVER"].rstrip("/") + path
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        fail(f"POST {path} -> HTTP {exc.code}: {body[:600]}")


def router_providers() -> tuple[list[dict], int | None]:
    """The gateway's providers (one per llama-swap machine) and the store revision."""
    payload = router_get("/api/v1/providers")
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    return items, (payload.get("revision") if isinstance(payload, dict) else None)


def router_models() -> list[dict]:
    payload = router_get("/api/v1/models")
    return payload.get("items", []) if isinstance(payload, dict) else payload


def is_llamaswap_provider(p: dict) -> bool:
    """A provider that fronts a llama-swap: a public prefix and a /v1 base URL."""
    return bool(p.get("modelPrefix")) and "/v1" in str(p.get("baseUrl") or "")


def run(cmd: list[str], cwd: Path | None = None, quiet: bool = False) -> str:
    print(f"  $ {' '.join(cmd)}" + (f"   (in {cwd})" if cwd else ""), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out[-1200:])
        fail(f"exit {proc.returncode}: {cmd[0]}")
    if not quiet:
        for line in out.strip().splitlines()[-6:]:
            print(f"     {line}"[:200])
    return out


def capture(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def ssh_capture(host: str, script: str) -> tuple[int, str]:
    """Run a shell script on host in a login shell, so uv/git are on PATH."""
    return capture(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                    "bash -lc " + shlex.quote(script)])


def remote_hosts() -> list[str]:
    """Every machine that reaches its llama-swap config over ssh.

    Taken from scripts/config.json so a new box only has to be added there.
    MODEL_SYNC_REMOTE_HOSTS=host1,host2 overrides it for a one-off run.
    """
    env = os.environ.get("MODEL_SYNC_REMOTE_HOSTS")
    if env is not None:
        return [h.strip() for h in env.split(",") if h.strip()]
    try:
        cfg = json.loads((OPENCODE_DIR / "scripts" / "config.json").read_text())
        machines = cfg.get("llama_swap", {}).get("machines", {})
    except (OSError, json.JSONDecodeError):
        return []
    hosts: list[str] = []
    for machine in machines.values():
        host = (machine or {}).get("ssh")
        if host and host not in hosts:
            hosts.append(host)
    return hosts


PROBE = ('[ -d "$HOME/.config/opencode/.git" ] && echo opencode; '
         '[ -f "$HOME/.config/Code/User/models.json" ] && echo vscode; '
         '[ -d "$HOME/.copilot" ] && echo copilot; true')

PULL = ('git -C "$DIR" fetch --prune origin'
        ' && ( git -C "$DIR" merge --ff-only origin/HEAD'
        ' || git -C "$DIR" pull --ff-only )')


def push_dir(host: str, local: Path, remote: str) -> tuple[int, str]:
    """Copy a directory to a host, replacing its contents (tar over ssh)."""
    if not local.exists():
        return 1, f"{local} does not exist"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(local, arcname=".")
    script = f'mkdir -p {remote} && tar -xzf - -C {remote}'
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         "bash -lc " + shlex.quote(script)],
        input=buf.getvalue(), capture_output=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr).decode("utf-8", "replace")


def sync_remote(hosts: list[str], vscode: bool) -> list[str]:
    """Pull the generated files on the other machines. Best effort per host.

    Returns the list of failures; main() exits non-zero if it is not empty.
    """
    print("\n6. pulling the generated files on the other machines ...")
    if not hosts:
        print("  no remote machines configured")
        return []
    failures: list[str] = []
    for host in hosts:
        rc, out = ssh_capture(host, PROBE)
        if rc != 0:
            print(f"  {host}: unreachable")
            for line in out.strip().splitlines()[-2:]:
                print(f"     {line}"[:200])
            failures.append(host)
            continue
        have = set(out.split())
        print(f"  {host}:")
        if "opencode" in have:
            rc, out = ssh_capture(host, PULL.replace("$DIR", "$HOME/.config/opencode"))
            print(f"    OpenCode  {'ok' if rc == 0 else 'FAILED'}")
            if rc:
                failures.append(f"{host}:opencode")
                for line in out.strip().splitlines()[-3:]:
                    print(f"      {line}"[:200])
        else:
            print("    OpenCode  no checkout, skipped")

        # The sync scripts live outside every repo, so copy them explicitly.
        rc, out = push_dir(host, SKILL_DIR, "$HOME/.agents/skills/model-catalog-sync")
        print(f"    Scripts   {'ok' if rc == 0 else 'FAILED'}")
        if rc:
            failures.append(f"{host}:scripts")
            for line in out.strip().splitlines()[-3:]:
                print(f"      {line}"[:200])

        if not vscode:
            continue
        if "vscode" in have:
            rc, out = ssh_capture(
                host,
                PULL.replace("$DIR", "$HOME/.config/Code/User")
                + ' && cd "$HOME/.config/Code/User" && uv run gen_vscode_models.py apply',
            )
            print(f"    VS Code   {'ok' if rc == 0 else 'FAILED'}")
            if rc:
                failures.append(f"{host}:vscode")
                for line in out.strip().splitlines()[-3:]:
                    print(f"      {line}"[:200])
        else:
            print("    VS Code   no models.json, skipped")

        if "vscode" in have:
            rc, out = ssh_capture(
                host,
                'cd "$HOME/.config/Code/User"'
                ' && if [ -f gen_copilot.py ]; then uv run gen_copilot.py apply;'
                ' else echo "no gen_copilot.py in this checkout"; fi',
            )
            print(f"    Copilot   {'ok' if rc == 0 else 'FAILED'}")
            if rc:
                failures.append(f"{host}:copilot")
                for line in out.strip().splitlines()[-3:]:
                    print(f"      {line}"[:200])
        elif "copilot" in have:
            rc, out = push_dir(host, VSCODE_DIR / "copilot", "$HOME/.copilot")
            print(f"    Copilot   {'ok' if rc == 0 else 'FAILED'}")
            if rc:
                failures.append(f"{host}:copilot")
                for line in out.strip().splitlines()[-3:]:
                    print(f"      {line}"[:200])
        else:
            print("    Copilot   no ~/.copilot, skipped")
    return failures


def opencode_view(path: Path) -> dict[str, dict]:
    """Parse the model blocks of an opencode.jsonc."""
    txt = path.read_text()
    out: dict[str, dict] = {}
    for key, body in re.findall(r'^\s{8}"([^"]+)":\s*\{(.*?)\n\s{8}\}', txt, re.M | re.S):
        lim = re.search(r'"limit":\s*\{\s*"context":\s*(\d+),\s*"input":\s*(\d+),\s*"output":\s*(\d+)', body)
        out[key] = {
            "context": int(lim.group(1)) if lim else None,
            "input": int(lim.group(2)) if lim else None,
            "output": int(lim.group(3)) if lim else None,
            "vision": '"image"' in body,
            "tool_call": re.search(r'"tool_call":\s*true', body) is not None,
            "reasoning": re.search(r'"reasoning":\s*true', body) is not None,
        }
    return out


def opencode_model_keys(path: Path | None = None) -> set[str]:
    return set(opencode_view(path or OPENCODE_DIR / "opencode.jsonc"))


def opencode_limits(path: Path | None = None) -> list[tuple[str, int, int, int]]:
    view = opencode_view(path or OPENCODE_DIR / "opencode.jsonc")
    return [(k, v["context"], v["input"], v["output"]) for k, v in view.items()
            if v["context"] is not None]


def vscode_view(path: Path | None = None) -> dict[str, dict]:
    path = path or VSCODE_DIR / "models.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    models = data.get("models", []) if isinstance(data, dict) else data
    return {
        (m.get("name") or m.get("id")): {
            "input": m.get("maxInputTokens"),
            "output": m.get("maxOutputTokens"),
            "vision": m.get("vision"),
            "tool_call": m.get("toolCalling"),
        }
        for m in models
    }


def vscode_model_keys(path: Path | None = None) -> set[str]:
    return set(vscode_view(path))


def print_limit_sanity(path: Path, note: str = "") -> None:
    print(f"\nlimit sanity (OpenCode{note})")
    limits = opencode_limits(path)
    bad = [(k, c, i) for k, c, i, o in limits if i <= 1 or o >= c]
    if bad:
        for k, c, i in bad:
            print(f"  !! {k}: context {c}, input {i} — OpenCode will compact immediately")
        print("     fix with an override in ~/.config/opencode/scripts/config.json")
    else:
        print("  ok: no entry with input<=1 or output>=context")
    for k, c, i, o in limits:
        if any(w.lower() in k.lower() for w in WATCH):
            print(f"  {k}: context {c}, input {i}, output {o}")


def print_remote(remote: list[str], failures: list[str] | None) -> None:
    print("\nother machines (step 6, over ssh)")
    if failures is None:
        print(f"  would pull/apply on: {', '.join(remote) or '(none)'}   (dry run)")
    elif failures:
        print(f"  FAILED: {', '.join(failures)}")
        print("  fix the host and re-run sync.py (every step is idempotent)")
    else:
        print(f"  updated: {', '.join(remote) or '(none)'}")


def print_router_drift() -> None:
    """Router vs llama-swap: the ids it serves, and the metadata the catalog needs."""
    print("\nRouter vs llama-swap (source of truth)")
    try:
        providers, _ = router_providers()
        models = router_models()
    except Exception as exc:
        print(f"  could not read the router ({str(exc)[:120]}); check skipped")
        return
    clean = True
    for p in providers:
        if not is_llamaswap_provider(p):
            continue
        prefix = p["modelPrefix"]
        base = str(p["baseUrl"]).rsplit("/v1", 1)[0]
        try:
            served = {m["id"] for m in http_json(base + "/v1/models", 30, auth=False).get("data", [])}
        except Exception as exc:
            print(f"  {prefix}: unreachable ({str(exc)[:100]})")
            clean = False
            continue
        here = [m for m in models if m.get("providerId") == p["id"]]
        have = {str(m.get("upstreamModel")) for m in here}
        missing = sorted(served - have)
        stale = sorted(have - served)
        drift = f"+{len(missing)} -{len(stale)}" if (missing or stale) else "ok"
        print(f"  {prefix}: {len(served)} served, {len(served & have)} in the router   {drift}")
        for i in missing:
            print(f"    + {prefix}{i}")
        for i in stale:
            print(f"    - {prefix}{i}")
        if missing or stale:
            clean = False
        gaps = [m.get("alias") for m in here
                if not (m.get("contextLength") and m.get("maxOutputTokens"))]
        if gaps:
            print(f"    ~ {len(gaps)} alias(es) still without context and/or max output"
                  f" (e.g. {gaps[0]}); a sync that reports them fixes that")
    if clean:
        print("  ok: every llama-swap id is in the router")


def plan_lists(plan: dict) -> tuple[list, list, list, list, list]:
    """added/updated/removed/blocked/missing, whichever way the plan spells them.

    A build may report counts (additions/updates) with an items list that carries
    the action, or the newer added/updated/removed arrays; accept both.
    """
    items = [i for i in (plan.get("items") or []) if isinstance(i, dict)]
    def pick(key, action):
        got = plan.get(key)
        if isinstance(got, list):
            return got
        return [i for i in items if i.get("action") == action]
    return (pick("added", "add"), pick("updated", "update"), pick("removed", "remove"),
            plan.get("blocked") or [], plan.get("missing") or [])


def sync_revision(applied, previous):
    """The sync response's revision may be a number or the whole store snapshot."""
    rev = (applied or {}).get("revision", previous)
    return rev.get("revision", previous) if isinstance(rev, dict) else rev


def sync_router() -> None:
    """Each machine provider: preview, then apply.

    Apply re-discovers rather than replaying the preview, so its plan is the
    authoritative one. Prune is unconditional, so a provider whose inventory
    came back empty would delete its models; the router refuses a zero-model
    discovery as an error, which is why a failure here stops the run.
    """
    providers, revision = router_providers()
    if revision is None:
        fail("GET /api/v1/providers returned no revision")
    targets = [p for p in providers if is_llamaswap_provider(p)]
    if not targets:
        fail("no llama-swap providers on the router")
    for p in targets:
        print(f"\n  {p['modelPrefix']:<10} {p.get('name')}")
        base = str(p["baseUrl"]).rsplit("/v1", 1)[0]
        try:
            served = http_json(base + "/v1/models", 30, auth=False).get("data", [])
        except Exception as exc:
            print(f"     skipped: unreachable ({str(exc)[:90]})")
            continue
        if not served:
            print("     skipped: the inventory came back empty; a sync would prune every model")
            continue
        preview = router_post(f"/api/v1/providers/{p['id']}/models/preview", {"revision": revision})
        plan = (preview or {}).get("plan") or {}
        add, upd, rem, blocked, missing = plan_lists(plan)
        print(f"     preview: discovered {plan.get('discovered', '?')}"
              f"  +{len(add) or plan.get('additions', 0)}"
              f" ~{len(upd) or plan.get('updates', 0)}"
              f" -{len(missing)}"
              + (f"  blocked {len(blocked)}" if blocked else ""))
        applied = router_post(f"/api/v1/providers/{p['id']}/models/sync", {"revision": revision})
        plan = (applied or {}).get("plan") or {}
        revision = sync_revision(applied, revision)
        add, upd, rem, blocked, missing = plan_lists(plan)
        print(f"     applied: +{len(add)} ~{len(upd)} -{len(rem)}"
              f"  unchanged {plan.get('unchanged', '?')}  (revision {revision})")
        for it in add:
            print(f"       + {it.get('alias')}")
        for it in upd:
            ch = it.get("changes")
            print(f"       ~ {it.get('alias')}" + (f": {', '.join(ch)}" if ch else ""))
        for it in rem:
            print(f"       - {it.get('alias')}  (upstream {it.get('upstreamModel')})")
        for it in blocked:
            print(f"       ! {it.get('alias')}: {it.get('reason')}")
        for w in plan.get("warnings") or []:
            print(f"       warning: {w}")


def make_preview(cmd: list[str], cwd: Path, target: Path, label: str) -> bool:
    """Generate a catalog file to a temp path. The generators stage only their
    own target, so a temp path outside a repo writes the file and stops at the
    skipped commit; that exit code is expected and ignored."""
    rc, out = capture(cmd + [str(target)], cwd=cwd)
    if target.exists():
        return True
    print(f"\n{label}\n  could not generate a preview (exit {rc})")
    for line in out.strip().splitlines()[-3:]:
        print(f"    {line}"[:200])
    return False


def print_view_diff(title: str, live: dict[str, dict], preview: dict[str, dict]) -> None:
    added = sorted(set(preview) - set(live))
    removed = sorted(set(live) - set(preview))
    changed = sorted(k for k in set(live) & set(preview) if live[k] != preview[k])
    if not (added or removed or changed):
        print(f"\n{title}\n  models: {len(live)}   (no change)")
        return
    print(f"\n{title}\n  models: {len(live)} -> {len(preview)}   "
          f"+{len(added)} -{len(removed)}" + (f" ~{len(changed)}" if changed else ""))
    for k in added:
        print(f"    + {k}")
    for k in removed:
        print(f"    - {k}")
    for k in changed:
        bits = [f"{f} {live[k].get(f)} -> {preview[k].get(f)}"
                for f in ("context", "input", "output", "vision", "tool_call", "reasoning")
                if live[k].get(f) != preview[k].get(f)]
        print(f"    ~ {k}" + (": " + ", ".join(bits) if bits else ""))


def dry_run_report(remote: list[str]) -> None:
    print("\n2. checking for drift (nothing is written) ...")
    print_router_drift()

    base = Path("/tmp/opencode")
    tmp = Path(tempfile.mkdtemp(prefix="model-catalog-dry-",
                                dir=str(base) if base.is_dir() else None))
    generated = None
    try:
        oc_target = tmp / "opencode.jsonc"
        if make_preview(["uv", "run", "scripts/gen_litellm_models.py"],
                        OPENCODE_DIR, oc_target, "OpenCode"):
            generated = oc_target
            print_view_diff("OpenCode (~/.config/opencode/opencode.jsonc)",
                            opencode_view(OPENCODE_DIR / "opencode.jsonc"),
                            opencode_view(oc_target))

        vs_target = tmp / "models.json"
        if make_preview(["uv", "run", "gen_vscode_models.py"],
                        VSCODE_DIR, vs_target, "VS Code"):
            print_view_diff("VS Code (~/.config/Code/User/models.json)",
                            vscode_view(), vscode_view(vs_target))

        print_limit_sanity(generated or OPENCODE_DIR / "opencode.jsonc",
                           ", generated" if generated else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print_remote(remote, None)


def report(litellm: list[dict], before: set[str], oc_before: set[str], vs_before: set[str],
           remote: list[str], remote_failures: list[str] | None = None) -> None:
    ids = {m["id"] for m in litellm}
    added, removed = sorted(ids - before), sorted(before - ids)
    print("\nrouter")
    print(f"  models: {len(ids)}" + (f"   +{len(added)} -{len(removed)}" if (added or removed) else "   (no change)"))
    for i in added:
        print(f"    + {i}")
    for i in removed:
        print(f"    - {i}")

    oc_now, vs_now = opencode_model_keys(), vscode_model_keys()
    print("\nOpenCode (~/.config/opencode/opencode.jsonc)")
    print(f"  models: {len(oc_now)}" + (f"   +{len(oc_now - oc_before)} -{len(oc_before - oc_now)}"
                                        if oc_now != oc_before else "   (no change)"))
    missing = sorted(i for i in (ids - oc_now)
                     if not re.search(r"embed|rerank", i, re.I))
    if missing:
        print(f"  in the router but not generated: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if not vs_before and not vs_now:
        print("\nVS Code: models.json not found, skipped")
    else:
        print(f"\nVS Code (~/.config/Code/User/models.json)")
        print(f"  models: {len(vs_now)}" + (f"   +{len(vs_now - vs_before)} -{len(vs_before - vs_now)}"
                                            if vs_now != vs_before else "   (no change)"))

    print_limit_sanity(OPENCODE_DIR / "opencode.jsonc")
    print_remote(remote, remote_failures)


def main() -> int:
    dry = "--dry-run" in sys.argv
    skip_vscode = "--no-vscode" in sys.argv
    need_env()

    print("1. reading the current catalog ...")
    try:
        before_models = litellm_models()
    except Exception as exc:
        fail(f"cannot read the gateway ({exc}); is the router up? try without --dry-run")
    before = {m["id"] for m in before_models}
    oc_before, vs_before = opencode_model_keys(), vscode_model_keys()
    remote = remote_hosts()
    print(f"  router {len(before)} models | OpenCode {len(oc_before)} | VS Code {len(vs_before)}")

    if dry:
        dry_run_report(remote)
        print("\n(dry run: nothing was regenerated)")
        return 0

    print("\n2. syncing the router from llama-swap ...")
    sync_router()

    print("\n3. confirming the gateway list ...")
    after_models = litellm_models()
    added = {m["id"] for m in after_models} - before
    print(f"  up: {len(after_models)} models" + (f"  (+{len(added)} new)" if added else ""))

    print("\n4. regenerating the OpenCode config ...")
    if not (OPENCODE_DIR / "scripts/gen_litellm_models.py").exists():
        fail(f"no scripts/gen_litellm_models.py under {OPENCODE_DIR}")
    run(["uv", "run", "scripts/gen_litellm_models.py"], cwd=OPENCODE_DIR)

    have_vscode = (VSCODE_DIR / "gen_vscode_models.py").exists()
    if skip_vscode:
        print("\n5. VS Code: skipped (--no-vscode)")
    elif have_vscode:
        print("\n5. regenerating the VS Code config ...")
        run(["uv", "run", "gen_vscode_models.py"], cwd=VSCODE_DIR)
        run(["uv", "run", "gen_vscode_models.py", "apply"], cwd=VSCODE_DIR)
        vs_before = vs_before
    else:
        print(f"\n5. VS Code: no gen_vscode_models.py under {VSCODE_DIR}, skipped")

    if skip_vscode:
        print("\n5b. Copilot: skipped (--no-vscode)")
    elif (VSCODE_DIR / "gen_copilot.py").exists():
        print("\n5b. translating the OpenCode customizations for Copilot ...")
        run(["uv", "run", "gen_copilot.py"], cwd=VSCODE_DIR)
        run(["uv", "run", "gen_copilot.py", "apply"], cwd=VSCODE_DIR)
    else:
        print(f"\n5b. Copilot: no gen_copilot.py under {VSCODE_DIR}, skipped")

    failures = sync_remote(remote, vscode=not skip_vscode and have_vscode)
    report(after_models, before, oc_before, vs_before, remote, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
