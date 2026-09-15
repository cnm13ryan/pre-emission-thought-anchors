"""``preanchor agent ...`` and ``preanchor lease ...`` Typer sub-apps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from preanchor.control import TOKENS_ENV, ControlError, control_root, presented_tokens
from preanchor.control import leases as lease_module
from preanchor.control import sessions as session_module
from preanchor.control.leases import LeaseHeld, LeaseRequired
from preanchor.control.receipts import list_receipts
from preanchor.logging import redact

agent_app = typer.Typer(no_args_is_help=True, help="Agent session registry (operational).")
lease_app = typer.Typer(no_args_is_help=True,
                        help="Persistent single-owner leases on shared resources.")

RootOption = Annotated[Path | None, typer.Option(help="Persistent root (default PREANCHOR_ROOT).")]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable output only.")]
SessionOption = Annotated[str, typer.Option("--session", help="Session id.")]
TokenOption = Annotated[str | None, typer.Option(
    "--token", help=f"Owner token (default: the resource's entry in {TOKENS_ENV}).")]
HELD, REQUIRED = 3, 4    # distinct exit codes so wrappers can tell contention from misuse


def emit(value: dict | list, as_json: bool, text: str | None = None, *,
         raw: bool = False) -> None:
    if as_json or text is None:
        typer.echo(json.dumps(value if raw else redact(value), sort_keys=True, indent=2))
    else:
        typer.echo(text)


def fail(exc: Exception, as_json: bool, code: int = 1) -> None:
    if as_json:
        typer.echo(json.dumps({"ok": False, "error": type(exc).__name__, "detail": str(exc),
                               **({"owner": getattr(exc, "info", None)}
                                  if isinstance(exc, LeaseHeld) else {})}, sort_keys=True))
    else:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
    raise typer.Exit(code) from None


def _token(resource: str, token: str | None) -> str:
    token = token or presented_tokens().get(resource)
    if not token:
        raise ControlError(f"No owner token for {resource}: pass --token or set {TOKENS_ENV}")
    return token


# ---- agent ----------------------------------------------------------------------------

@agent_app.command("register")
def agent_register(
    task: Annotated[str, typer.Option(help="What this session is doing.")],
    role: Annotated[str, typer.Option(help="implementation | tests | review | backup | ...")],
    worktree: Annotated[Path, typer.Option(help="Checkout this session owns.")] = Path("."),
    allow: Annotated[list[str] | None, typer.Option(help="Allowed write root; repeatable.")]
    = None,
    forbid: Annotated[list[str] | None, typer.Option(help="Forbidden write root; repeatable.")]
    = None,
    base_commit: Annotated[str | None, typer.Option()] = None,
    root: RootOption = None, as_json: JsonOption = False,
) -> None:
    """Register one session for one worktree/branch; prints its session id."""
    try:
        record = session_module.register(control_root(root), task=task, role=role,
                                         worktree=worktree, allowed_write_roots=allow,
                                         forbidden_write_roots=forbid, base_commit=base_commit)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(record, as_json, f"registered {record['session_id']} on {record['branch']} "
         f"({record['worktree']})")


@agent_app.command("heartbeat")
def agent_heartbeat(session: SessionOption, note: Annotated[str | None, typer.Option()] = None,
                    root: RootOption = None, as_json: JsonOption = False) -> None:
    """Refresh the session heartbeat and its current commit."""
    try:
        record = session_module.heartbeat(control_root(root), session, note=note)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(record, as_json, f"heartbeat {record['session_id']} @ {record['heartbeat']}")


@agent_app.command("status")
def agent_status(stale_after: Annotated[float, typer.Option(min=1)] = 300.0,
                 include_closed: Annotated[bool, typer.Option()] = False,
                 root: RootOption = None, as_json: JsonOption = False) -> None:
    """Active and stale sessions with their branch/worktree, plus every lease."""
    try:
        control = control_root(root)
        rows = session_module.status(control, stale_after=stale_after,
                                     include_closed=include_closed)
        leases = lease_module.list_leases(control, stale_after=stale_after)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    text = session_module.render_status(rows) + "\nLeases:" + (
        "".join("\n  " + lease_module.render_owner(i).replace("\n", "\n    ") for i in leases)
        or "\n  (none)")
    emit({"sessions": rows, "leases": leases}, as_json, text)


@agent_app.command("close")
def agent_close(session: SessionOption, outcome: Annotated[str, typer.Option()] = "closed",
                root: RootOption = None, as_json: JsonOption = False) -> None:
    """Close one session (never touches any other session or its leases)."""
    try:
        record = session_module.close(control_root(root), session, outcome=outcome)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(record, as_json, f"closed {record['session_id']} ({outcome})")


@agent_app.command("receipts")
def agent_receipts(kind: Annotated[str | None, typer.Option()] = None,
                   subject: Annotated[str | None, typer.Option()] = None,
                   root: RootOption = None, as_json: JsonOption = False) -> None:
    """List immutable lifecycle receipts."""
    try:
        rows = list_receipts(control_root(root), kind=kind, subject=subject)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(rows, as_json, "\n".join(f"{r['path']}  {r['kind']}  {r['subject']}" for r in rows)
         or "(no receipts)")


# ---- lease ----------------------------------------------------------------------------

@lease_app.command("list")
def lease_list(stale_after: Annotated[float, typer.Option(min=1)] = 300.0,
               root: RootOption = None, as_json: JsonOption = False) -> None:
    """Every held, stale or incomplete lease."""
    try:
        rows = lease_module.list_leases(control_root(root), stale_after=stale_after)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(rows, as_json, "\n\n".join(lease_module.render_owner(r) for r in rows) or "(no leases)")


@lease_app.command("acquire")
def lease_acquire(resource: Annotated[str, typer.Argument()], session: SessionOption,
                  purpose: Annotated[str, typer.Option()],
                  env: Annotated[bool, typer.Option(
                      "--env", help=f"Print `export {TOKENS_ENV}=...` for eval.")] = False,
                  root: RootOption = None, as_json: JsonOption = False) -> None:
    """Take a resource for this session. Exactly one concurrent caller wins (exit 3 otherwise)."""
    try:
        record = lease_module.acquire(control_root(root), resource, session_id=session,
                                      purpose=purpose)
    except LeaseHeld as exc:
        fail(exc, as_json, HELD)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    if env:
        existing = presented_tokens()
        existing[resource] = record["owner_token"]
        typer.echo(f"export {TOKENS_ENV}=" + ",".join(f"{k}={v}" for k, v in existing.items()))
        return
    emit(record, as_json, f"acquired {resource} for {session}\nowner_token: "
         f"{record['owner_token']}\nexport {TOKENS_ENV}={resource}={record['owner_token']}",
         raw=True)


@lease_app.command("heartbeat")
def lease_heartbeat(resource: Annotated[str, typer.Argument()], token: TokenOption = None,
                    root: RootOption = None, as_json: JsonOption = False) -> None:
    """Refresh the lease heartbeat (owner token required)."""
    try:
        info = lease_module.heartbeat(control_root(root), resource, _token(resource, token))
    except LeaseRequired as exc:
        fail(exc, as_json, REQUIRED)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(info, as_json, f"heartbeat {resource} @ {info['heartbeat_at']}")


@lease_app.command("release")
def lease_release(resource: Annotated[str, typer.Argument()], token: TokenOption = None,
                  reason: Annotated[str | None, typer.Option()] = None,
                  root: RootOption = None, as_json: JsonOption = False) -> None:
    """Give a resource back (owner token required; nobody else can release it)."""
    try:
        record = lease_module.release(control_root(root), resource, _token(resource, token),
                                      reason=reason)
    except LeaseRequired as exc:
        fail(exc, as_json, REQUIRED)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(record, as_json, f"released {resource}\nreceipt: {record['receipt']}")


@lease_app.command("reclaim")
def lease_reclaim(resource: Annotated[str, typer.Argument()], session: SessionOption,
                  reason: Annotated[str, typer.Option(help="Why the holder is gone.")],
                  force: Annotated[bool, typer.Option(
                      help="Override an ambiguous (other host / unknown pid) stale lease.")]
                  = False,
                  stale_after: Annotated[float, typer.Option(min=1)] = 300.0,
                  root: RootOption = None, as_json: JsonOption = False) -> None:
    """Audited end of someone else's lease; ownership is never transferred."""
    try:
        record = lease_module.reclaim(control_root(root), resource, session_id=session,
                                      reason=reason, force=force, stale_after=stale_after)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(record, as_json, f"reclaimed {resource} (not owned by anyone now)\n"
         f"receipt: {record['receipt']}")


@lease_app.command("check")
def lease_check(resources: Annotated[list[str], typer.Argument()],
                root: RootOption = None, as_json: JsonOption = False) -> None:
    """Exit 0 only if every resource is leased and its token is in PREANCHOR_LEASE_TOKENS.
    Used by scripts/start_vllm.sh before any GPU or Docker action."""
    try:
        owners = lease_module.require(resources, root=root)
    except LeaseRequired as exc:
        fail(exc, as_json, REQUIRED)
    except (ControlError, OSError) as exc:
        fail(exc, as_json)
    emit(owners, as_json, "\n".join(f"{r}: held by {o['session_id']}" for r, o in owners.items()))
