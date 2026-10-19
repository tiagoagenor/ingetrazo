# SPDX-License-Identifier: GPL-3.0-or-later
"""The History recorder (golden/SCHEMA.md; docs/rust-rewrite-plan.md, M0.4).

``install()`` wraps every ``Command`` subclass so that, while recording:

- its constructor arguments are kept (``to_op``), one-shot iterators
  turned into lists first so the command sees the same values;
- when it runs (``do``), the arguments are encoded against the scene as it
  is at that moment — live faces, edges and groups become positional
  selectors — and inner commands it runs become nested ops;
- ``History.execute``/``undo``/``redo`` append steps to the history's
  script, with the L2 signature after each one.

``from_op`` rebuilds a command from its op against the scene as it is when
it runs, and ``replay`` runs a whole script on a fresh scene, so a script
can be checked in Python before Rust ever reads it. Nothing is wrapped
until ``install()``; the app itself never imports this module.
"""
from __future__ import annotations

import functools
import inspect
import re
import tempfile
from pathlib import Path

from tests.golden.signature import scene_signature, signature_hash
from tests.golden.values import Decoder, Encoder, NotSerializable

SCHEMA = 1

_installed = False
_state = {"active": 0, "paused": 0}
_frames: list = []          # one per command currently inside do()
#: Commands that already ran in the current history step: id → (cmd, op).
_ran: dict = {}
#: Every History created since the last ``take_histories()``.
_histories: list = []

_ARGS = "_golden_args"
#: Commands that carry a state, not an operation: a closure, or the
#: before/after snapshots of one already applied (push/pull's commit).
NO_OP = {
    "SnapshotMutation": "a snapshot of a tool's mutation (no op until the tool tags it)",
    "SnapshotImport": "a snapshot of an import (no op until the importer tags it)",
}
_OP = "_golden_op"


def op_name(cls) -> str:
    name = cls.__name__
    if name.endswith("Command"):
        name = name[: -len("Command")]
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower().lstrip("_")


def _all_commands():
    from core.history import Command

    out, stack = [], [Command]
    while stack:
        c = stack.pop()
        for s in c.__subclasses__():
            if s not in out:
                out.append(s)
                stack.append(s)
    return out


_imported = False


def _import_commands() -> None:
    """Import every module that may define a Command (once), so a script
    read from disk finds classes its test never imported."""
    global _imported
    if _imported:
        return
    import importlib
    import pkgutil

    for pkg in ("core", "tools", "plugins"):
        try:
            mod = importlib.import_module(pkg)
        except Exception:  # noqa: BLE001 — a package that cannot load defines nothing
            continue
        for info in pkgutil.walk_packages(mod.__path__, pkg + "."):
            try:
                importlib.import_module(info.name)
            except Exception:  # noqa: BLE001
                pass
    _imported = True


def registry() -> dict:
    """op name → class (an ambiguous name maps to None)."""
    _import_commands()
    reg: dict = {}
    for cls in _all_commands():
        n = op_name(cls)
        reg[n] = None if n in reg and reg[n] is not cls else cls
    return reg


# ---- capture ---------------------------------------------------------------

def _one_shot(v) -> bool:
    if isinstance(v, (str, bytes, list, tuple, dict, set, frozenset)):
        return False
    try:
        return iter(v) is v
    except TypeError:
        return False


def _wrap_init(cls):
    original = cls.__dict__["__init__"]
    if getattr(original, "_golden", False):
        return
    sig = inspect.signature(original)

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if _state["active"] and not hasattr(self, _ARGS):
            try:
                bound = sig.bind(self, *args, **kwargs)
            except TypeError:
                return original(self, *args, **kwargs)
            for k, v in list(bound.arguments.items()):
                if _one_shot(v):
                    bound.arguments[k] = list(v)
            args, kwargs = bound.args[1:], bound.kwargs
            named = dict(bound.arguments)
            named.pop(next(iter(sig.parameters)))          # self
            try:
                object.__setattr__(self, _ARGS, (sig, named))
            except AttributeError:
                pass
        return original(self, *args, **kwargs)

    __init__._golden = True
    cls.__init__ = __init__


class _Frame:
    def __init__(self, cmd) -> None:
        self.cmd = cmd
        self.inner_ops: dict = {}


def _encode_args(cmd, scene):
    """The op of ``cmd`` against ``scene`` now; command-valued arguments
    stay pending until the inner commands have run."""
    if type(cmd).__name__ in NO_OP:
        raise NotSerializable(NO_OP[type(cmd).__name__])
    captured = getattr(cmd, _ARGS, None)
    if captured is None:
        raise NotSerializable(f"{type(cmd).__name__}: built before recording")
    _sig, named = captured
    from core.history import Command

    pending = []

    class _Deferred(Encoder):
        def encode(self, v):
            if isinstance(v, Command) and id(v) in _ran:
                # A sibling that already ran (AutoTagDrawnFaces reads the
                # faces its AddFace siblings made): the same instance.
                ran_op = _ran[id(v)][1]
                if "rid" not in ran_op:
                    ran_op["rid"] = sum(1 for _c, o in _ran.values() if "rid" in o)
                return {"$ran": ran_op["rid"]}
            if isinstance(v, Command):
                box = {"$cmd": None}
                pending.append((box, v))
                return box
            if isinstance(v, (type({}.keys()), type({}.values()))):
                return [self.encode(x) for x in v]
            if callable(v) and not hasattr(v, "to_dict") and inspect.isroutine(v):
                raise NotSerializable("a function argument (closure)")
            return super().encode(v)

    enc = _Deferred(scene)
    args = {k: enc.encode(v) for k, v in named.items()}
    return {"op": op_name(type(cmd)), "args": args}, pending


def _wrap_do(cls):
    original = cls.__dict__["do"]
    if getattr(original, "_golden", False):
        return

    @functools.wraps(original)
    def do(self, scene):
        if not _state["active"] or _state["paused"] or any(f.cmd is self for f in _frames):
            return original(self, scene)
        try:
            op, pending = _encode_args(self, scene)
        except NotSerializable as exc:
            op, pending = {"op": op_name(type(self)), "error": str(exc)}, []
        frame = _Frame(self)
        _frames.append(frame)
        try:
            result = original(self, scene)
        finally:
            _frames.pop()
        for box, inner in pending:
            inner_op = frame.inner_ops.get(id(inner))
            if inner_op is None:
                op.setdefault("error", f"inner {type(inner).__name__} never ran")
            else:
                box["$cmd"] = inner_op
                if "error" in inner_op:
                    op.setdefault("error", inner_op["error"])
        if _frames:
            _frames[-1].inner_ops[id(self)] = op
        _ran[id(self)] = (self, op)
        object.__setattr__(self, _OP, op)
        return result

    do._golden = True
    cls.do = do


# ---- history ---------------------------------------------------------------

def _snapshot_document(scene) -> bytes:
    from formats import igz

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "start.igz"
        igz.save_scene(scene, path)
        return path.read_bytes()


def _sig(scene) -> str:
    return signature_hash(scene_signature(scene))


def counters() -> dict:
    """The process-wide counters a step may draw from (SCHEMA: Determinism)."""
    import itertools

    import core.group as group
    import core.mesh as mesh

    n = next(group._counter)
    group._counter = itertools.count(n)
    return {"group": n, "curve": mesh._CURVE_COUNTER}


def set_counters(c: dict) -> None:
    import itertools

    import core.group as group
    import core.mesh as mesh

    group._counter = itertools.count(c["group"])
    mesh._CURVE_COUNTER = c["curve"]


def _edit_path(scene):
    g = getattr(scene, "edit_group", None)
    if g is None:
        return None
    from tests.golden.values import group_paths
    for path, x in group_paths(scene):
        if x is g:
            return list(path)
    return "?"


def _patch_history():
    from core.history import History

    if getattr(History.execute, "_golden", False):
        return
    execute, undo, redo, init = History.execute, History.undo, History.redo, History.__init__

    def __init__(self, scene):
        init(self, scene)
        self._golden = None
        if _state["active"] and not _state["paused"]:
            _histories.append(self)

    def _script(self):
        if self._golden is None:
            try:
                start = _snapshot_document(self.scene)
                err = None
            except Exception as exc:  # noqa: BLE001 — reported, not raised
                start, err = None, f"start: {type(exc).__name__}: {exc}"
            parts = scene_signature(self.scene)
            self._golden = {"start": start, "start_sig": signature_hash(parts),
                            "start_parts": parts,
                            "steps": [], "errors": [] if err is None else [err],
                            "last_sig": None}
        return self._golden

    def _check_untouched(self, script):
        if script["last_sig"] is not None and _sig(self.scene) != script["last_sig"]:
            script["errors"].append(
                f"step {len(script['steps'])}: the scene changed outside the history")

    def _execute(self, cmd):
        if not _state["active"] or _state["paused"]:
            return execute(self, cmd)
        script = _script(self)
        _check_untouched(self, script)
        where = _edit_path(self.scene)
        before = counters()
        _ran.clear()
        execute(self, cmd)
        _ran.clear()
        op = dict(getattr(cmd, _OP, None) or {"op": op_name(type(cmd)), "error": "did not run"})
        if where is not None:
            op["in"] = where
            op.setdefault("error", "edit context (not replayed yet)")
        if self.last_rejected is not None:
            op["rejected"] = self.last_rejected
        if self.last_error is not None:
            op.setdefault("error", f"failed: {self.last_error}")
        op["counters"] = before
        op["sig"] = script["last_sig"] = _sig(self.scene)
        script["steps"].append(op)

    def _control(name, fn):
        def wrapped(self):
            if not _state["active"] or _state["paused"]:
                return fn(self)
            script = _script(self)
            _check_untouched(self, script)
            _state["paused"] += 1
            try:
                ok = fn(self)
            finally:
                _state["paused"] -= 1
            script["steps"].append({"op": name, "ok": ok,
                                    "sig": _sig(self.scene)})
            script["last_sig"] = script["steps"][-1]["sig"]
            return ok
        return wrapped

    History.__init__ = __init__
    History.execute = _execute
    History.undo = _control("undo", undo)
    History.redo = _control("redo", redo)
    History.execute._golden = True


def install() -> None:
    """Wrap every Command class and History. Idempotent."""
    global _installed
    if _installed:
        return
    for cls in _all_commands():
        if "__init__" in cls.__dict__:
            _wrap_init(cls)
        if "do" in cls.__dict__:
            _wrap_do(cls)
    _patch_history()
    _installed = True


def start() -> None:
    install()
    _state["active"] += 1


def stop() -> None:
    _state["active"] = max(0, _state["active"] - 1)


def take_histories() -> list:
    out = list(_histories)
    _histories.clear()
    return out


def script_of(history) -> dict | None:
    """The recorded script of a history, or None if nothing ran."""
    s = getattr(history, "_golden", None)
    if not s or not s["steps"]:
        return None
    return s


# ---- replay ----------------------------------------------------------------

class _Lazy:
    """An inner command rebuilt when it runs — its selectors were written
    against the scene at that moment, after its siblings had run."""


def from_op(op: dict, scene, reg=None, ran=None):
    reg = reg or registry()
    cls = reg.get(op["op"])
    if cls is None:
        raise LookupError(f"no unique command for op {op['op']!r}")
    from core.history import Command

    ran = ran if ran is not None else {}

    def make(inner_op):
        holder = {}

        class Lazy(Command):
            def _real(self, s):
                if "cmd" not in holder:
                    holder["cmd"] = from_op(inner_op, s, reg, ran)
                return holder["cmd"]

            def do(self, s):
                self._real(s).do(s)

            def undo(self, s):
                holder["cmd"].undo(s)

            def __getattr__(self, name):
                if "cmd" not in holder:
                    raise AttributeError(name)
                return getattr(holder["cmd"], name)

        lazy = Lazy()
        if "rid" in inner_op:
            ran[inner_op["rid"]] = lazy
        return lazy

    dec = Decoder(scene, make, ran)
    named = {k: dec.decode(v) for k, v in op.get("args", {}).items()}
    sig = inspect.signature(cls.__init__)
    args, kwargs = [], {}
    for p in list(sig.parameters.values())[1:]:
        if p.name not in named:
            continue
        v = named[p.name]
        if p.kind is p.VAR_POSITIONAL:
            args.extend(v)
        elif p.kind is p.VAR_KEYWORD:
            kwargs.update(v)
        elif p.kind is p.POSITIONAL_ONLY:
            args.append(v)
        else:
            kwargs[p.name] = v
    return cls(*args, **kwargs)


def replay(script: dict) -> dict:
    """Run ``script`` on a fresh scene. ``{"ok": bool, "at": step index of
    the first divergence or None, "why": text}``."""
    from core.history import History
    from core.scene import Scene
    from formats import igz

    _state["paused"] += 1
    try:
        scene = Scene()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "start.igz"
            path.write_bytes(script["start"])
            igz.load_into(scene, path)
        if _sig(scene) != script["start_sig"]:
            parts = sorted(k for k, v in scene_signature(scene).items()
                           if v != script.get("start_parts", {}).get(k))
            return {"ok": False, "at": -1,
                    "why": f"start document does not reopen to the same signature ({', '.join(parts)})"}
        history = History(scene)
        reg = registry()
        for i, step in enumerate(script["steps"]):
            try:
                if step["op"] in ("undo", "redo"):
                    ok = getattr(history, step["op"])()
                    if ok != step["ok"]:
                        return {"ok": False, "at": i, "why": f"{step['op']} returned {ok}"}
                else:
                    cmd = from_op(step, scene, reg)
                    set_counters(step["counters"])   # as when it ran: after it was built
                    history.execute(cmd)
                    if (history.last_rejected is not None) != ("rejected" in step):
                        return {"ok": False, "at": i, "why": f"rejected: {history.last_rejected!r}"}
                    if history.last_error is not None:
                        return {"ok": False, "at": i, "why": history.last_error}
            except Exception as exc:  # noqa: BLE001 — a divergence, reported
                return {"ok": False, "at": i, "why": f"{type(exc).__name__}: {exc}"}
            if _sig(scene) != step["sig"]:
                return {"ok": False, "at": i, "why": f"signature differs after {step['op']}"}
        return {"ok": True, "at": None, "why": ""}
    finally:
        _state["paused"] -= 1
