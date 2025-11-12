# semantic.py
from __future__ import annotations
from typing import Set, Tuple, Iterable

class SemanticError(Exception):
    """Raised when a semantic rule is violated."""
    pass


def check_semantics(assignments: Iterable[Tuple[object, object]]) -> None:
    """
    Enforce language rules on a sequence of assignments produced by the parser.

    Each assignment is (target, expr), where target is either:
      - ('id', name)       for identifiers
      - ('out', index)     for outputs

    Expressions may reference:
      - ('in', index)
      - ('ref_id', name)
      - ('ref_out', index)
      - ('not', child)
      - ('and', [children...])
      - ('or',  [children...])
      - (optional) ('paren', child) is treated like its child.

    Rules enforced:
      1) Using a name (or out[k]) before it’s defined -> error.
      2) Reassigning a name or out[k] -> error.
      3) Using the target within its own assignment -> error.

    Note: Using out[k] on RHS is allowed *after* it’s defined.
    """
    defined_names: Set[str] = set()
    defined_outs: Set[int] = set()

    for lineno, (target, expr) in enumerate(assignments, start=1):
        tgt_kind, tgt_val = _target_kind_and_value(target)

        # 1) Redefinition of target
        if tgt_kind == "id":
            if tgt_val in defined_names:
                raise SemanticError(f"Line {lineno}: name '{tgt_val}' reassigned.")
        else:  # "out"
            if tgt_val in defined_outs:
                raise SemanticError(f"Line {lineno}: out[{tgt_val}] reassigned.")

        # 2) Collect RHS references
        ids_used, outs_used, _ = collect(expr)

        # 3) Self-use (target appears in its own RHS)
        if tgt_kind == "id":
            if tgt_val in ids_used:
                raise SemanticError(
                    f"Line {lineno}: target '{tgt_val}' used in its own assignment."
                )
        else:
            if tgt_val in outs_used:
                raise SemanticError(
                    f"Line {lineno}: target out[{tgt_val}] used in its own assignment."
                )

        # 4) Used-before-defined (identifiers and outs on RHS must be previously defined)
        for name in ids_used:
            if name not in defined_names:
                raise SemanticError(f"Line {lineno}: '{name}' used before definition.")
        for k in outs_used:
            if k not in defined_outs:
                raise SemanticError(f"Line {lineno}: out[{k}] used before definition.")

        # 5) Commit definition of LHS
        if tgt_kind == "id":
            defined_names.add(tgt_val)
        else:
            defined_outs.add(tgt_val)


def collect(expr: object) -> Tuple[Set[str], Set[int], Set[int]]:
    """
    Traverse an expression and return (ids_used, outs_used, ins_used).
    Works with the tuple-shaped nodes documented above.
    """
    ids: Set[str] = set()
    outs: Set[int] = set()
    ins: Set[int] = set()

    def walk(e: object) -> None:
        if e is None:
            return
        if not isinstance(e, tuple) or not e:
            return  # ignore unknown nodes silently

        tag = e[0]
        if tag == "in":
            ins.add(int(e[1])); return
        if tag == "ref_id":
            ids.add(str(e[1])); return
        if tag == "ref_out":
            outs.add(int(e[1])); return
        if tag == "not":
            walk(e[1]); return
        if tag in ("and", "or"):
            for c in e[1]:
                walk(c)
            return
        if tag == "paren":
            walk(e[1]); return

        # Unknown tag -> ignore

    walk(expr)
    return ids, outs, ins


def _target_kind_and_value(target: object) -> Tuple[str, object]:
    """
    Return ("id", name) or ("out", index) for a target node.
    Supports only the simple tuple forms ('id', name) / ('out', idx).
    """
    if isinstance(target, tuple) and target:
        tag = target[0]
        if tag == "id":
            return "id", str(target[1])
        if tag == "out":
            return "out", int(target[1])
    raise TypeError(f"Unrecognized target node: {target!r}")


__all__ = ["SemanticError", "check_semantics", "collect"]
