#!/usr/bin/env python3
"""
place.py – integrates steps 1–7 for the PLACE assignment.

Pipeline:
  1) CLI & input validation (lightweight here)
  2) Lexing (lexer.tokenize -> list of tokens)
  3) Parsing (parser.parse expects [(kind, value), ...])
  4) Semantic checks (tuple-shaped AST expected by semantic.py)
  5) Netlist DAG build (convert parser AST -> netlist AST, then build)
  6) Topological order (detect cycles; stable left→right ordering if provided)
  7) Placement (choose unique (x,y) for each node per spec)

Set PLACE_STOP_BEFORE_RENDER=1 to skip rendering (future steps).
Optionals:
  - PLACE_TOPO_METHOD = "kahn" | "dfs"   (default: "kahn")
  - PLACE_TOPO_KEY = "name" | ""         (default: "")  -> demo key for stability
  - PLACE_ENABLE_PLACEMENT = "0" to disable step 7 (default enabled)
"""

import os
import re
import sys
from typing import Tuple, List, Iterable, Hashable, Any, Dict

# ---- Imports from your repo ----
from lexer import tokenize
import parser as parser_mod
from semantic import check_semantics
from netlist import (
    Program as NLProgram, Assignment as NLAsn,
    Ident as NLIdent, OutRef as NLOutRef,
    InRef as NLInRef, IdentRef as NLIdentRef,
    Not as NLNot, And as NLAnd, Or as NLOr,
    build_netlist,
)

# ---- Step 6: Topological order
from topological import topo_order, verify_topological_order, CycleError

# ---- Step 7: Placement
import placement  # requires placement.py from our previous step


# ---------------- CLI helpers (local) ----------------

def _validate_cli(argv: List[str]) -> Tuple[str, str]:
    if len(argv) != 3:
        print("Usage: python3 place.py {formulas} {image}", file=sys.stderr)
        sys.exit(2)
    infile, outfile = argv[1], argv[2]

    # input exists & readable
    if not os.path.isfile(infile):
        print(f"Error: input file not found: {infile}", file=sys.stderr)
        sys.exit(1)
    if not os.access(infile, os.R_OK):
        print(f"Error: input file is not readable: {infile}", file=sys.stderr)
        sys.exit(1)

    # output must be .ppm and directory writable
    if not outfile.lower().endswith(".ppm"):
        print(f"Error: output image must be a .ppm file: {outfile}", file=sys.stderr)
        sys.exit(1)
    outdir = os.path.dirname(os.path.abspath(outfile)) or "."
    if not os.path.isdir(outdir):
        print(f"Error: output directory does not exist: {outdir}", file=sys.stderr)
        sys.exit(1)
    if not os.access(outdir, os.W_OK):
        print(f"Error: output directory is not writable: {outdir}", file=sys.stderr)
        sys.exit(1)

    return infile, outfile


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_ppm_placeholder(path: str) -> None:
    # Minimal valid ASCII PPM (1x1 white)
    with open(path, "w", encoding="ascii") as f:
        f.write("P3\n1 1\n255\n255 255 255\n")


# -------------- Converters --------------

def _tokens_for_parser(source: str):
    """Convert lexer.Token dataclasses -> (kind, value) tuples for parser.parse()."""
    return [(t.kind, t.value) for t in tokenize(source)]

def _expr_to_semantic_tuple(e):
    """Convert parser AST nodes -> tuple AST expected by semantic.py."""
    from parser import Identifier, In, Out, Not, And, Or
    if isinstance(e, Identifier):
        return ("ref_id", e.name)
    if isinstance(e, In):
        return ("in", e.index)
    if isinstance(e, Out):
        # semantics allows out[k] on RHS (after defined)
        return ("ref_out", e.index)
    if isinstance(e, Not):
        return ("not", _expr_to_semantic_tuple(e.expr))
    if isinstance(e, And):
        return ("and", [_expr_to_semantic_tuple(x) for x in e.parts])
    if isinstance(e, Or):
        return ("or", [_expr_to_semantic_tuple(x) for x in e.parts])
    raise TypeError(f"Unknown parser expr type {type(e)}")

def _program_to_semantic_tuples(program):
    """
    Convert parser.Program -> list[(target, expr)] for semantic.check_semantics.
    target is ('id', name) or ('out', k).
    """
    items = []
    for a in program.assignments:
        if a.target.is_ident():
            tgt = ("id", a.target.ident)
        else:
            tgt = ("out", a.target.out_index)
        expr = _expr_to_semantic_tuple(a.expr)
        items.append((tgt, expr))
    return items

def _expr_to_netlist_ast(e):
    """Convert parser expr nodes -> netlist's tiny AST nodes."""
    from parser import Identifier, In, Out, Not, And, Or
    if isinstance(e, Identifier):
        return NLIdentRef(e.name)
    if isinstance(e, In):
        return NLInRef(e.index)
    if isinstance(e, Out):
        raise TypeError("RHS out[k] not supported by netlist builder in step 1–5")
    if isinstance(e, Not):
        return NLNot(_expr_to_netlist_ast(e.expr))
    if isinstance(e, And):
        return NLAnd([_expr_to_netlist_ast(x) for x in e.parts])
    if isinstance(e, Or):
        return NLOr([_expr_to_netlist_ast(x) for x in e.parts])
    raise TypeError(f"Unknown parser expr type {type(e)}")

def _program_to_netlist_ast(program) -> NLProgram:
    """
    Convert parser.Program -> netlist.Program using netlist AST classes.
    Targets:
      - ident -> NLIdent(name)
      - out[k] -> NLOutRef(k)
    """
    nl_assignments = []
    for a in program.assignments:
        if a.target.is_ident():
            nl_target = NLIdent(a.target.ident)
        else:
            nl_target = NLOutRef(a.target.out_index)
        nl_expr = _expr_to_netlist_ast(a.expr)
        nl_assignments.append(NLAsn(nl_target, nl_expr))
    return NLProgram(nl_assignments)


# -------------- Step 5/6 bridge helpers --------------

def _extract_nodes_edges(net):
    """
    Be liberal in what we accept from build_netlist():
      - (nodes, edges)
      - object with .nodes / .edges attributes
      - dict-like with ['nodes'], ['edges']
    Returns (nodes, edges) where nodes is a list/set and edges is list of (u,v).
    """
    if isinstance(net, tuple) and len(net) == 2:
        nodes, edges = net
        return list(nodes), list(edges)

    # attribute style
    if hasattr(net, "nodes") and hasattr(net, "edges"):
        return list(getattr(net, "nodes")), list(getattr(net, "edges"))

    # mapping style
    if isinstance(net, dict) and "nodes" in net and "edges" in net:
        return list(net["nodes"]), list(net["edges"])

    # last resort: try to guess
    try:
        return list(net.nodes), list(net.edges)
    except Exception:
        pass

    raise TypeError("build_netlist() returned an unsupported structure for nodes/edges")


def _optional_topo_key(nodes: Iterable[Hashable]) -> Any:
    """
    Demo stable key for left→right flavor if desired.
    Env:
      PLACE_TOPO_KEY = 'name' -> uses str(node)
      '' (default)          -> no custom key
    """
    mode = os.environ.get("PLACE_TOPO_KEY", "").strip().lower()
    if mode == "name":
        return lambda n: str(n)
    return None


# -------------- Step 7 bridge: netlist -> placement inputs --------------

_RE_IN  = re.compile(r"^in\[(\d+)\]$", re.IGNORECASE)
_RE_OUT = re.compile(r"^out\[(\d+)\]$", re.IGNORECASE)

def _node_kind_and_index(n) -> Tuple[str, int | None]:
    """
    Try very hard to infer placement 'kind' and 'io_index' from a netlist node.
    Supports:
      - objects with attributes: .kind, .io_index
      - mapping-like with keys: ['kind'], ['io_index']
      - strings like 'in[0]' / 'out[2]' (case-insensitive)
      - fallback: treat as a gate ('AND') with io_index=None for placement purposes
    """
    # attribute style
    if hasattr(n, "kind"):
        k = getattr(n, "kind")
        idx = getattr(n, "io_index", None)
        return str(k).upper(), idx

    # mapping style
    if isinstance(n, dict) and "kind" in n:
        k = n["kind"]
        idx = n.get("io_index", None)
        return str(k).upper(), idx

    s = str(n)

    m = _RE_IN.match(s)
    if m:
        return "IN", int(m.group(1))

    m = _RE_OUT.match(s)
    if m:
        return "OUT", int(m.group(1))

    # Fallback: unknown gate; we only need to know it's a gate for placement
    return "AND", None


def _edges_to_fanins(edges) -> Dict[str, List[str]]:
    """
    Convert list of (u,v) edges to fanins dict: v -> [u,...]
    Use str(node) as id to be stable across node object types.
    """
    fanins: Dict[str, List[str]] = {}
    for u, v in edges:
        su, sv = str(u), str(v)
        fanins.setdefault(sv, []).append(su)
    return fanins


def _netlist_to_placement_inputs(nodes, edges):
    """
    Produce (place_nodes, fanins) for placement.place().
    Each place_node: {"id": str, "kind": "IN"|"OUT"|"NOT"|"AND"|"OR", "io_index": int|None}
    """
    place_nodes: List[Dict] = []
    for n in nodes:
        kind, idx = _node_kind_and_index(n)
        place_nodes.append({"id": str(n), "kind": kind, "io_index": idx})
    fanins = _edges_to_fanins(edges)
    return place_nodes, fanins


# -------------- Public helper for tests --------------

def pipeline_build_topo(source: str):
    """
    Run steps 2–6 on in-memory source.
    Returns nodes, edges, order for test/checkers.
    """
    tokens = _tokens_for_parser(source)
    program = parser_mod.parse(tokens)
    assignments = _program_to_semantic_tuples(program)
    check_semantics(assignments)
    nl_prog = _program_to_netlist_ast(program)
    net = build_netlist(nl_prog)

    nodes, edges = _extract_nodes_edges(net)

    method = os.environ.get("PLACE_TOPO_METHOD", "kahn").strip().lower() or "kahn"
    key = _optional_topo_key(nodes)
    order = topo_order(nodes, edges, method=method, key=key)

    if not verify_topological_order(order, edges):
        raise RuntimeError("Internal error: produced topological order is invalid.")
    return nodes, edges, order


# -------------- Main --------------

def main():
    try:
        infile, outfile = _validate_cli(sys.argv)
        src = _read_text(infile)

        # (2–6) Full pipeline, with topo
        nodes, edges, order = pipeline_build_topo(src)

        # ---- (7) Placement (unless disabled) ----
        do_place = os.environ.get("PLACE_ENABLE_PLACEMENT", "1") != "0"
        if do_place:
            p_nodes, fanins = _netlist_to_placement_inputs(nodes, edges)

            # Provide topo order as strings if available (helps stability, but placement recomputes levels too)
            topo_order_str = [str(n) for n in order] if order else None

            place_result = placement.place(
                p_nodes,
                fanins,
                topo_order=topo_order_str,
                row_spacing=placement.ROW_SPACING,
                col_spacing=placement.COL_SPACING,
            )

            # Small, helpful summary
            coords = place_result["coords"]
            width, height = place_result["width"], place_result["height"]
            print(f"[OK] Placement: width={width}, height={height}")
            # Print first ~8 nodes in left-to-right, top-to-bottom
            preview = sorted(coords.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))[:8]
            for nid, (x, y) in preview:
                print(f"  {nid} -> ({x}, {y})")

        # steps 1–7 mode without rendering? (still output placeholder to satisfy contract)
        _write_ppm_placeholder(outfile)
        if os.environ.get("PLACE_STOP_BEFORE_RENDER") == "1":
            print("[OK] Parsed, checked, built DAG, topo, placement. (steps 1–7 mode)")
            sys.exit(0)

        # (Future) real rendering; for now we already wrote placeholder
        print("[OK] Pipeline completed (placeholder render).")
        sys.exit(0)

    except CycleError as ce:
        detail = f" (cycle nodes: {ce.cycle_nodes})" if getattr(ce, "cycle_nodes", None) else ""
        print(f"Error: cycle detected in netlist{detail}", file=sys.stderr)
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
