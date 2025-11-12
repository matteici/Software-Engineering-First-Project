#!/usr/bin/env python3
"""
temp_place.py — Full pipeline (Steps 1–8) for the PLACE assignment

Pipeline:
  1) CLI & input validation
  2) Lexing (lexer.tokenize)
  3) Parsing (parser.parse expects [(kind, value), ...])
  4) Semantic checks (tuple-shaped AST expected by semantic.check_semantics)
  5) Netlist DAG build (build_netlist)
  6) Topological order (Kahn/DFS)
  7) Placement (choose unique (x,y) for each node per spec)
  8) Routing + PPM render (Manhattan paths; color rules; soft halo + lane gaps)

Environment toggles:
  - PLACE_TOPO_METHOD = "kahn" | "dfs"        (default: "kahn")
  - PLACE_TOPO_KEY    = "name" | ""           (default: "")
  - PLACE_ENABLE_PLACEMENT = "0" to disable   (default: enabled)
  - PLACE_STOP_BEFORE_RENDER = "1" to stop after placement and write placeholder
"""

import os
import re
import sys
from typing import Tuple, List, Iterable, Hashable, Any, Dict, Set, DefaultDict

# ---- Repo imports ----
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
from topological import topo_order, verify_topological_order, CycleError
import temp_placement as placement
import routing  # provides Grid + route() with soft halo
from PIL import Image


# ---- Colors (for rendering) ----
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)       # bends/tees
BLUE   = (0, 0, 255)     # straight wires & 4-way crossings; IO endpoints
RED    = (255, 0, 0)     # NOT
YELLOW = (255, 255, 0)   # AND
GREEN  = (0, 255, 0)     # OR

# =============================== CLI helpers ===============================

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

# ============================ Parser converters ============================

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
        return ("ref_out", e.index)  # allowed on RHS after defined
    if isinstance(e, Not):
        return ("not", _expr_to_semantic_tuple(e.expr))
    if isinstance(e, And):
        return ("and", [_expr_to_semantic_tuple(x) for x in e.parts])
    if isinstance(e, Or):
        return ("or",  [_expr_to_semantic_tuple(x) for x in e.parts])
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
        raise TypeError("RHS out[k] is not supported by build_netlist (Step 5)")
    if isinstance(e, Not):
        return NLNot(_expr_to_netlist_ast(e.expr))
    if isinstance(e, And):
        return NLAnd([_expr_to_netlist_ast(x) for x in e.parts])
    if isinstance(e, Or):
        return NLOr([_expr_to_netlist_ast(x) for x in e.parts])
    raise TypeError(f"Unknown parser expr type {type(e)}")

def _program_to_netlist_ast(program) -> NLProgram:
    """Convert parser.Program -> netlist.Program using netlist AST classes."""
    nl_assignments = []
    for a in program.assignments:
        if a.target.is_ident():
            nl_target = NLIdent(a.target.ident)
        else:
            nl_target = NLOutRef(a.target.out_index)
        nl_expr = _expr_to_netlist_ast(a.expr)
        nl_assignments.append(NLAsn(nl_target, nl_expr))
    return NLProgram(nl_assignments)

# ============================ Step 5/6 helpers =============================

def _optional_topo_key(nodes: Iterable[Hashable]) -> Any:
    mode = os.environ.get("PLACE_TOPO_KEY", "").strip().lower()
    if mode == "name":
        return lambda n: str(n)
    return None

def _gate_color(kind: str):
    k = (kind or "").upper()
    if k == "NOT":  return RED
    if k == "AND":  return YELLOW
    if k == "OR":   return GREEN
    return WHITE  # IN/OUT -> WHITE (blue applied for IO endpoints separately)

# ---------- fanins / fanouts using string ids ----------

def _edges_to_fanins_str(edges: Iterable[Tuple[Any, Any]]) -> Dict[str, List[str]]:
    """Convert (u,v) edges (using numeric nids) into fanins dict keyed by string ids."""
    fanins: Dict[str, List[str]] = {}
    for u, v in edges:
        su, sv = str(u), str(v)
        fanins.setdefault(sv, []).append(su)
    return fanins

def _fanouts_from_edges_str(edges: Iterable[Tuple[Any, Any]]) -> Tuple[Dict[str, List[Tuple[str,int]]], Dict[str, List[str]]]:
    """
    Build:
      fanouts: "u" -> [("v", i), ...] where i is index of u in fanins[v]
      fanins : "v" -> ["u", ...]
    from edges that use numeric nids.
    """
    fanins = _edges_to_fanins_str(edges)
    fanouts: DefaultDict[str, List[Tuple[str,int]]] = {}
    for v, us in fanins.items():
        for i, u in enumerate(us):
            fanouts.setdefault(u, []).append((v, i))
    return fanouts, fanins

# ---------- small helpers for lane planning ----------

def _lane_offsets(n: int) -> List[int]:
    # symmetric offsets: e.g., n=3 -> [-1,0,1]
    base = [j - (n - 1)//2 for j in range(n)]
    return base

def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))

def _find_clear_lane_x(ux: int, uy: int, offsets: List[int],
                       occupied_centers: Set[Tuple[int,int]],
                       width: int, height: int, start_dx: int = 3) -> int:
    """
    Find a split column > ux where:
      - all candidate waypoints (lane_x, uy+off) are not gate centers, and
      - the trunk mid-point (lane_x, uy) is not a gate center.
    If no perfect column is found, pick the first that avoids the trunk center.
    """
    fallback = None
    for dx in range(start_dx, max(start_dx, width - ux)):
        lx = ux + dx
        mid_ok = (lx, uy) not in occupied_centers
        lanes_ok = True
        for off in offsets + [0]:
            yy = _clamp(uy + off, 0, height - 1)
            if (lx, yy) in occupied_centers:
                lanes_ok = False
                break
        if mid_ok and lanes_ok:
            return lx
        if fallback is None and mid_ok:
            fallback = lx
    return fallback if fallback is not None else min(width - 1, ux + start_dx)

# ====================== Step 8: placement -> routing =======================

def _render_and_route_ppm(outfile: str,
                          place_result: Dict,
                          net,  # netlist.Netlist
                          topo_order_ids: List[int],
                          place_nodes: List[Dict]):
    """
    Simple routing:
      - Paint gates (NOT/AND/OR) at placed centers.
      - Paint IN/OUT endpoints blue at the borders (even rows).
      - For each net edge u -> v, route directly from:
            u_out = (ux+1, uy)
            v_in  = (vx-1, vy or vy+offset)
        using routing.route (soft halo, then relaxed).
      - No explicit lane-split columns; BFS chooses paths.
      - Upscale with nearest neighbor and save PPM.
    """
    # --- tunables ---
    UPSCALE = 4
    # ---------------

    width, height = place_result["width"], place_result["height"]
    coords: Dict[str, Tuple[int, int]] = place_result["coords"]
    kinds: Dict[str, str] = {pn["id"]: (pn["kind"] or "").upper()
                             for pn in place_nodes}

    # 1px logical grid
    grid = routing.Grid(width, height, WHITE)

    # 1) Paint gates at centers (IN/OUT left white for now)
    for nid_str, (x, y) in coords.items():
        k = kinds.get(nid_str, "")
        c = _gate_color(k)
        if c != WHITE and grid.inside(x, y):
            grid.set(x, y, c)

    # 2) Paint IO endpoints BLUE per spec
    for pn in place_nodes:
        nid = pn["id"]
        k = (pn["kind"] or "").upper()
        if nid not in coords:
            continue
        x, y = coords[nid]
        # Inputs: x=0, even rows
        if k == "IN" and x == 0 and (y % 2 == 0) and grid.inside(x, y):
            grid.set(x, y, BLUE)
        # Outputs: x=w-1, even rows
        if k == "OUT" and x == width - 1 and (y % 2 == 0) and grid.inside(x, y):
            grid.set(x, y, BLUE)

    # 3) Fanouts/fanins from net edges (numeric ids → string ids)
    fanouts, fanins = _fanouts_from_edges_str(net.edges)

    print("\n[DEBUG] Fanouts per driver:")
    for u, tlist in sorted(fanouts.items(), key=lambda kv: int(kv[0])):
        print(f"  {u}: {len(tlist)} sink(s) -> {[v for (v, _) in tlist]}")
    print("[DEBUG] Planned pin routes (u, u_out) -> (v, v_in):")

    # All node centers (IO + gates) are treated as occupied to avoid running THROUGH them
    occupied_centers: Set[Tuple[int, int]] = set(coords.values())

    # 4) Route each net u -> v directly
    for u, targets in fanouts.items():
        if u not in coords:
            continue
        ux, uy = coords[u]
        u_out = (ux + 1, uy)  # drive right from center

        for (v, idx) in targets:
            if v not in coords:
                continue
            vx, vy = coords[v]
            us = fanins.get(v, [])
            # Decide input pin location at destination
            if kinds.get(v, "") == "OUT":
                v_in = (vx - 1, vy)
            elif len(us) <= 1:
                v_in = (vx - 1, vy)
            else:
                # vertically stagger multiple inputs to same gate
                off = idx - (len(us) - 1) // 2
                v_in = (vx - 1, _clamp(vy + off, 0, height - 1))

            print(f"  {u} {u_out}  ->  {v} {v_in}")

            # First try with soft halo (prefers whitespace), then without
            if not routing.route(grid, u_out, v_in, occupied_centers, prefer_soft_halo=True):
                if not routing.route(grid, u_out, v_in, occupied_centers, prefer_soft_halo=False):
                    print(f"[WARN] Could not route from {u_out} to {v_in} (u={u} -> v={v})")

    # 5) Upscale and save
    big = grid.img.resize((width * UPSCALE, height * UPSCALE), Image.NEAREST)
    big.save(outfile, format="PPM")

# ================================ Main =====================================

def main():
    try:
        infile, outfile = _validate_cli(sys.argv)
        src = _read_text(infile)

        # (2) Tokens -> (3) Parser
        tokens = _tokens_for_parser(src)
        program = parser_mod.parse(tokens)

        # (4) Semantic checks on tuple-shaped AST
        assignments = _program_to_semantic_tuples(program)
        check_semantics(assignments)

        # (5) Netlist
        nl_prog = _program_to_netlist_ast(program)
        net = build_netlist(nl_prog)  # netlist.Netlist

        # (6) Topological order (operates on numeric node ids)
        nodes_for_topo = list(net.nodes.keys())
        edges_for_topo = list(net.edges)
        method = os.environ.get("PLACE_TOPO_METHOD", "kahn").strip().lower() or "kahn"
        key = _optional_topo_key(nodes_for_topo)
        order = topo_order(nodes_for_topo, edges_for_topo, method=method, key=key)
        if not verify_topological_order(order, edges_for_topo):
            raise RuntimeError("Internal error: produced topological order is invalid.")

        # Allow stopping before render (Steps 1–7)
        if os.environ.get("PLACE_STOP_BEFORE_RENDER") == "1":
            _write_ppm_placeholder(outfile)
            print("[OK] Parsed, checked, built DAG, topo. (steps 1–6 mode)")
            sys.exit(0)

        # (7) Placement inputs from *real* netlist kinds
        place_nodes: List[Dict] = []
        for nid, nd in net.nodes.items():  # nd.op in {"IN","OUT","NOT","AND","OR"}, nd.index for IN/OUT
            kind = str(nd.op).upper()
            io_index = nd.index if kind in ("IN", "OUT") else None
            place_nodes.append({"id": str(nid), "kind": kind, "io_index": io_index})

        # fanins: v -> [u,...] using string ids
        fanins_str = _edges_to_fanins_str(net.edges)

        # Placement (deterministic via provided topo order as strings)
        do_place = os.environ.get("PLACE_ENABLE_PLACEMENT", "1") != "0"
        if not do_place:
            _write_ppm_placeholder(outfile)
            print("[OK] Pipeline completed (placement disabled; wrote placeholder).")
            sys.exit(0)

        topo_order_str = [str(n) for n in order] if order else None
        place_result = placement.place(
            place_nodes,
            fanins_str,
            topo_order=topo_order_str,
            row_spacing=placement.ROW_SPACING,   # stay at least 2 (even rows for IO)
            col_spacing=placement.COL_SPACING,
        )

        # Helpful summary
        coords = place_result["coords"]
        width, height = place_result["width"], place_result["height"]
        print(f"[OK] Placement: width={width}, height={height}")
        preview = sorted(coords.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))[:8]
        for nid, (x, y) in preview:
            print(f"  {nid} -> ({x}, {y})")

        # (8) Routing + render
        _render_and_route_ppm(outfile, place_result, net, order, place_nodes)
        print(f"[OK] Pipeline completed (rendered with routing) -> {outfile}")
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
