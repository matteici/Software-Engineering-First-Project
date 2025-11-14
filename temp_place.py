#!/usr/bin/env python3
"""
temp_place.py – integrates steps 1–8 for the PLACE assignment.

Pipeline:
  1) CLI & input validation
  2) Lexing (lexer.tokenize)
  3) Parsing (parser.parse expects [(kind, value), ...])
  4) Semantic checks (tuple-shaped AST expected by semantic.check_semantics)
  5) Netlist DAG build (build_netlist -> Netlist)
  6) Topological order (detect cycles; stable left→right ordering if provided)
  7) Placement (choose unique (x,y) for each node per spec)
  8) Routing + render to PPM (Manhattan paths; color rules + gate pin constraints)

Env toggles:
  - PLACE_TOPO_METHOD = "kahn" | "dfs"   (default: "kahn")
  - PLACE_TOPO_KEY    = "name" | ""      (default: "")
  - PLACE_ENABLE_PLACEMENT   = "0" to disable   (default: enabled)
  - PLACE_STOP_BEFORE_RENDER = "1" to stop after placement and write placeholder
"""

import os
import re
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Hashable, Iterable, List, Optional, Set, Tuple

from PIL import Image

# ---- Repo imports ----
from lexer import tokenize
import parser as parser_mod
from semantic import check_semantics
from netlist import (
    Program as NLProgram,
    Assignment as NLAsn,
    Ident as NLIdent,
    OutRef as NLOutRef,
    InRef as NLInRef,
    IdentRef as NLIdentRef,
    Not as NLNot,
    And as NLAnd,
    Or as NLOr,
    build_netlist,
)
from topological import topo_order, verify_topological_order, CycleError
import placement
import routing

# ---- Colors (rendering) ----
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)    # bends/tees
BLUE   = (0,   0, 255)    # wires & IO endpoints
RED    = (255, 0,   0)    # NOT
YELLOW = (255, 255, 0)    # AND
GREEN  = (0, 255,   0)    # OR


# =============================== CLI helpers ===============================

def _validate_cli(argv: List[str]) -> Tuple[str, str]:
    if len(argv) != 3:
        print("Usage: python3 temp_place.py {formulas} {image}", file=sys.stderr)
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
    """Write a minimal valid 1x1 white ASCII PPM."""
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
        # semantics allows out[k] on RHS (after defined)
        return ("ref_out", e.index)
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
        # Allow out[k] on the RHS; semantics already ensured it's defined.
        return NLOutRef(e.index)

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


def _edges_to_fanins(edges) -> Dict[str, List[str]]:
    """
    Convert list of (u,v) edges to fanins dict: v -> [u,...]
    Use str(node) for stability.
    """
    fanins: Dict[str, List[str]] = {}
    for u, v in edges:
        su, sv = str(u), str(v)
        fanins.setdefault(sv, []).append(su)
    return fanins


def _fanouts_from_edges(edges) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, List[str]]]:
    """
    Build:
      fanouts: u -> [(v, i), ...] where i is index of u in v's fanins
      fanins : v -> [u, ...]
    """
    fanins = _edges_to_fanins(edges)
    fanouts: DefaultDict[str, List[Tuple[str, int]]] = defaultdict(list)
    for v, us in fanins.items():
        for i, u in enumerate(us):
            fanouts[str(u)].append((str(v), i))
    return fanouts, fanins


# ---- helper for check_place_pipeline.py (unchanged API) ----

def pipeline_build_topo(source: str):
    """
    Run steps 2–6 on in-memory source.
    Returns (nodes, edges, order) for external checkers.
    """
    tokens = _tokens_for_parser(source)
    program = parser_mod.parse(tokens)
    assignments = _program_to_semantic_tuples(program)
    check_semantics(assignments)
    nl_prog = _program_to_netlist_ast(program)
    net = build_netlist(nl_prog)

    nodes = list(net.nodes.keys())
    edges = list(net.edges)

    method = os.environ.get("PLACE_TOPO_METHOD", "kahn").strip().lower() or "kahn"
    key = _optional_topo_key(nodes)
    order = topo_order(nodes, edges, method=method, key=key)

    if not verify_topological_order(order, edges):
        raise RuntimeError("Internal error: produced topological order is invalid.")
    return nodes, edges, order


# ====================== Step 8: placement -> routing =======================

def _gate_color(kind: str) -> Tuple[int, int, int]:
    k = (kind or "").upper()
    if k == "NOT":
        return RED
    if k == "AND":
        return YELLOW
    if k == "OR":
        return GREEN
    return WHITE  # IN/OUT -> WHITE (IO endpoints painted separately)


def _lane_offsets(n: int) -> List[int]:
    """Symmetric lane offsets around the driver's row, e.g., n=3 -> [-1,0,1]."""
    return [j - (n - 1) // 2 for j in range(n)]


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _find_clear_lane_x(
    ux: int,
    uy: int,
    offsets: List[int],
    occupied_centers: Set[Tuple[int, int]],
    width: int,
    height: int,
    start_dx: int = 2,
) -> int:
    """
    Find a split column > ux where:
      - (lane_x, uy) is not a gate center
      - (lane_x, uy+off) are not gate centers for all lane offsets
    Fallback: first trunk-clear column.
    """
    fallback = None
    for dx in range(start_dx, max(start_dx, width - ux)):
        lx = ux + dx
        mid = (lx, uy)
        if mid in occupied_centers:
            continue
        lanes_ok = True
        for off in offsets:
            yy = _clamp(uy + off, 0, height - 1)
            if (lx, yy) in occupied_centers:
                lanes_ok = False
                break
        if lanes_ok:
            return lx
        if fallback is None:
            fallback = lx
    return fallback if fallback is not None else min(width - 1, ux + start_dx)


def _render_and_route_ppm(
    outfile: str,
    place_result: Dict[str, Any],
    net,
    edges,
) -> None:
    """
    Rendering with routing:

      - Paint gates (NOT/AND/OR) at placed centers (IN/OUT left white)
      - Paint IO endpoints (IN at x=0 even rows; OUT at x=w-1 even rows) BLUE
      - Build fanouts/fanins from edges
      - For each gate, assign distinct input pins (left/up/down neighbors) and
        fix the output pin at right neighbor => ensure:
          * NOT has exactly 2 non-white neighbors
          * AND/OR have exactly 3 non-white neighbors
      - Block all *other* neighbors of gate centers so wires cannot touch them
      - Route nets via Manhattan paths with lane splitting + light wire halo
      - Upscale and save PPM
    """
    # --- tunables ---
    UPSCALE        = 4
    SPLIT_START_DX = 2
    LANE_COL_GAP   = 2

    width  = place_result["width"]
    height = place_result["height"]
    coords: Dict[str, Tuple[int, int]] = place_result["coords"]

    # Kinds from the real netlist (Netlist.nodes[nid].op)
    kinds: Dict[str, str] = {}
    for nid, node in net.nodes.items():
        kinds[str(nid)] = (node.op or "").upper()

    # 1px logical grid
    grid = routing.Grid(width, height, WHITE)

    # 1) Paint gate centers
    for nid_str, (x, y) in coords.items():
        c = _gate_color(kinds.get(nid_str, ""))
        if c != WHITE and grid.inside(x, y):
            grid.set(x, y, c)

    # 2) Paint IO endpoints BLUE per spec
    for nid, node in net.nodes.items():
        nid_str = str(nid)
        if nid_str not in coords:
            continue
        x, y = coords[nid_str]
        k = (node.op or "").upper()
        if k == "IN" and x == 0 and (y % 2 == 0):
            grid.set(x, y, BLUE)
        if k == "OUT" and x == width - 1 and (y % 2 == 0):
            grid.set(x, y, BLUE)

    # 3) Fanouts / fanins (string ids)
    fanouts, fanins = _fanouts_from_edges(edges)

    routing_failed = False

    # --- Gate pin assignment & neighbor blocking ---

    # For each gate v, we assign:
    #   - output pin at (vx+1, vy)
    #   - NOT: 1 input pin among {L,U,D}
    #   - AND/OR: 2 distinct input pins among {L,U,D}
    # and block all other neighbors around the gate.
    gate_input_pins: Dict[Tuple[str, int], Tuple[int, int]] = {}
    gate_forbidden_neighbors: Set[Tuple[int, int]] = set()

    for v_str, us in fanins.items():
        kind_v = kinds.get(v_str, "")
        if kind_v not in ("NOT", "AND", "OR"):
            continue
        if v_str not in coords:
            continue
        vx, vy = coords[v_str]
        n_us = len(us)

        # neighbors
        nbrs: Dict[str, Tuple[int, int]] = {}
        for name, (dx, dy) in {
            "L": (-1, 0),
            "U": (0, -1),
            "D": (0, 1),
            "R": (1, 0),
        }.items():
            nx, ny = vx + dx, vy + dy
            if 0 <= nx < width and 0 <= ny < height:
                nbrs[name] = (nx, ny)

        # output pin is always to the right
        out_pin = nbrs.get("R")

        # candidate inputs: left, up, down
        candidates = [nbrs[n] for n in ("L", "U", "D") if n in nbrs]

        desired_inputs = 1 if kind_v == "NOT" else 2
        assigned_inputs: List[Tuple[int, int]] = candidates[:desired_inputs]

        # Fallback if not enough neighbor cells exist: reuse left with offsets
        while len(assigned_inputs) < min(desired_inputs, n_us) and "L" in nbrs:
            assigned_inputs.append(nbrs["L"])

        # Map each fanin index to an input pin
        for idx, _u in enumerate(us):
            key = (v_str, idx)
            if idx < len(assigned_inputs):
                gate_input_pins[key] = assigned_inputs[idx]
            else:
                # Extra fanins (shouldn't happen in well-formed XOR) get stacked on left
                if "L" in nbrs:
                    gate_input_pins[key] = nbrs["L"]
                else:
                    gate_input_pins[key] = (vx, vy)  # worst-case fallback

        # Block all gate neighbors except the assigned pins + output
        allowed: Set[Tuple[int, int]] = set(assigned_inputs)
        if out_pin is not None:
            allowed.add(out_pin)
        for (nx, ny) in nbrs.values():
            if (nx, ny) not in allowed:
                gate_forbidden_neighbors.add((nx, ny))

    # --- DEBUG: fanouts and planned routes ---
    print("\n[DEBUG] Fanouts per driver:")
    for u, tlist in sorted(fanouts.items(), key=lambda kv: kv[0]):
        print(f"  {u}: {len(tlist)} sink(s) -> {[v for (v, _) in tlist]}")

    print("[DEBUG] Planned pin routes (u, u_out) -> (v, v_in):")

    # Occupied centers: all node centers (gates + IO)
    occupied_centers: Set[Tuple[int, int]] = set(coords.values())

    def _wire_halo_local(g: routing.Grid) -> Set[Tuple[int, int]]:
        """
        Light halo: for every BLUE/BLACK wire pixel, mark its 4-neighbors
        (if still WHITE) as temporarily blocked to encourage whitespace.
        """
        halo: Set[Tuple[int, int]] = set()
        for yy in range(g.h):
            for xx in range(g.w):
                c = g.get(xx, yy)
                if c == BLUE or c == BLACK:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = xx + dx, yy + dy
                        if g.inside(nx, ny) and g.get(nx, ny) == WHITE:
                            halo.add((nx, ny))
        return halo

    # 4) Route each driver's fanouts with lane splitting
    for u, targets in fanouts.items():
        if str(u) not in coords or not targets:
            continue
        ux, uy = coords[str(u)]
        u_out = (ux + 1, uy)  # drive right

        offsets = _lane_offsets(len(targets))

        # choose split column
        lane_x = _find_clear_lane_x(
            ux, uy, offsets, occupied_centers, width, height, start_dx=SPLIT_START_DX
        )

        # (A) trunk: u_out -> trunk_mid
        trunk_mid = (lane_x, uy)
        blocked_trunk = occupied_centers | gate_forbidden_neighbors | _wire_halo_local(grid)
        if not routing.route(grid, u_out, trunk_mid, blocked_trunk):
            print(f"[WARN] Lane trunk failed from {u_out} to {trunk_mid} for {u}")
            routing_failed = True


        # (B) split lanes: trunk_mid -> per-lane waypoint
        waypoints: List[Tuple[int, int]] = []
        for lane_idx, off in enumerate(offsets):
            wy = _clamp(uy + off, 0, height - 1)
            lane_col = _clamp(lane_x + LANE_COL_GAP * lane_idx, 0, width - 1)
            wp = (lane_col, wy)
            blocked_split = occupied_centers | gate_forbidden_neighbors | _wire_halo_local(grid)
            if not routing.route(grid, trunk_mid, wp, blocked_split):
                print(f"[WARN] Lane split failed from {trunk_mid} to {wp} for {u}")
                routing_failed = True
            waypoints.append(wp)


        # (C) waypoint -> target pin
        for (v, idx), wp in zip(targets, waypoints):
            v_str = str(v)
            if v_str not in coords:
                continue
            vx, vy = coords[v_str]
            us = fanins.get(v_str, [])
            kind_v = kinds.get(v_str, "")

            if kind_v == "OUT":
                v_in = (vx - 1, vy)
            elif kind_v in ("NOT", "AND", "OR") and (v_str, idx) in gate_input_pins:
                v_in = gate_input_pins[(v_str, idx)]
            elif len(us) <= 1:
                v_in = (vx - 1, vy)
            else:
                off = idx - (len(us) - 1) // 2
                yy = _clamp(vy + off, 0, height - 1)
                v_in = (vx - 1, yy)

            print(f"  {u} {u_out}  ->  {v_str} {v_in}")
            blocked_net = occupied_centers | gate_forbidden_neighbors | _wire_halo_local(grid)
            if not routing.route(grid, wp, v_in, blocked_net):
                print(f"[WARN] Could not route from {wp} to {v_in} (u={u} -> v={v_str})")
                routing_failed = True


    if routing_failed:
        print(
            "Error: routing failed for at least one net; not writing image.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 5) Upscale and save
    big = grid.img.resize((width * UPSCALE, height * UPSCALE), Image.NEAREST)
    big.save(outfile, format="PPM")



# ================================ Main =====================================

def main() -> None:
    try:
        infile, outfile = _validate_cli(sys.argv)
        src = _read_text(infile)

        # Full pipeline (2–6) in-line so we keep the real Netlist object.
        tokens = _tokens_for_parser(src)
        program = parser_mod.parse(tokens)
        assignments = _program_to_semantic_tuples(program)
        check_semantics(assignments)
        nl_prog = _program_to_netlist_ast(program)
        net = build_netlist(nl_prog)   # netlist.Netlist

        nodes = list(net.nodes.keys())
        edges = list(net.edges)

        method = os.environ.get("PLACE_TOPO_METHOD", "kahn").strip().lower() or "kahn"
        key = _optional_topo_key(nodes)
        order = topo_order(nodes, edges, method=method, key=key)
        if not verify_topological_order(order, edges):
            raise RuntimeError("Internal error: produced topological order is invalid.")

        # Placement (unless disabled)
        do_place = os.environ.get("PLACE_ENABLE_PLACEMENT", "1") != "0"
        if not do_place:
            _write_ppm_placeholder(outfile)
            print("[OK] Pipeline completed (placement disabled; wrote placeholder).")
            sys.exit(0)

        # Build placement nodes from *real* netlist data
        place_nodes: List[Dict[str, Any]] = []
        for nid, nd in net.nodes.items():
            kind = (nd.op or "").upper()
            io_index = nd.index if kind in ("IN", "OUT") else None
            place_nodes.append({"id": str(nid), "kind": kind, "io_index": io_index})

        fanins = _edges_to_fanins(edges)
        topo_order_str = [str(n) for n in order] if order else None

        place_result = placement.place(
            place_nodes,
            fanins,
            topo_order=topo_order_str,
            row_spacing=max(placement.ROW_SPACING, 2),
            col_spacing=placement.COL_SPACING,
        )

        coords = place_result["coords"]
        w, h = place_result["width"], place_result["height"]
        print(f"[OK] Placement: width={w}, height={h}")
        preview = sorted(coords.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))[:8]
        for nid_str, (x, y) in preview:
            print(f"  {nid_str} -> ({x}, {y})")

        # Optional stop before render
        if os.environ.get("PLACE_STOP_BEFORE_RENDER") == "1":
            _write_ppm_placeholder(outfile)
            print("[OK] Parsed, checked, built DAG, topo, placement. (steps 1–7 mode)")
            sys.exit(0)

        # Routing + render
        _render_and_route_ppm(outfile, place_result, net, edges)
        print(f"[OK] Pipeline completed (rendered with routing) -> {outfile}")
        sys.exit(0)

    except CycleError as ce:
        detail = (
            f" (cycle nodes: {ce.cycle_nodes})"
            if getattr(ce, "cycle_nodes", None)
            else ""
        )
        print(f"Error: cycle detected in netlist{detail}", file=sys.stderr)
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
