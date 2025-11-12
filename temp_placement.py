#!/usr/bin/env python3
"""
temp_placement.py — Step 7: choose (x, y) coordinates for each node.

Key rules (per spec):
  - IN at left border (x=0) on even rows.
  - OUT at right border on even rows.
  - Gates (NOT/AND/OR) strictly inside (1 <= x <= out_x-1).
  - Gate columns by topo-level: x = 2 + COL_SPACING * level.
  - Gate rows spaced away from IO rows.

This version adds:
  - Unary gates (NOT with a single fanin) are snapped vertically to
    the row of their fanin gate. This keeps red NOT visually in the
    same lane as the yellow gate that drives it (e.g. XOR's "both").
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

# Kinds
K_IN  = "IN"
K_OUT = "OUT"
GATE_KINDS = {"NOT", "AND", "OR"}

# Spacing defaults (spec-friendly: IO on even rows)
ROW_SPACING = 2    # IO rows: 0,2,4,...
COL_SPACING = 3    # horizontal spacing between gate columns

def column_x(level: int, col_spacing: int = COL_SPACING) -> int:
    """
    Gate columns by topo-level.
    Level 0 is reserved for inputs (x=0), so we start gates at x=2.
    """
    return 2 + col_spacing * level

def is_io_row(y: int, row_spacing: int = ROW_SPACING) -> bool:
    """
    Inputs/outputs live on "even rows" in terms of row_spacing.
    If row_spacing=2, legal IO rows are 0,2,4,...
    """
    return (y % row_spacing) == 0

@dataclass
class PlacementResult:
    coords: Dict[str, Tuple[int, int]]
    width: int
    height: int

# ---------------- Level computation (from topo / fanins) -------------------

def _compute_levels(
    nodes: List[Dict],
    fanins: Dict[str, List[str]],
    topo_order: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Assign a 'level' to each node:
      - IN at level 0
      - other nodes at max(level(pred)) + 1
    If topo_order is provided, we walk it; otherwise we compute a Kahn topo.
    """
    levels: Dict[str, int] = {}

    # Seed inputs with level 0
    for n in nodes:
        if n["kind"] == K_IN:
            levels[n["id"]] = 0

    # Build topo if needed
    if topo_order is None:
        indeg = {n["id"]: 0 for n in nodes}
        for dst, preds in fanins.items():
            indeg[dst] = len(preds)
        q = deque([nid for nid, d in indeg.items() if d == 0])
        topo: List[str] = []
        succs: Dict[str, List[str]] = defaultdict(list)
        for dst, preds in fanins.items():
            for p in preds:
                succs[p].append(dst)
        while q:
            u = q.popleft()
            topo.append(u)
            for v in succs.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    else:
        topo = topo_order

    # Propagate levels
    for nid in topo:
        if nid not in levels:
            preds = fanins.get(nid, [])
            if not preds:
                # Isolated non-input node; place it after inputs.
                levels[nid] = 1
            else:
                levels[nid] = max(levels.get(p, 0) for p in preds) + 1

    # Ensure every node has some level
    for n in nodes:
        if n["id"] not in levels:
            levels[n["id"]] = 0

    return levels

# ----------------------------- Main placer ---------------------------------

def place(
    nodes: List[Dict],
    fanins: Dict[str, List[str]],
    topo_order: Optional[List[str]] = None,
    row_spacing: int = ROW_SPACING,
    col_spacing: int = COL_SPACING,
) -> Dict:
    """
    Compute coordinates for each node under the contract constraints.

    Returns:
      {
        "coords": { node_id: (x,y), ... },
        "width":  int,
        "height": int,
      }
    """
    by_id = {n["id"]: n for n in nodes}
    inputs  = sorted([n for n in nodes if n["kind"] == K_IN],  key=lambda n: n["io_index"])
    outputs = sorted([n for n in nodes if n["kind"] == K_OUT], key=lambda n: n["io_index"])
    gates   = [n for n in nodes if n["kind"] in GATE_KINDS]

    levels = _compute_levels(nodes, fanins, topo_order)

    # Group gates by column (level) for deterministic stacking
    gates_by_level: Dict[int, List[Dict]] = defaultdict(list)
    for g in gates:
        lvl = levels[g["id"]]
        gates_by_level[lvl].append(g)

    coords: Dict[str, Tuple[int,int]] = {}

    # Inputs: x=0, y = 0, row_spacing, 2*row_spacing, ...
    for idx, n in enumerate(inputs):
        y = idx * row_spacing
        coords[n["id"]] = (0, y)

    # Helper: approximate which input index a gate is "closest" to
    def _nearest_input_index(nid: str) -> int:
        preds = fanins.get(nid, [])
        best = 10**9
        found = False
        for p in preds:
            kind = by_id[p]["kind"]
            if kind == K_IN:
                idx = by_id[p]["io_index"] or 0
                if not found or idx < best:
                    best = idx
                    found = True
        return best if found else 0

    # Initial gate placement: stack per level, spaced from IO rows
    for lvl, glist in gates_by_level.items():
        glist.sort(key=lambda g: (_nearest_input_index(g["id"]), g["id"]))
        x = column_x(lvl, col_spacing)
        for i, g in enumerate(glist):
            # place gates on rows 1, 3, 5, ... between IO rows (0,2,4,...)
            y = max(1, i * row_spacing + 1)
            coords[g["id"]] = (x, y)

    # --- Unary alignment: snap NOT gates to their single gate fanin row ---
    for g in gates:
        if g["kind"] != "NOT":
            continue
        gid = g["id"]
        preds = fanins.get(gid, [])
        if len(preds) != 1:
            continue
        parent = preds[0]
        if parent not in coords:
            continue
        px, py = coords[parent]
        gx, _  = coords[gid]
        # keep same column, move NOT vertically to its parent gate's row
        # as long as that row is not an IO row
        if not is_io_row(py, row_spacing):
            coords[gid] = (gx, py)

    # Outputs: x at right border, y chosen from fanins' rows (median) snapped to IO row.
    all_x = [0] + [x for x, _ in coords.values()]
    max_gate_x = max(all_x)
    out_x = max_gate_x + col_spacing
    used_out_rows: Set[int] = set()

    def _snap_to_io_row(y: int) -> int:
        k = round(y / row_spacing)
        return k * row_spacing

    for out in outputs:
        preds = fanins.get(out["id"], [])
        if preds:
            ys = [coords[p][1] for p in preds if p in coords]
            ys.sort()
            mid = ys[len(ys)//2] if ys else 0
            y_guess = _snap_to_io_row(mid)
        else:
            y_guess = 0
        y = y_guess
        while y in used_out_rows:
            y += row_spacing
        used_out_rows.add(y)
        coords[out["id"]] = (out_x, y)

    # Compute width/height
    all_x = [x for x, _ in coords.values()]
    all_y = [y for _, y in coords.values()] or [0]
    width = max(all_x) + 1
    height = max(all_y) + 1

    # Sanity checks
    for nid, (x, y) in coords.items():
        n = by_id[nid]
        if n["kind"] == K_IN:
            if x != 0:
                raise ValueError(f"Input {nid} must be at x=0, got {x}")
            if not is_io_row(y, row_spacing):
                raise ValueError(f"Input {nid} must be on IO row, got y={y}")
        elif n["kind"] == K_OUT:
            if x != out_x:
                raise ValueError(f"Output {nid} must be at x={out_x}, got {x}")
            if not is_io_row(y, row_spacing):
                raise ValueError(f"Output {nid} must be on IO row, got y={y}")
        elif n["kind"] in GATE_KINDS:
            if not (1 <= x <= out_x - 1):
                raise ValueError(f"Gate {nid} must be interior (1..{out_x-1}), got x={x}")
        else:
            raise ValueError(f"Unknown node kind for {nid}: {n['kind']}")

    return {
        "coords": coords,
        "width": width,
        "height": height,
    }
