#!/usr/bin/env python3
"""
placement.py — Step 7: choose (x, y) coordinates for each node.

Upgrades:
  • Deterministic stacking of gates in each column using key (nearest_input_index, node_id).
  • OUT rows placed at the median of fanins' rows, snapped to the nearest even row.
    If that (x,y) is occupied on the right border, we bump to the next free even row.

Contract (unchanged essentials):
  - IN at left border (x=0) on even rows.
  - OUT at right border on even rows.
  - Gates (NOT/AND/OR) strictly inside.
  - Gate columns by topo-level: x = 2 + COL_SPACING * level.
  - Rows for gates spaced by +ROW_SPACING (defaults to 2).
  - Every node has a unique (x,y).

Inputs:
  - nodes: list of dicts {"id": str, "kind": "IN"|"OUT"|"NOT"|"AND"|"OR", "io_index": int|None}
  - fanins: dict node_id -> list of predecessor node_ids
  - topo_order: optional topological order (list of node_ids)

Returns:
  {
    "coords": dict node_id -> (x, y),
    "width": int,
    "height": int,
    "levels": dict node_id -> int,   # IN+GATE only
  }
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Tunables
ROW_SPACING = 2
COL_SPACING = 3
GATE_X_OFFSET = 2

K_IN, K_OUT = "IN", "OUT"
K_NOT, K_AND, K_OR = "NOT", "AND", "OR"
GATE_KINDS = {K_NOT, K_AND, K_OR}


# ---------- helpers exposed for downstream routing/rendering ----------

def column_x(level: int, col_spacing: int = COL_SPACING) -> int:
    """Gate column for a given topo level."""
    return GATE_X_OFFSET + col_spacing * level

def is_io_row(y: int, row_spacing: int = ROW_SPACING) -> bool:
    """Even rows are reserved for I/O."""
    return (y % row_spacing) == 0


# ---------- internal level computation ----------

def _compute_levels(
    nodes: List[Dict], fanins: Dict[str, List[str]], topo_order: Optional[List[str]]
) -> Dict[str, int]:
    kind = {n["id"]: n["kind"] for n in nodes}
    levels: Dict[str, int] = {}

    # Inputs at level 0
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
        topo_order = topo

    # Sweep to assign gate levels
    for nid in topo_order:
        k = kind.get(nid)
        if k in GATE_KINDS:
            preds = fanins.get(nid, [])
            if not preds:
                levels[nid] = 1
            else:
                levels[nid] = 1 + max(levels.get(p, 0) for p in preds)
    return levels


# ---------- deterministic stacking key ----------

def _nearest_input_index(
    nid: str,
    nodes_by_id: Dict[str, Dict],
    fanins: Dict[str, List[str]],
    memo: Dict[str, Optional[int]],
) -> Optional[int]:
    """Return the minimum io_index over all transitive IN ancestors of nid."""
    if nid in memo:
        return memo[nid]

    k = nodes_by_id[nid]["kind"]
    if k == K_IN:
        memo[nid] = nodes_by_id[nid]["io_index"]
        return memo[nid]

    best: Optional[int] = None
    for p in fanins.get(nid, []):
        val = _nearest_input_index(p, nodes_by_id, fanins, memo)
        if val is not None:
            best = val if best is None else min(best, val)
    memo[nid] = best
    return best


def _gate_sort_key(
    nid: str,
    nodes_by_id: Dict[str, Dict],
    fanins: Dict[str, List[str]],
    nearest_cache: Dict[str, Optional[int]],
) -> Tuple[int, str]:
    near = _nearest_input_index(nid, nodes_by_id, fanins, nearest_cache)
    # Put nodes without an input ancestor after those with a known ancestor
    near_key = 10**9 if near is None else near
    return (near_key, nid)


# ---------- main placement ----------

def place(
    nodes: List[Dict],
    fanins: Dict[str, List[str]],
    topo_order: Optional[List[str]] = None,
    row_spacing: int = ROW_SPACING,
    col_spacing: int = COL_SPACING,
) -> Dict:
    by_id = {n["id"]: n for n in nodes}
    inputs  = sorted([n for n in nodes if n["kind"] == K_IN],  key=lambda n: n["io_index"])
    outputs = sorted([n for n in nodes if n["kind"] == K_OUT], key=lambda n: n["io_index"])
    gates   = [n for n in nodes if n["kind"] in GATE_KINDS]

    levels = _compute_levels(nodes, fanins, topo_order)

    # Group gates by column (level) for deterministic stacking
    gates_by_col: Dict[int, List[str]] = defaultdict(list)
    for n in gates:
        lvl = levels.get(n["id"], 1)
        x = column_x(lvl, col_spacing)
        gates_by_col[x].append(n["id"])

    # Deterministic order per column
    nearest_cache: Dict[str, Optional[int]] = {}
    for x in list(gates_by_col.keys()):
        gates_by_col[x].sort(key=lambda g: _gate_sort_key(g, by_id, fanins, nearest_cache))

    coords: Dict[str, Tuple[int, int]] = {}

    # 1) Place inputs at left border on even rows
    for n in inputs:
        y = (n["io_index"] or 0) * row_spacing
        coords[n["id"]] = (0, y)

    # 2) Place gates in interior, stacking deterministically per column
    max_gate_x = 0
    for x in sorted(gates_by_col.keys()):
        next_y = 1  # start on odd row
        for nid in gates_by_col[x]:
            coords[nid] = (x, next_y)
            next_y += row_spacing
        if x > max_gate_x:
            max_gate_x = x

    # 3) Right border for outputs
    out_x = max_gate_x + 2

    # Place outputs at median fanin y (snapped to nearest even), de-colliding if needed
    occupied: set[Tuple[int, int]] = set(coords.values())
    for n in outputs:
        dst = n["id"]
        fanin_ys = [coords[s][1] for s in fanins.get(dst, []) if s in coords]
        if not fanin_ys:
            # default to io_index row if no placed fanins
            base_y = (n["io_index"] or 0) * row_spacing
        else:
            fanin_ys.sort()
            m = len(fanin_ys)
            median_y = fanin_ys[m // 2] if (m % 2 == 1) else (fanin_ys[m // 2 - 1] + fanin_ys[m // 2]) / 2
            # snap to nearest even row
            base_y = int(round(median_y / row_spacing) * row_spacing)

        # ensure it's even
        if base_y % row_spacing != 0:
            base_y += 1  # should not happen with the rounding above

        y = base_y
        # bump if (out_x, y) is taken
        while (out_x, y) in occupied:
            y += row_spacing
        coords[dst] = (out_x, y)
        occupied.add((out_x, y))

    # 4) Validate invariants
    _validate_invariants(nodes, coords, out_x)

    # 5) width/height
    max_y = max(y for _, y in coords.values()) if coords else 0
    width = out_x + 1
    height = max_y + 1

    return {"coords": coords, "width": width, "height": height, "levels": levels}


def _validate_invariants(nodes: List[Dict], coords: Dict[str, Tuple[int, int]], out_x: int) -> None:
    seen: Dict[Tuple[int, int], str] = {}
    for n in nodes:
        nid = n["id"]
        if nid not in coords:
            raise ValueError(f"Missing coordinate for node {nid}")
        xy = coords[nid]
        if xy in seen:
            raise ValueError(f"Two nodes share coordinate {xy}: {seen[xy]} and {nid}")
        seen[xy] = nid
        x, y = xy
        if n["kind"] == K_IN:
            if x != 0:
                raise ValueError(f"Input {nid} must be at x=0, got {x}")
            if y % ROW_SPACING != 0:
                raise ValueError(f"Input {nid} must be on even row, got y={y}")
        elif n["kind"] == K_OUT:
            if x != out_x:
                raise ValueError(f"Output {nid} must be at x={out_x}, got {x}")
            if y % ROW_SPACING != 0:
                raise ValueError(f"Output {nid} must be on even row, got y={y}")
        elif n["kind"] in GATE_KINDS:
            if not (1 <= x <= out_x - 1):
                raise ValueError(f"Gate {nid} must be interior (1..{out_x-1}), got x={x}")
        else:
            raise ValueError(f"Unknown node kind for {nid}: {n['kind']}")
