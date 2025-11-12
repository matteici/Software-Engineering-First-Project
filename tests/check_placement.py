#!/usr/bin/env python3
"""
check_placement.py — tests for upgraded placement.py

Validates:
  • Deterministic stacking (stable gate order per column).
  • OUT y equals median(fanins.y) snapped to even, with collision bumping.
  • Core invariants (I/O borders, interior gates, unique coords, column formula).
"""

from typing import Dict, List
from placement import place, ROW_SPACING, COL_SPACING, GATE_X_OFFSET, column_x

# Build a small DAG with two gates at the same level to test deterministic stacking
nodes: List[Dict] = [
    {"id": "in0",   "kind": "IN",  "io_index": 0},
    {"id": "in1",   "kind": "IN",  "io_index": 1},
    {"id": "a_or",  "kind": "OR",  "io_index": None},   # level 1
    {"id": "b_and", "kind": "AND", "io_index": None},   # level 1
    {"id": "c_not", "kind": "NOT", "io_index": None},   # level 2
    {"id": "fin",   "kind": "AND", "io_index": None},   # level 3
    {"id": "out0",  "kind": "OUT", "io_index": 0},
    {"id": "out1",  "kind": "OUT", "io_index": 1},
]

fanins: Dict[str, List[str]] = {
    "a_or":  ["in0", "in1"],
    "b_and": ["in0", "in1"],
    "c_not": ["b_and"],
    "fin":   ["a_or", "c_not"],
    "out0":  ["fin"],
    "out1":  ["a_or", "b_and"],  # to exercise median snapping & potential collision
}

if __name__ == "__main__":
    result = place(nodes, fanins)
    coords = result["coords"]
    levels = result["levels"]
    width, height = result["width"], result["height"]

    # --- invariants ---
    # Unique coords
    seen = set()
    for nid, xy in coords.items():
        assert xy not in seen, f"Duplicate coordinate for {nid}"
        seen.add(xy)

    # INs pinned left/even
    for nid in ("in0", "in1"):
        x, y = coords[nid]
        assert x == 0 and y % ROW_SPACING == 0

    # OUTs at right border/even
    right = width - 1
    for nid in ("out0", "out1"):
        x, y = coords[nid]
        assert x == right and y % ROW_SPACING == 0

    # Gate columns per level
    for nid in ("a_or", "b_and", "c_not", "fin"):
        lvl = levels[nid]
        x, y = coords[nid]
        assert x == column_x(lvl, COL_SPACING), f"{nid} wrong x for level {lvl}"
        assert 1 <= x <= right - 1

    # Deterministic stacking in level-1 column:
    # nearest_input_index(a_or) = 0, nearest_input_index(b_and) = 0 too,
    # fallback to node_id ordering => a_or must be placed above b_and (lower y).
    x_l1 = column_x(1, COL_SPACING)
    y_a = coords["a_or"][1]
    y_b = coords["b_and"][1]
    assert y_a < y_b, "Deterministic stacking failed for level-1 gates"

    # OUT[0] median: fin is the only fanin, so median y = y(fin); snapped to even row
    fin_y = coords["fin"][1]
    expected_out0_y = round(fin_y / ROW_SPACING) * ROW_SPACING
    assert coords["out0"][1] == expected_out0_y, "out0 not snapped to median even row"

    # OUT[1] median: fanins a_or and b_and; median is middle of their ys -> snap to even
    median = sorted([coords["a_or"][1], coords["b_and"][1]])
    med_val = (median[0] + median[1]) / 2
    expected_out1_y = round(med_val / ROW_SPACING) * ROW_SPACING
    # if collision with out0 at (right, expected_out1_y) occurs, the placer bumps it down
    y1 = coords["out1"][1]
    assert y1 == expected_out1_y or y1 == expected_out1_y + ROW_SPACING

    print("[PASS] placement invariants + deterministic stacking + median OUTs")
    print(f"width={width}, height={height}")
    for nid in sorted(coords.keys(), key=lambda k: (coords[k][0], coords[k][1], k)):
        print(f"{nid:>6} -> {coords[nid]}")
