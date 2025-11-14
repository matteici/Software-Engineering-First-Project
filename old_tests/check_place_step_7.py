#!/usr/bin/env python3
"""
check_place_pipeline.py — end-to-end test for Steps 2–7 via place.py + placement.py

What it does
------------
1) Builds a small source program (XOR-ish) as a string:
       either = in[0] or in[1];
       both   = in[0] and in[1];
       nboth  = not both;
       out[0] = either and nboth;
       out[1] = either or both;

2) Runs your pipeline up to topo (place.pipeline_build_topo),
   converts the netlist to placement inputs, and calls placement.place.

3) Asserts invariants:
   - Inputs at (x=0, even y = io_index*ROW_SPACING)
   - Outputs at (x = right border, even y), using median(fanins.y) snapping with collision bump
   - Gates strictly interior; gate columns follow x = 2 + COL_SPACING * level
   - Unique coordinates
   - Deterministic stacking: second placement call yields identical coords

4) Prints a compact summary.

Run:
  python3 check_place_step_7.py
"""

from typing import Dict, List, Tuple
import sys

import place                             # your integrated pipeline (steps 1–7 helpers inside)
import placement                         # the placement module (step 7)
from placement import (
    ROW_SPACING, COL_SPACING, column_x, GATE_KINDS
)

def _build_source() -> str:
    return (
        "either = in[0] or in[1];\n"
        "both   = in[0] and in[1];\n"
        "nboth  = not both;\n"
        "out[0] = either and nboth;\n"
        "out[1] = either or both;\n"
    )

def _median_even(ys: List[int]) -> int:
    ys = sorted(ys)
    m = len(ys)
    if m == 0:
        return 0
    if m % 2 == 1:
        med = ys[m // 2]
    else:
        med = (ys[m // 2 - 1] + ys[m // 2]) / 2
    # snap to nearest even row (multiple of ROW_SPACING)
    return int(round(med / ROW_SPACING) * ROW_SPACING)

def main() -> None:
    # -------- 2–6: Parse → Semantics → Netlist → Topo --------
    src = _build_source()
    nodes, edges, order = place.pipeline_build_topo(src)

    # -------- bridge to placement inputs --------
    p_nodes, fanins = place._netlist_to_placement_inputs(nodes, edges)  # uses robust inference

    topo_order_str = [str(n) for n in order] if order else None

    # -------- 7: Placement (twice, to check determinism) --------
    res1 = placement.place(p_nodes, fanins, topo_order=topo_order_str)
    res2 = placement.place(p_nodes, fanins, topo_order=topo_order_str)

    coords1: Dict[str, Tuple[int, int]] = res1["coords"]
    coords2: Dict[str, Tuple[int, int]] = res2["coords"]
    levels: Dict[str, int] = res1["levels"]
    width, height = res1["width"], res1["height"]
    right = width - 1

    # --- Deterministic stacking: coordinates identical across runs
    assert coords1 == coords2, "Placement is not deterministic across runs"

    # --- Uniqueness
    seen = set()
    for nid, xy in coords1.items():
        assert xy not in seen, f"Duplicate coordinate for {nid}"
        seen.add(xy)

    # --- Inputs pinned left/even at expected rows
    for n in p_nodes:
        if n["kind"] == "IN":
            nid = n["id"]
            x, y = coords1[nid]
            expect_y = (n["io_index"] or 0) * ROW_SPACING
            assert x == 0, f"{nid} must be at x=0; got {x}"
            assert y % ROW_SPACING == 0, f"{nid} y must be even; got {y}"
            assert y == expect_y, f"{nid} y={y}, expected {expect_y}"

    # --- Outputs at right border/even; y = median(fanins.y) snapped to even (with collision bump if needed)
    occupied = set(coords1.values())
    for n in p_nodes:
        if n["kind"] == "OUT":
            nid = n["id"]
            x, y = coords1[nid]
            assert x == right, f"{nid} must be at right border x={right}; got {x}"
            assert y % ROW_SPACING == 0, f"{nid} y must be even; got {y}"
            # median of fanins (by coords already placed)
            fanin_ys = [coords1[s][1] for s in fanins.get(nid, []) if s in coords1]
            expected_y = _median_even(fanin_ys) if fanin_ys else (n["io_index"] or 0) * ROW_SPACING
            if y != expected_y:
                # If it differs, it should be because expected slot is occupied (collision bump)
                assert (right, expected_y) in occupied, (
                    f"{nid} y deviates from median-even but no collision at {(right, expected_y)}"
                )

    # --- Gates interior and column rule x = 2 + COL_SPACING * level
    for n in p_nodes:
        if n["kind"] in GATE_KINDS:
            nid = n["id"]
            lvl = levels.get(nid, 1)
            x, y = coords1[nid]
            assert 1 <= x <= right - 1, f"{nid} must be interior; got x={x}"
            expect_x = column_x(lvl, COL_SPACING)
            assert x == expect_x, f"{nid} x={x}, expected {expect_x} for level {lvl}"

    print("[PASS] pipeline → topo → placement invariants + determinism")
    print(f"width={width}, height={height}")
    for nid in sorted(coords1.keys(), key=lambda k: (coords1[k][0], coords1[k][1], k)):
        print(f"{nid:>8} -> {coords1[nid]}")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
