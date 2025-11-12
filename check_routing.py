#!/usr/bin/env python3
"""
check_routing.py — self-contained checker for Step 8: Routing Primitives

If run without a path argument, it will generate a demo routed PNG using
routing.py (must be importable from the same directory), save it to ./output/,
and then run the checks on that image.

Usage:
  python3 check_routing.py                # generate demo then check it
  python3 check_routing.py output/foo.png # check an existing image
"""

import sys
import os
from datetime import datetime
from collections import deque
from typing import List, Tuple, Set
from PIL import Image

# --- Colors (match assignment spec) ---
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)       # bends/junctions
BLUE   = (0, 0, 255)     # straight wire / crossing
RED    = (255, 0, 0)     # NOT gate
YELLOW = (255, 255, 0)   # AND gate
GREEN  = (0, 255, 0)     # OR gate
GATE_COLORS = {RED, YELLOW, GREEN}

# --- Demo routes (keep in sync with routing.py if you customize) ---
ROUTES: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
    ((0,  2), (39,  2)),
    ((0,  6), (39, 10)),
    ((0, 12), (39, 12)),
    ((5,  9), (30,  4)),
]

# Try to import routing.py for demo generation
try:
    import routing
except ImportError:
    routing = None

# ----------------- helpers -----------------
def neighbors4(w: int, h: int, x: int, y: int):
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        xx, yy = x + dx, y + dy
        if 0 <= xx < w and 0 <= yy < h:
            yield (xx, yy), (dx, dy)

def deg_and_dirs(px, w: int, h: int, x: int, y: int):
    dirs = []
    for (xx, yy), (dx, dy) in neighbors4(w, h, x, y):
        if px[xx, yy] != WHITE:
            dirs.append((dx, dy))
    return len(dirs), set(dirs)

def is_straight_dirs(dirs) -> bool:
    return dirs in ({(1,0), (-1,0)}, {(0,1), (0,-1)})

def connected_nonwhite(px, w: int, h: int, s: Tuple[int,int], t: Tuple[int,int]) -> bool:
    (sx, sy), (tx, ty) = s, t
    if px[sx, sy] == WHITE or px[tx, ty] == WHITE:
        return False
    Q = deque([(sx, sy)])
    seen = {(sx, sy)}
    while Q:
        x, y = Q.popleft()
        if (x, y) == (tx, ty):
            return True
        for (xx, yy), _ in neighbors4(w, h, x, y):
            if (xx, yy) not in seen and px[xx, yy] != WHITE:
                seen.add((xx, yy))
                Q.append((xx, yy))
    return False

def ensure_outdir(path="output"):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def run_routing_demo() -> str:
    """
    Generate a demo routed image via routing.py and return its path.
    Requires routing.py in the same directory.
    """
    if routing is None:
        raise RuntimeError("routing.py not available to generate demo image.")
    grid = routing.Grid(routing.CANVAS_W, routing.CANVAS_H, routing.WHITE)
    occupied = routing.place_demo_gates(grid)
    for (s, t) in ROUTES:
        routing.route(grid, s, t, occupied)
    ensure_outdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("output", f"test_{stamp}.png")
    grid.img.save(out_path)
    print(f"[OK] demo image written to {out_path}")
    return out_path

# ----------------- main -----------------
def main():
    # 1) Get or create image path
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = run_routing_demo()

    # 2) Load image and prepare
    img = Image.open(path).convert("RGB")
    w, h = img.size
    px = img.load()

    # Precompute allowed BLUE endpoints (all declared route endpoints)
    allowed_blue_endpoints: Set[Tuple[int,int]] = set()
    for s, t in ROUTES:
        allowed_blue_endpoints.add(s)
        allowed_blue_endpoints.add(t)

    all_ok = True
    pass_count = 0

    # 3) Check each route
    for idx, (s, t) in enumerate(ROUTES, 1):
        ok = True
        msg = []

        # Connectivity via non-white pixels
        if not connected_nonwhite(px, w, h, s, t):
            ok = False
            msg.append("no connected non-white path")

        # Endpoints must not be gates
        if img.getpixel(s) in GATE_COLORS or img.getpixel(t) in GATE_COLORS:
            ok = False
            msg.append("endpoints on gates")

        # Local pixel rules:
        #  - BLUE: degree 2 and straight; degree 1 allowed ONLY if pixel is declared endpoint
        #  - BLACK: degree >= 2 (bend/junction); not isolated
        for y in range(h):
            for x in range(w):
                c = px[x, y]
                if c not in (BLUE, BLACK) and c not in GATE_COLORS:
                    continue

                if c in (BLUE, BLACK):
                    deg, dirs = deg_and_dirs(px, w, h, x, y)

                    if c == BLUE:
                        if deg == 1:
                            if (x, y) not in allowed_blue_endpoints:
                                ok = False
                                msg.append(f"dangling BLUE endpoint at {(x,y)}")
                        elif deg == 2:
                            if not is_straight_dirs(dirs):
                                ok = False
                                msg.append(f"BLUE bend at {(x,y)} (should be BLACK)")
                        elif deg > 2:
                            ok = False
                            msg.append(f"BLUE junction at {(x,y)} (should be BLACK)")

                    elif c == BLACK:
                        if deg < 2:
                            ok = False
                            msg.append(f"lonely BLACK at {(x,y)}")

                # Gate pixels can exist, but the checker just ensures they are not endpoints.
                # (Routing code already avoids stepping *through* gates.)

        print(
            ("[PASS]" if ok else "[FAIL]")
            + f" route {idx}: {s} -> {t}"
            + ("" if not msg else " | " + "; ".join(sorted(set(msg))))
        )
        pass_count += int(ok)
        all_ok &= ok

    print(f"\n[SUMMARY] {pass_count} passed, {len(ROUTES) - pass_count} failed (total {len(ROUTES)})")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
