#!/usr/bin/env python3
# routing.py — Step 8 (PPM output)

from PIL import Image
from collections import deque
from datetime import datetime
import os
from typing import List, Tuple, Optional, Dict, Set

WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
BLUE   = (0, 0, 255)
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN  = (0, 255, 0)
GATE_COLORS = {RED, YELLOW, GREEN}

class Grid:
    def __init__(self, w: int, h: int, bg: Tuple[int,int,int]=WHITE):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w, h), bg)
        self.px = self.img.load()
    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h
    def get(self, x: int, y: int) -> Tuple[int,int,int]:
        return self.px[x, y]
    def set(self, x: int, y: int, c: Tuple[int,int,int]) -> None:
        self.px[x, y] = c
    def neighbors4(self, x: int, y: int):
        # up, down, right
        for dx, dy in ((0,-1), (0,1), (1,0)):
            xx, yy = x+dx, y+dy
            if self.inside(xx, yy):
                yield (xx, yy)
                

def deg_and_axes(grid: Grid, x: int, y: int):
    """Degree considering only wire pixels (BLUE/BLACK)."""
    dirs = []
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        xx, yy = x+dx, y+dy
        if grid.inside(xx, yy):
            c = grid.get(xx, yy)
            if c == BLUE or c == BLACK:
                dirs.append((dx, dy))
    return len(dirs), set(dirs)

def _is_straight(dirs: set[tuple[int,int]]) -> bool:
    return dirs in ({(1,0),(-1,0)}, {(0,1),(0,-1)})

def _wire_halo(grid: Grid) -> Set[Tuple[int,int]]:
    """
    Compute a 4-neighbor, 1-radius 'soft' keep-out set around BLUE/BLACK wires.
    Gates are *not* added to the halo. Halo cells remain traversable as a fallback.
    """
    halo: Set[Tuple[int,int]] = set()
    for y in range(grid.h):
        for x in range(grid.w):
            c = grid.get(x, y)
            if c == BLUE or c == BLACK:
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    xx, yy = x + dx, y + dy
                    if grid.inside(xx, yy) and grid.get(xx, yy) == WHITE:
                        halo.add((xx, yy))
    return halo

def bfs_route(grid: Grid,
              start: Tuple[int,int],
              goal: Tuple[int,int],
              blocked: Set[Tuple[int,int]]) -> Optional[List[Tuple[int,int]]]:
    (sx, sy), (tx, ty) = start, goal
    if start == goal:
        return [start]
    Q = deque([start])
    parent = {start: None}
    while Q:
        x, y = Q.popleft()
        for xx, yy in grid.neighbors4(x, y):
            if (xx, yy) in parent:
                continue
            color = grid.get(xx, yy)
            if color in GATE_COLORS:
                continue
            # Honor blocked set but allow stepping onto the final goal
            if (xx, yy) in blocked and (xx, yy) != (tx, ty):
                continue
            parent[(xx, yy)] = (x, y)
            if (xx, yy) == (tx, ty):
                path = [(tx, ty)]
                cur = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            Q.append((xx, yy))
    return None

def draw_wire(grid: Grid, path: List[Tuple[int,int]]) -> None:
    if not path:
        return
    # pre-fill path pixels as BLUE (don’t overwrite gates)
    for (x, y) in path:
        if grid.get(x, y) == WHITE:
            grid.set(x, y, BLUE)

    # fix colors based on degree among wire pixels
    for (x, y) in path:
        if grid.get(x, y) in GATE_COLORS:
            continue
        deg, dirs = deg_and_axes(grid, x, y)

        if deg <= 1:
            # endpoints: stay BLUE unless they later become junctions
            continue
        if deg == 2:
            grid.set(x, y, BLUE if _is_straight(dirs) else BLACK)  # straight=BLUE, bend=BLACK
        elif deg == 3:
            grid.set(x, y, BLACK)  # T-junction
        elif deg == 4:
            grid.set(x, y, BLUE)   # 4-way crossing must be BLUE

def _route_via(grid: Grid,
               start: Tuple[int,int],
               goal: Tuple[int,int],
               via: Tuple[int,int],
               blocked: Set[Tuple[int,int]]) -> Optional[List[Tuple[int,int]]]:
    """Try s -> via -> t. Returns full path or None."""
    p1 = bfs_route(grid, start, via, blocked)
    if not p1:
        return None
    p2 = bfs_route(grid, via, goal, blocked)
    if not p2:
        return None
    return p1 + p2[1:]  # avoid duplicating 'via'
'''
def route(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occupied: Optional[Set[Tuple[int,int]]] = None,
    prefer_soft_halo: bool = False,
) -> bool:
    """
    Route a single net from start to goal.

    - `occupied` are permanent blocks (gate centers, etc.).
    - Existing wires (BLUE/BLACK) are ALSO treated as blocked so that
      one line = one signal (no sharing).
    - If `prefer_soft_halo` is True, we also block *adjacent* pixels
      around existing wires (soft keep-out) to create whitespace.
    """
    blocked: Set[Tuple[int,int]] = set(occupied or [])

    # 1) Treat existing wires as blocked so later nets cannot reuse them.
    wire_pixels: Set[Tuple[int,int]] = set()
    for y in range(grid.h):
        for x in range(grid.w):
            c = grid.get(x, y)
            if c == BLUE or c == BLACK:
                wire_pixels.add((x, y))
                blocked.add((x, y))
                if prefer_soft_halo:
                    # Add a 4-neighbor halo around wires (only over WHITE pixels)
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x + dx, y + dy
                        if grid.inside(nx, ny) and grid.get(nx, ny) == WHITE:
                            blocked.add((nx, ny))

    # 2) Don't block the actual start/goal pixels
    blocked.discard(start)
    blocked.discard(goal)

    # 3) Try direct BFS
    path = bfs_route(grid, start, goal, blocked)

    # 4) Fallback: HV/VH via Manhattan bend points
    if path is None:
        (sx, sy), (tx, ty) = start, goal
        for via in ((sx, ty), (tx, sy)):  # HV then VH
            cand = _route_via(grid, start, goal, via, blocked)
            if cand:
                path = cand
                break

    if path is None:
        return False

    draw_wire(grid, path)
    return True
'''

def route(grid: Grid,
          start: Tuple[int, int],
          goal: Tuple[int, int],
          occupied: Optional[Set[Tuple[int,int]]] = None) -> bool:
    """
    Route a single net from start to goal.

    - `occupied`: permanent blocks (gate centers, etc.)
    - We ADD a halo around existing wires (BLUE/BLACK):
        every WHITE pixel 4-neighbor to a wire becomes blocked,
        so there is always at least one white cell around each wire.
    """
    # Start from externally-occupied pixels (gate centers, etc.)
    blocked: Set[Tuple[int,int]] = set(occupied or [])

    # --- HALO AROUND EXISTING WIRES ---
    for y in range(grid.h):
        for x in range(grid.w):
            c = grid.get(x, y)
            if c == BLUE or c == BLACK:
                # For each wire pixel, block its 4-neighbors if they are WHITE
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    nx, ny = x + dx, y + dy
                    if grid.inside(nx, ny) and grid.get(nx, ny) == WHITE:
                        blocked.add((nx, ny))
    # Don't block the pins themselves
    blocked.discard(start)
    blocked.discard(goal)

    # 1) direct BFS
    path = bfs_route(grid, start, goal, blocked)

    # 2) fallback: try HV then VH via the two Manhattan bend points
    if path is None:
        (sx, sy), (tx, ty) = start, goal
        for via in ((sx, ty), (tx, sy)):  # HV then VH
            cand = _route_via(grid, start, goal, via, blocked)
            if cand:
                path = cand
                break

    if path is None:
        return False

    draw_wire(grid, path)
    return True


# ---------------- Demo helpers (independent checker uses these) ------------

CANVAS_W, CANVAS_H = 40, 16

def place_demo_gates(grid: Grid) -> Set[Tuple[int,int]]:
    gates = {(12, 6), (20, 8), (28, 4)}
    grid.set(12, 6, RED)
    grid.set(20, 8, YELLOW)
    grid.set(28, 4, GREEN)
    return gates

DEMO_ROUTES = [
    ((0,  2), (39,  2)),
    ((0,  6), (39, 10)),
    ((0, 12), (39, 12)),
    ((5,  9), (30,  4)),
]

def ensure_outdir(path="output"):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def main():
    grid = Grid(CANVAS_W, CANVAS_H, WHITE)
    occupied = place_demo_gates(grid)
    for (s, t) in DEMO_ROUTES:
        ok = route(grid, s, t, occupied)
        if not ok:
            print(f"[WARN] Could not route from {s} to {t}")
    ensure_outdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("output", f"test_{stamp}.ppm")
    # PIL writes binary PPM (P6) by default
    grid.img.save(out_path, format="PPM")
    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
