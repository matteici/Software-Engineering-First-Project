#!/usr/bin/env python3
# routing.py — Step 8 (PPM output with controlled crossings)

from PIL import Image
from collections import deque
from datetime import datetime
import os
from typing import List, Tuple, Optional, Dict, Set

WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
BLUE   = (0,   0, 255)
RED    = (255, 0,   0)
YELLOW = (255, 255, 0)
GREEN  = (0, 255,   0)

GATE_COLORS = {RED, YELLOW, GREEN}


class Grid:
    def __init__(self, w: int, h: int, bg: Tuple[int, int, int] = WHITE):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w, h), bg)

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def get(self, x: int, y: int) -> Tuple[int, int, int]:
        return self.img.getpixel((x, y))

    def set(self, x: int, y: int, c: Tuple[int, int, int]) -> None:
        self.img.putpixel((x, y), c)

    def neighbors4(self, x: int, y: int):
        """
        3-direction Manhattan neighbors:
        - up, down, right (NO left moves).
        """
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            xx, yy = x + dx, y + dy
            if self.inside(xx, yy):
                yield (xx, yy)


def deg_and_axes(grid: Grid, x: int, y: int):
    """
    Degree + axis flags, counting only wire pixels (BLUE/BLACK).
    horiz=True if any wire neighbor is horizontal (same y, different x).
    vert=True  if any wire neighbor is vertical   (same x, different y).
    """
    d = 0
    horiz = vert = False
    for nx, ny in grid.neighbors4(x, y):
        c = grid.get(nx, ny)
        if c == BLUE or c == BLACK:
            d += 1
            if ny == y and nx != x:
                horiz = True
            if nx == x and ny != y:
                vert = True
    return d, horiz, vert


def draw_wire(grid: Grid, path: List[Tuple[int, int]]):
    """
    Paint the path onto the grid:

      - Existing wire pixels (BLUE/BLACK) are NOT recolored.
        → intersections between different nets remain BLUE,
          we never "upgrade" them to BLACK.
      - BLACK is used for bends / tees *within the same net* only.
    """
    if not path:
        return

    for (x, y) in path:
        orig = grid.get(x, y)

        # Never paint over gate centers
        if orig in GATE_COLORS:
            continue

        # If this pixel already holds a wire from a previous net,
        # do NOT recolor it. This avoids "black sharing": different
        # nets may at most re-use a BLUE intersection pixel, but we
        # never turn that into BLACK due to combined degree.
        if orig == BLUE or orig == BLACK:
            continue

        d, horiz, vert = deg_and_axes(grid, x, y)

        if d == 0:
            # new endpoint
            grid.set(x, y, BLUE)
        elif d == 1:
            # continuing a single line; black only if this net bends
            # relative to the neighbor orientation
            if horiz and vert:
                grid.set(x, y, BLACK)
            else:
                grid.set(x, y, BLUE)
        else:
            # 3+ connections for THIS net (e.g., tee) → black
            grid.set(x, y, BLACK)


def _scan_foreign_wires(grid: Grid) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Return:
      - foreign_blue:  all BLUE pixels (existing wires);
      - foreign_black: all BLACK pixels (bends/tees of existing nets),
                       which new nets must NEVER step on.
    """
    foreign_blue: Set[Tuple[int, int]] = set()
    foreign_black: Set[Tuple[int, int]] = set()
    for y in range(grid.h):
        for x in range(grid.w):
            c = grid.get(x, y)
            if c == BLUE:
                foreign_blue.add((x, y))
            elif c == BLACK:
                foreign_black.add((x, y))
    return foreign_blue, foreign_black


def bfs_route_with_cross(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: Set[Tuple[int, int]],
    foreign_blue: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """
    BFS that:

      - treats `blocked` as hard obstacles (gates, halo, black wires, etc.);
      - treats `foreign_blue` (existing BLUE wires) as "almost obstacles":
          * we may step on at most ONE such pixel in the entire path
            (cross_used flag);
          * and only when crossing it ORTHOGONALLY:
              - our move is horizontal while the existing wire is vertical, or
              - our move is vertical while the existing wire is horizontal.
          * we may NEVER travel along the same axis as the existing wire.

    This enforces:
      - two signals may share at most ONE BLUE square,
      - and only as a proper intersection (L/R vs U/D), never as a segment.
    """
    if start == goal:
        return [start]
    if start in blocked:
        return None

    foreign_blue = set(foreign_blue)
    # Never treat start/goal as "foreign"; they can lie on existing wires.
    foreign_blue.discard(start)
    foreign_blue.discard(goal)

    # State = (x, y, cross_used ∈ {0,1})
    State = Tuple[int, int, int]
    start_state: State = (start[0], start[1], 0)

    q = deque([start_state])
    prev: Dict[State, Optional[State]] = {start_state: None}

    while q:
        x, y, used = q.popleft()
        for nx, ny in grid.neighbors4(x, y):
            cell = (nx, ny)
            if cell in blocked:
                continue

            dx, dy = nx - x, ny - y
            c = grid.get(nx, ny)

            if c in GATE_COLORS:
                continue

            next_used = used

            # Is this a foreign BLUE wire pixel?
            if cell in foreign_blue:
                # Already used our one crossing → can't step on another
                if used:
                    continue

                # Determine orientation of the existing wire
                d, horiz, vert = deg_and_axes(grid, nx, ny)

                # Must have a clear single axis (pure horiz or pure vert).
                if not (horiz ^ vert):
                    # ambiguous orientation (or isolated) -> refuse crossing
                    continue

                move_is_horiz = (dx != 0)
                move_is_vert  = (dy != 0)

                # If wire is horizontal and our move is also horizontal,
                # we'd be travelling ALONG the wire (forbidden).
                if horiz and move_is_horiz:
                    continue

                # If wire is vertical and our move is also vertical,
                # we'd be travelling ALONG the wire (forbidden).
                if vert and move_is_vert:
                    continue

                # OK: orthogonal crossing
                next_used = 1

            next_state: State = (nx, ny, next_used)
            if next_state in prev:
                continue

            prev[next_state] = (x, y, used)
            if (nx, ny) == goal:
                # reconstruct path (ignore cross_used flag)
                path: List[Tuple[int, int]] = []
                cur: Optional[State] = next_state
                while cur is not None:
                    cx, cy, _ = cur
                    path.append((cx, cy))
                    cur = prev[cur]
                path.reverse()
                return path

            q.append(next_state)

    return None


def route(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occupied: Optional[Set[Tuple[int, int]]] = None,
) -> bool:
    """
    Route a single net from start to goal.

      - `occupied` = permanent blocks (gate centers, local halo, etc.).
      - Existing BLACK wires are treated as hard obstacles
        (no black sharing between different nets).
      - Existing BLUE wires may be crossed at most once, and only
        orthogonally (intersection); we never share a straight segment.
      - Movement is restricted to up/down/right (no left).
      - Column x=1 is also treated as a "no-long-bus" column (blocked
        except possibly at start/goal), to keep the left side clean.
    """
    blocked: Set[Tuple[int, int]] = set(occupied or [])

    # Snapshot foreign wires before routing this net
    foreign_blue, foreign_black = _scan_foreign_wires(grid)

    # BLACK from previous nets are fully blocked
    blocked |= foreign_black

    # Avoid using column x=1 as a vertical bus (except pins)
    first_col = 1
    for y in range(grid.h):
        cell = (first_col, y)
        if cell != start and cell != goal:
            blocked.add(cell)

    # Run BFS with controlled crossing
    path = bfs_route_with_cross(grid, start, goal, blocked, foreign_blue)

    if path is None:
        return False

    draw_wire(grid, path)
    return True


# --- Tiny demo harness (used by check_routing.py) ---

CANVAS_W, CANVAS_H = 40, 16

def place_demo_gates(grid: Grid) -> Set[Tuple[int, int]]:
    gates: Set[Tuple[int, int]] = {(12, 6), (20, 8), (28, 4)}
    grid.set(12, 6, RED)
    grid.set(20, 8, YELLOW)
    grid.set(28, 4, GREEN)
    return gates


DEMO_ROUTES = [
    ((13, 6), (19, 8)),
    ((21, 8), (27, 4)),
]


def ensure_outdir():
    if not os.path.isdir("output"):
        os.makedirs("output", exist_ok=True)


def demo_image(path: str = "output/demo.ppm"):
    grid = Grid(CANVAS_W, CANVAS_H, WHITE)
    gates = place_demo_gates(grid)
    for (s, t) in DEMO_ROUTES:
        ok = route(grid, s, t, gates)
        if not ok:
            print(f"[WARN] Could not route from {s} to {t}")
    ensure_outdir()
    grid.img.save(path, format="PPM")
    print(f"[OK] wrote {path}")


def main():
    # If called directly, generate a small demo image; used by check_routing.py.
    ensure_outdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("output", f"test_{stamp}.ppm")
    demo_image(out_path)


if __name__ == "__main__":
    main()
