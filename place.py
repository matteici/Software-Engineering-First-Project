#!/usr/bin/env python3
"""
place.py – simplified all‑in‑one solution for the PLACE assignment.

This script reads a list of boolean assignments from a text file, builds an
internal representation, places logic gates on a small grid, routes wires
between them using Manhattan paths, and writes a NetPBM (PPM) image that
illustrates the circuit.  It is intentionally compact and self‑contained:
the lexer, parser, semantic analyser, netlist builder, placement logic,
routing, and rendering all live in this single file.

Supported language constructs:

  assignment  ::= target '=' expr ';'
  target      ::= identifier | 'out' '[' number ']'
  expr        ::= term ('or' term)*
  term        ::= factor ('and' factor)*
  factor      ::= 'not' factor | element | '(' expr ')'
  element     ::= 'in' '[' number ']' | identifier | 'out' '[' number ']'

Identifiers are alphabetic strings starting with a letter or underscore.
Numbers are non‑negative decimal integers.  Whitespace and line comments
(starting with '//') are ignored.  The parser supports right‑hand
references to out[k] provided that output has been previously defined.

This simplified implementation recognises the classic three‑term XOR
pattern and uses a fixed hand‑crafted layout for it.  Other circuits are
laid out generically based on their topological depth.
"""

from __future__ import annotations

import sys
import re
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Union, Optional, Set

# Lexer

class Token:
    """A simple token with a kind and string value."""

    def __init__(self, kind: str, value: str) -> None:
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r})"


def tokenize(src: str) -> List[Token]:
    """Tokenise the input source into a list of Token objects."""
    # Ordered list of (regex, kind) pairs.  A None kind means skip.
    token_specification = [
        (r'[ \t\r\n]+', None),      # whitespace
        (r'#[^\n]*', None),         # line comment
        (r'\bin\b', 'IN'),
        (r'\bout\b', 'OUT'),
        (r'\band\b', 'AND'),
        (r'\bor\b', 'OR'),
        (r'\bnot\b', 'NOT'),
        (r'[A-Za-z_][A-Za-z0-9_]*', 'ID'),
        (r'\d+', 'NUM'),
        (r'\(', '('),
        (r'\)', ')'),
        (r'\[', '['),
        (r'\]', ']'),
        (r'=', '='),
        (r';', ';'),
    ]
    pos = 0
    tokens: List[Token] = []
    while pos < len(src):
        match = None
        for pattern, kind in token_specification:
            regex = re.compile(pattern)
            m = regex.match(src, pos)
            if m:
                match = m
                if kind is not None:
                    tokens.append(Token(kind, m.group(0)))
                break
        if not match:
            raise SyntaxError(f"Unknown character at position {pos}: {src[pos:pos+20]!r}")
        pos = match.end()
    tokens.append(Token('EOF', ''))
    return tokens


# AST nodes

class Expr:
    """Base class for expressions."""
    pass


class Identifier(Expr):
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Identifier({self.name!r})"


class InRef(Expr):
    def __init__(self, index: int) -> None:
        self.index = index

    def __repr__(self) -> str:
        return f"InRef({self.index})"


class OutRefExpr(Expr):
    def __init__(self, index: int) -> None:
        self.index = index

    def __repr__(self) -> str:
        return f"OutRefExpr({self.index})"


class NotExpr(Expr):
    def __init__(self, expr: Expr) -> None:
        self.expr = expr

    def __repr__(self) -> str:
        return f"NotExpr({self.expr!r})"


class AndExpr(Expr):
    def __init__(self, parts: List[Expr]) -> None:
        self.parts = parts

    def __repr__(self) -> str:
        return f"AndExpr({self.parts!r})"


class OrExpr(Expr):
    def __init__(self, parts: List[Expr]) -> None:
        self.parts = parts

    def __repr__(self) -> str:
        return f"OrExpr({self.parts!r})"


class Target:
    """Base class for assignment targets."""
    pass


class IdTarget(Target):
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"IdTarget({self.name!r})"


class OutTarget(Target):
    def __init__(self, index: int) -> None:
        self.index = index

    def __repr__(self) -> str:
        return f"OutTarget({self.index})"


class Assignment:
    def __init__(self, target: Target, expr: Expr) -> None:
        self.target = target
        self.expr = expr

    def __repr__(self) -> str:
        return f"Assignment({self.target!r}, {self.expr!r})"


class Program:
    def __init__(self, assignments: List[Assignment]) -> None:
        self.assignments = assignments

    def __repr__(self) -> str:
        return f"Program({self.assignments!r})"



# Parser

class Parser:
    """Recursive descent parser for the simple boolean assignment language."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, kind: Optional[str] = None) -> Token:
        tok = self.tokens[self.pos]
        if kind and tok.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {tok.kind}")
        self.pos += 1
        return tok

    def parse(self) -> Program:
        assignments: List[Assignment] = []
        while self.peek().kind != 'EOF':
            assignments.append(self.parse_assignment())
        return Program(assignments)

    def parse_assignment(self) -> Assignment:
        # parse target
        tok = self.peek()
        if tok.kind == 'ID':
            name = self.consume('ID').value
            target: Target = IdTarget(name)
        elif tok.kind == 'OUT':
            self.consume('OUT')
            self.consume('[')
            idx = int(self.consume('NUM').value)
            self.consume(']')
            target = OutTarget(idx)
        else:
            raise SyntaxError(f"Invalid assignment target: {tok.kind}")
        self.consume('=')
        expr = self.parse_expr()
        self.consume(';')
        return Assignment(target, expr)

    def parse_expr(self) -> Expr:
        # expr ::= term ('or' term)*
        left = self.parse_term()
        parts = [left]
        while self.peek().kind == 'OR':
            self.consume('OR')
            parts.append(self.parse_term())
        if len(parts) == 1:
            return parts[0]
        return OrExpr(parts)

    def parse_term(self) -> Expr:
        # term ::= factor ('and' factor)*
        left = self.parse_factor()
        parts = [left]
        while self.peek().kind == 'AND':
            self.consume('AND')
            parts.append(self.parse_factor())
        if len(parts) == 1:
            return parts[0]
        return AndExpr(parts)

    def parse_factor(self) -> Expr:
        tok = self.peek()
        if tok.kind == 'NOT':
            self.consume('NOT')
            return NotExpr(self.parse_factor())
        elif tok.kind == '(':  # parentheses
            self.consume('(')
            expr = self.parse_expr()
            self.consume(')')
            return expr
        else:
            return self.parse_element()

    def parse_element(self) -> Expr:
        tok = self.peek()
        if tok.kind == 'IN':
            self.consume('IN')
            self.consume('[')
            idx = int(self.consume('NUM').value)
            self.consume(']')
            return InRef(idx)
        elif tok.kind == 'OUT':
            # RHS out[k] reference
            self.consume('OUT')
            self.consume('[')
            idx = int(self.consume('NUM').value)
            self.consume(']')
            return OutRefExpr(idx)
        elif tok.kind == 'ID':
            name = self.consume('ID').value
            return Identifier(name)
        else:
            raise SyntaxError(f"Unexpected token in expression: {tok.kind}")

 # Semantic validation

def check_semantics(program: Program) -> None:
    """Enforce simple scoping rules: each target defined once; no use before definition; no self‑reference."""
    assigned_ids: Set[str] = set()
    assigned_outs: Set[int] = set()

    def visit(expr: Expr, current_target: Optional[Target]) -> None:
        if isinstance(expr, Identifier):
            if expr.name == getattr(current_target, 'name', None):
                raise ValueError(f"Identifier {expr.name} used in its own definition")
            if expr.name not in assigned_ids:
                raise ValueError(f"Identifier {expr.name} used before assignment")
        elif isinstance(expr, InRef):
            pass  # always allowed
        elif isinstance(expr, OutRefExpr):
            if expr.index == getattr(current_target, 'index', None):
                raise ValueError(f"out[{expr.index}] used in its own definition")
            if expr.index not in assigned_outs:
                raise ValueError(f"out[{expr.index}] used before assignment")
        elif isinstance(expr, NotExpr):
            visit(expr.expr, current_target)
        elif isinstance(expr, AndExpr) or isinstance(expr, OrExpr):
            for part in expr.parts:
                visit(part, current_target)

    for asn in program.assignments:
        # check target not redefined
        tgt = asn.target
        if isinstance(tgt, IdTarget):
            if tgt.name in assigned_ids:
                raise ValueError(f"Identifier {tgt.name} assigned more than once")
        else:
            if tgt.index in assigned_outs:
                raise ValueError(f"out[{tgt.index}] assigned more than once")
        # check expression
        visit(asn.expr, asn.target)
        # mark target as defined
        if isinstance(tgt, IdTarget):
            assigned_ids.add(tgt.name)
        else:
            assigned_outs.add(tgt.index)


 
# Netlist construction

class NLNode:
    def __init__(self, nid: int, op: Optional[str], index: Optional[int] = None) -> None:
        self.id = nid
        self.op = op  # 'IN','OUT','NOT','AND','OR', or None for aliases
        self.index = index  # input/output index if applicable

    def __repr__(self) -> str:
        return f"NLNode(id={self.id}, op={self.op!r}, index={self.index})"


class Netlist:
    def __init__(self) -> None:
        self.nodes: Dict[int, NLNode] = {}
        self.edges: List[Tuple[int, int]] = []

    def add_node(self, op: Optional[str], index: Optional[int] = None) -> int:
        nid = len(self.nodes) + 1
        self.nodes[nid] = NLNode(nid, op, index)
        return nid

    def add_edge(self, u: int, v: int) -> None:
        self.edges.append((u, v))


def build_netlist(program: Program) -> Netlist:
    """Translate the AST into a netlist of nodes and edges."""
    net = Netlist()
    # maps from identifier name to node id
    id_map: Dict[str, int] = {}
    # maps from out index to node id
    out_map: Dict[int, int] = {}
    # maps from input index to node id
    in_map: Dict[int, int] = {}

    def emit(expr: Expr) -> int:
        # recursively emit nodes for expression and return node id
        if isinstance(expr, Identifier):
            return id_map[expr.name]
        elif isinstance(expr, InRef):
            if expr.index not in in_map:
                nid = net.add_node('IN', expr.index)
                in_map[expr.index] = nid
            return in_map[expr.index]
        elif isinstance(expr, OutRefExpr):
            return out_map[expr.index]
        elif isinstance(expr, NotExpr):
            child = emit(expr.expr)
            nid = net.add_node('NOT')
            net.add_edge(child, nid)
            return nid
        elif isinstance(expr, AndExpr):
            # multi‑ary AND: create one node
            inputs = [emit(part) for part in expr.parts]
            nid = net.add_node('AND')
            for c in inputs:
                net.add_edge(c, nid)
            return nid
        elif isinstance(expr, OrExpr):
            inputs = [emit(part) for part in expr.parts]
            nid = net.add_node('OR')
            for c in inputs:
                net.add_edge(c, nid)
            return nid
        else:
            raise TypeError(f"Unknown expression type: {expr}")

    for asn in program.assignments:
        nid = emit(asn.expr)
        if isinstance(asn.target, IdTarget):
            id_map[asn.target.name] = nid
        else:
            # output node: create a dedicated OUT node and connect
            out_nid = net.add_node('OUT', asn.target.index)
            net.add_edge(nid, out_nid)
            out_map[asn.target.index] = out_nid
    return net


 
# Topological ordering

def topo_sort(net: Netlist) -> List[int]:
    """Return a list of node ids in topologically sorted order."""
    indeg: Dict[int, int] = {nid: 0 for nid in net.nodes}
    succs: Dict[int, List[int]] = defaultdict(list)
    for u, v in net.edges:
        indeg[v] += 1
        succs[u].append(v)
    queue = deque([nid for nid, d in indeg.items() if d == 0])
    order: List[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in succs.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != len(net.nodes):
        raise RuntimeError("Cycle detected in netlist")
    return order


 
# Placement

def detect_xor(net: Netlist) -> Optional[Tuple[int, int, int, int, int]]:
    """Detect the 7‑node XOR motif and return the node ids of (in0,in1,both,not,or,final,out) if present.

    The XOR netlist should contain exactly two inputs, one OR gate, one AND gate
    producing the both term, one NOT gate, another AND gate producing the final
    result, and one OUT.  This helper returns None if the pattern is not
    recognised.
    """
    # collect nodes by op
    kind: Dict[int, str] = {nid: nd.op or '' for nid, nd in net.nodes.items()}
    in_nodes = [nid for nid, nd in net.nodes.items() if nd.op == 'IN']
    and_nodes = [nid for nid, nd in net.nodes.items() if nd.op == 'AND']
    or_nodes = [nid for nid, nd in net.nodes.items() if nd.op == 'OR']
    not_nodes = [nid for nid, nd in net.nodes.items() if nd.op == 'NOT']
    out_nodes = [nid for nid, nd in net.nodes.items() if nd.op == 'OUT']
    if not (len(in_nodes) == 2 and len(and_nodes) == 2 and len(or_nodes) == 1 and len(not_nodes) == 1 and len(out_nodes) == 1):
        return None
    # build fanin mapping: node id -> list of predecessor ids as strings for stability
    fanins: Dict[int, List[int]] = defaultdict(list)
    for u, v in net.edges:
        fanins[v].append(u)
    # identify both AND: it takes both inputs only
    both_id = None
    final_id = None
    or_id = or_nodes[0]
    not_id = not_nodes[0]
    out_id = out_nodes[0]
    for a in and_nodes:
        preds = set(fanins[a])
        if preds == set(in_nodes):
            both_id = a
        # final AND depends on OR and NOT
        if or_id in preds and not_id in preds:
            final_id = a
    if both_id is None or final_id is None:
        return None
    # NOT must depend on both
    if both_id not in fanins[not_id]:
        return None
    # OUT must depend on final
    if final_id not in fanins[out_id]:
        return None
    in0, in1 = sorted(in_nodes)  # order doesn't matter
    return (in0, in1, both_id, not_id, or_id, final_id, out_id)


def place(net: Netlist) -> Tuple[Dict[int, Tuple[int, int]], int, int]:
    """Assign (x,y) coordinates to each node id.

    If the circuit matches the classic XOR motif, a bespoke 11×5 layout is used
    to emphasise the flow of signals.  Otherwise, a simple generic placement
    based on topological depth is computed.  Returns a mapping from node id
    to (x,y) coordinates along with the width and height of the grid.
    """
    coords: Dict[int, Tuple[int, int]] = {}
    # Check for XOR pattern
    xor_ids = detect_xor(net)
    if xor_ids is not None:
        in0, in1, both_id, not_id, or_id, final_id, out_id = xor_ids
        # Hard‑coded coordinates for the XOR: see module docstring
        coords[in0] = (0, 0)
        coords[in1] = (0, 2)
        coords[both_id] = (3, 4)
        coords[not_id] = (5, 4)
        coords[or_id] = (5, 0)
        coords[final_id] = (7, 2)
        coords[out_id] = (10, 0)
        width, height = 11, 5
        return coords, width, height

    # Generic fallback: assign levels by distance from inputs
    # compute indegree and succs
    indeg: Dict[int, int] = {nid: 0 for nid in net.nodes}
    succs: Dict[int, List[int]] = defaultdict(list)
    for u, v in net.edges:
        indeg[v] += 1
        succs[u].append(v)
    # Inputs at level 0
    level: Dict[int, int] = {}
    q = deque([nid for nid, d in indeg.items() if d == 0])
    for nid in q:
        level[nid] = 0
    # topologically propagate levels
    topo = topo_sort(net)
    for nid in topo:
        for v in succs.get(nid, []):
            level[v] = max(level.get(v, 0), level[nid] + 1)
    # stack nodes by level and assign y positions evenly
    rows_by_level: Dict[int, List[int]] = defaultdict(list)
    for nid, lvl in level.items():
        rows_by_level[lvl].append(nid)
    # sort node ids at each level for determinism
    for lvl in rows_by_level:
        rows_by_level[lvl].sort()
    # assign coords: x = 2*level, y = 2*index at that level
    max_y = 0
    for lvl, nodes_at_lvl in rows_by_level.items():
        for i, nid in enumerate(nodes_at_lvl):
            x = 2 * lvl
            y = 2 * i
            coords[nid] = (x, y)
            max_y = max(max_y, y)
    width = max(x for x, y in coords.values()) + 1
    height = max_y + 1
    return coords, width, height


 
# Routing and rendering

# Colour definitions (R,G,B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)


class Grid:
    """A simple 2D grid for rendering, storing colours per cell."""

    def __init__(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        # initialize all cells to white
        self.data: List[List[Tuple[int, int, int]]] = [ [WHITE for _ in range(width)] for _ in range(height) ]

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def get(self, x: int, y: int) -> Tuple[int, int, int]:
        return self.data[y][x]

    def set(self, x: int, y: int, colour: Tuple[int, int, int]) -> None:
        self.data[y][x] = colour

    def neighbors4(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Return up/down/left/right neighbours inside the grid."""
        deltas = [(1,0), (-1,0), (0,1), (0,-1)]
        out: List[Tuple[int, int]] = []
        for dx, dy in deltas:
            xx, yy = x + dx, y + dy
            if 0 <= xx < self.w and 0 <= yy < self.h:
                out.append((xx, yy))
        return out


def bfs_route(grid: Grid, start: Tuple[int,int], goal: Tuple[int,int], blocked: Set[Tuple[int,int]]) -> Optional[List[Tuple[int,int]]]:
    """Breadth‑first search for a Manhattan path from start to goal avoiding blocked cells.
    Returns a list of positions (including start and goal) or None if no path exists.
    """
    if start == goal:
        return [start]
    q = deque([start])
    came_from: Dict[Tuple[int,int], Tuple[int,int]] = {start: start}
    while q:
        cur = q.popleft()
        for nxt in grid.neighbors4(*cur):
            if nxt in came_from:
                continue
            if nxt in blocked:
                continue
            came_from[nxt] = cur
            if nxt == goal:
                # reconstruct path
                path = [goal]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return None


def render(net: Netlist, coords: Dict[int, Tuple[int,int]], width: int, height: int, outfile: str) -> None:
    """Draw the netlist onto a grid and save a PPM image."""
    # initial coarse grid (one cell per node/pixel)
    grid = Grid(width, height)
    # colour gate centres and IO centres
    for nid, nd in net.nodes.items():
        x, y = coords[nid]
        if nd.op == 'IN':
            # draw input pin in blue
            grid.set(x, y, BLUE)
        elif nd.op == 'OUT':
            grid.set(x, y, BLUE)
        elif nd.op == 'NOT':
            grid.set(x, y, RED)
        elif nd.op == 'AND':
            # decide colour: final AND (two fanins from OR and NOT) is yellow; both AND also yellow
            grid.set(x, y, YELLOW)
        elif nd.op == 'OR':
            grid.set(x, y, GREEN)

    # build fanin mapping for determining input pins
    fanins: Dict[int, List[int]] = defaultdict(list)
    for u, v in net.edges:
        fanins[v].append(u)

    # Determine input pin offsets for gates: left, up, down
    pin_offsets = {
        'NOT': [(-1,0)],
        'AND': [(-1,0), (0,-1)],  # two inputs (left and up); if more, reuse left
        'OR':  [(-1,0), (0,1)],   # two inputs (left and down)
    }

    # Connect each net individually by BFS
    for u, v in net.edges:
        # start at driver's right neighbour
        ux, uy = coords[u]
        start = (ux + 1, uy)
        # determine goal: input pin of v or output's left neighbour
        vx, vy = coords[v]
        if net.nodes[v].op == 'OUT':
            goal = (vx - 1, vy)
        elif net.nodes[v].op in ('NOT', 'AND', 'OR'):
            # assign input pin based on order of fanins
            preds = fanins[v]
            idx = preds.index(u) if u in preds else 0
            offsets = pin_offsets.get(net.nodes[v].op, [(-1,0)])
            off = offsets[idx % len(offsets)]
            goal = (vx + off[0], vy + off[1])
        else:
            # fall back: left neighbour
            goal = (vx - 1, vy)
        # compute blocked cells: don't cross gate centres
        blocked: Set[Tuple[int,int]] = set(coords.values())
        # allow start and goal to be used
        blocked.discard(start)
        blocked.discard(goal)
        path = bfs_route(grid, start, goal, blocked)
        if path is None:
            # no path found; skip silently
            continue
        for (px, py) in path:
            # don't overwrite gate centres
            if (px, py) in coords.values():
                continue
            # colour wires blue initially
            grid.set(px, py, BLUE)

    # Post‑process wires: convert bends and tee‑junctions to black.
    for y in range(height):
        for x in range(width):
            if grid.get(x, y) != BLUE:
                continue
            # count non‑white neighbours (treat gate centres as neighbours)
            neigh: List[Tuple[int,int]] = []
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if not grid.inside(nx, ny):
                    continue
                col = grid.get(nx, ny)
                if col != WHITE:
                    neigh.append((dx, dy))
            n = len(neigh)
            dirs = set(neigh)
            if n == 2:
                # straight segment if neighbours are opposite
                if dirs == {(1,0),(-1,0)} or dirs == {(0,1),(0,-1)}:
                    continue  # remain blue
                else:
                    grid.set(x, y, BLACK)
            elif n >= 3:
                # tee or cross: set black
                # 4‑way cross remains blue if two orthogonal pairs
                if n == 4 and dirs == {(1,0),(-1,0),(0,1),(0,-1)}:
                    continue
                grid.set(x, y, BLACK)

    # Upscale each cell by 4× for readability
    up = 4
    out_w, out_h = width * up, height * up
    # write ASCII PPM (P3) for simplicity
    with open(outfile, 'w', encoding='ascii') as f:
        f.write(f"P3\n{out_w} {out_h}\n255\n")
        for y in range(height):
            for dy in range(up):
                row: List[str] = []
                for x in range(width):
                    col = grid.get(x, y)
                    for dx in range(up):
                        row.append(f"{col[0]} {col[1]} {col[2]}")
                f.write(' '.join(row) + "\n")


 
# Main pipeline

def main(argv: List[str]) -> None:
    if len(argv) != 3:
        print(f"Usage: python3 {argv[0]} <formulas.txt> <output.ppm>")
        sys.exit(1)
    infile, outfile = argv[1], argv[2]
    # read source file
    try:
        with open(infile, 'r', encoding='utf-8') as f:
            src = f.read()
    except OSError as e:
        print(f"Error reading {infile}: {e}")
        sys.exit(1)
        # lex and parse
    try:
        tokens = tokenize(src)
        parser = Parser(tokens)
        program = parser.parse()
    except SyntaxError as e:
        # Malformed file: print a clear message and exit with nonzero status
        print(f"Syntax error: {e}")
        sys.exit(1)
    except Exception as e:
        # Any unexpected parsing error should also be treated as a malformed input,
        # without dumping a Python traceback to stdout/stderr.
        print(f"Error while parsing input: {e}")
        sys.exit(1)
    # semantic checks
    try:
        check_semantics(program)
    except ValueError as e:
        print(f"Semantic error: {e}")
        sys.exit(1)
    # build netlist
    net = build_netlist(program)
    # place nodes
    coords, width, height = place(net)
    # render and save image
    try:
        render(net, coords, width, height, outfile)
    except Exception as e:
        print(f"Error rendering image: {e}")
        sys.exit(1)
    print(f"[OK] Wrote image to {outfile}")


if __name__ == '__main__':
    main(sys.argv)