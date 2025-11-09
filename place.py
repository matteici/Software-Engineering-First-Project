#!/usr/bin/env python3
"""
place.py — Compile a tiny Boolean assignment language into a PPM pixel circuit.

USAGE
  python3 place.py input.formulas output.ppm

This is a self-contained reference implementation that:
  • Lexes and parses an infix Boolean DSL with identifiers, in[k], out[k].
  • Builds a DAG of signals (combinational only).
  • Assigns simple left→right columns (topological levels) and unique rows.
  • Routes nets with Manhattan wiring using BLUE pixels, branches via BLACK, and
    uses gate pixels (RED=NOT, YELLOW=AND, GREEN=OR) that drive to the RIGHT.
  • Stamps input/output border pixels (BLUE) on even rows as required.
  • Validates gate neighbor counts and basic border constraints.
  • Emits a P3 (ASCII) PPM.

NOTE: This aims for correctness first, not area optimality. It produces compact
      but conservative layouts and favors clarity/robustness over minimal pixels.
"""
from __future__ import annotations
import sys
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Set

# --------------------------- Colors (RGB) ---------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
YELLOW= (255, 255, 0)
GREEN = (0, 255, 0)

# Grid directions
DIRS = [(1,0),(0,1),(-1,0),(0,-1)]  # R, D, L, U

# --------------------------- Lexer ---------------------------
TokenType = str

@dataclass
class Token:
    typ: TokenType
    val: str
    line: int
    col: int

KEYWORDS = {"in", "out", "not", "and", "or"}

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.i = 0
        self.line = 1
        self.col = 1
        self.n = len(text)

    def _peek(self) -> str:
        return self.text[self.i] if self.i < self.n else ""

    def _adv(self) -> str:
        ch = self._peek()
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_ws_and_comments(self):
        while self.i < self.n:
            ch = self._peek()
            if ch in " \t\r\n":
                self._adv()
                continue
            if ch == '#':
                while self.i < self.n and self._peek() != '\n':
                    self._adv()
                continue
            break

    def tokens(self) -> List[Token]:
        toks: List[Token] = []
        while True:
            self._skip_ws_and_comments()
            if self.i >= self.n:
                break
            ch = self._peek()
            line, col = self.line, self.col
            if ch in "()[]=;":
                toks.append(Token(ch, ch, line, col))
                self._adv()
            elif ch.isdigit():
                start = self.i
                while self._peek().isdigit():
                    self._adv()
                toks.append(Token("NUMBER", self.text[start:self.i], line, col))
            elif ch.isalpha() or ch == '_':
                start = self.i
                while True:
                    c = self._peek()
                    if c.isalnum() or c == '_':
                        self._adv()
                    else:
                        break
                word = self.text[start:self.i]
                if word in KEYWORDS:
                    toks.append(Token(word.upper(), word, line, col))
                else:
                    toks.append(Token("IDENT", word, line, col))
            else:
                raise SyntaxError(f"Illegal character '{ch}' at {line}:{col}")
        toks.append(Token("EOF", "", self.line, self.col))
        return toks

# --------------------------- Parser / AST ---------------------------
@dataclass
class Element:
    # either IDENT name or ('in', idx)
    kind: str  # 'ident' or 'in'
    name: Optional[str] = None
    index: Optional[int] = None

@dataclass
class Expr:
    op: str  # 'elem','not','and','or'
    a: Optional['Expr'] = None
    b: Optional['Expr'] = None
    # For n-ary and/or we chain as left-assoc (a op b op c ...)
    elem: Optional[Element] = None

@dataclass
class Target:
    # either IDENT or out[idx]
    kind: str  # 'ident' or 'out'
    name: Optional[str] = None
    index: Optional[int] = None

@dataclass
class Assign:
    target: Target
    expr: Expr

class Parser:
    def __init__(self, toks: List[Token]):
        self.toks = toks
        self.k = 0

    def _cur(self) -> Token:
        return self.toks[self.k]

    def _eat(self, typ: str) -> Token:
        tok = self._cur()
        if tok.typ != typ:
            raise SyntaxError(f"Expected {typ} at {tok.line}:{tok.col}, got {tok.typ}")
        self.k += 1
        return tok

    def _match(self, typ: str) -> bool:
        if self._cur().typ == typ:
            self.k += 1
            return True
        return False

    def parse(self) -> List[Assign]:
        assigns: List[Assign] = []
        while self._cur().typ != 'EOF':
            assigns.append(self.parse_assignment())
        return assigns

    def parse_assignment(self) -> Assign:
        tgt = self.parse_target()
        self._eat('=')
        e = self.parse_expr()
        self._eat(';')
        return Assign(tgt, e)

    def parse_target(self) -> Target:
        tok = self._cur()
        if tok.typ == 'IDENT':
            self._eat('IDENT')
            return Target('ident', name=tok.val)
        if tok.typ == 'OUT':
            self._eat('OUT')
            self._eat('[')
            num = self._eat('NUMBER')
            idx = int(num.val)
            if idx < 0:
                raise SyntaxError(f"out index must be non-negative at {num.line}:{num.col}")
            self._eat(']')
            return Target('out', index=idx)
        raise SyntaxError(f"Invalid assignment target at {tok.line}:{tok.col}")

    # expr: or_expr
    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        e = self.parse_and()
        while self._match('OR'):
            rhs = self.parse_and()
            e = Expr('or', a=e, b=rhs)
        return e

    def parse_and(self) -> Expr:
        e = self.parse_unary()
        while self._match('AND'):
            rhs = self.parse_unary()
            e = Expr('and', a=e, b=rhs)
        return e

    def parse_unary(self) -> Expr:
        if self._match('NOT'):
            return Expr('not', a=self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self._cur()
        if tok.typ == 'IDENT':
            self._eat('IDENT')
            return Expr('elem', elem=Element('ident', name=tok.val))
        if tok.typ == 'IN':
            self._eat('IN')
            self._eat('[')
            num = self._eat('NUMBER')
            idx = int(num.val)
            if idx < 0:
                raise SyntaxError(f"in index must be non-negative at {num.line}:{num.col}")
            self._eat(']')
            return Expr('elem', elem=Element('in', index=idx))
        if tok.typ == '(':  # parenthesized
            self._eat('(')
            e = self.parse_expr()
            self._eat(')')
            return e
        raise SyntaxError(f"Unexpected token {tok.typ} at {tok.line}:{tok.col}")

# --------------------------- Semantic / Netlist ---------------------------
NodeId = int

@dataclass
class Node:
    kind: str  # 'IN','NOT','AND','OR','WIRE','OUT'
    name: Optional[str] = None
    index: Optional[int] = None
    inputs: List[NodeId] = None  # sources

class Netlist:
    def __init__(self):
        self.nodes: Dict[NodeId, Node] = {}
        self.next_id: int = 0
        # mapping from signal names to driving node id
        self.signals: Dict[str, NodeId] = {}
        self.inputs: Set[int] = set()
        self.outputs: Set[int] = set()
        self.out_to_signal: Dict[int, NodeId] = {}

    def add_node(self, kind: str, name: Optional[str]=None, index: Optional[int]=None, inputs: Optional[List[NodeId]]=None) -> NodeId:
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = Node(kind=kind, name=name, index=index, inputs=list(inputs or []))
        return nid

    def topo_order(self) -> List[NodeId]:
        # simple Kahn on implicit edges via inputs
        indeg: Dict[NodeId, int] = {i:0 for i in self.nodes}
        adj: Dict[NodeId, List[NodeId]] = {i:[] for i in self.nodes}
        for v, node in self.nodes.items():
            for u in (node.inputs or []):
                indeg[v] += 1
                adj[u].append(v)
        q = [i for i,d in indeg.items() if d==0]
        out: List[NodeId] = []
        while q:
            u = q.pop(0)
            out.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(out) != len(self.nodes):
            raise ValueError("Cycle detected (should be impossible with this language)")
        return out

# Build netlist from assignments

def build_netlist(assigns: List[Assign]) -> Netlist:
    nl = Netlist()
    defined: Set[str] = set()

    def compile_expr(e: Expr) -> NodeId:
        if e.op == 'elem':
            el = e.elem
            assert el is not None
            if el.kind == 'ident':
                if el.name not in defined:
                    raise ValueError(f"Use before assignment: {el.name}")
                return nl.signals[el.name]
            else:  # in[idx]
                idx = el.index
                if idx not in nl.inputs:
                    nid = nl.add_node('IN', index=idx)
                    nl.inputs.add(idx)
                else:
                    # find existing IN node
                    nid = next(i for i,n in nl.nodes.items() if n.kind=='IN' and n.index==idx)
                return nid
        elif e.op == 'not':
            a = compile_expr(e.a)
            return nl.add_node('NOT', inputs=[a])
        elif e.op == 'and':
            a = compile_expr(e.a)
            b = compile_expr(e.b)
            return nl.add_node('AND', inputs=[a,b])
        elif e.op == 'or':
            a = compile_expr(e.a)
            b = compile_expr(e.b)
            return nl.add_node('OR', inputs=[a,b])
        else:
            raise ValueError(f"Unknown expr op {e.op}")

    for a in assigns:
        # compile RHS
        rhs = compile_expr(a.expr)
        # bind target
        if a.target.kind == 'ident':
            name = a.target.name
            if name in defined:
                raise ValueError(f"Reassignment of {name}")
            defined.add(name)
            nl.signals[name] = rhs
        else:  # out[idx]
            idx = a.target.index
            if idx in nl.outputs:
                raise ValueError(f"Reassignment of out[{idx}]")
            nl.outputs.add(idx)
            outnode = nl.add_node('OUT', index=idx, inputs=[rhs])
            nl.out_to_signal[idx] = rhs
    return nl

# --------------------------- Placement & Routing ---------------------------
@dataclass
class Point:
    x: int
    y: int

class Grid:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.pix = [[WHITE for _ in range(width)] for _ in range(height)]

    def inside(self, x:int,y:int)->bool:
        return 0<=x<self.w and 0<=y<self.h

    def set(self, x:int,y:int,color:Tuple[int,int,int]):
        if not self.inside(x,y):
            raise ValueError("Pixel out of bounds")
        self.pix[y][x] = color

    def get(self, x:int,y:int):
        if not self.inside(x,y):
            return WHITE
        return self.pix[y][x]

    def write_ppm(self, path:str):
        with open(path, 'w') as f:
            f.write(f"P3\n{self.w} {self.h}\n255\n")
            for row in self.pix:
                f.write(" ".join(f"{r} {g} {b}" for (r,g,b) in row) + "\n")

# Very simple placer: columns by topo layer, rows unique

def place_and_route(nl: Netlist) -> Grid:
    order = nl.topo_order()
    # Compute topological level (distance from inputs)
    level: Dict[NodeId,int] = {}
    def lev(nid:NodeId)->int:
        if nid in level:
            return level[nid]
        node = nl.nodes[nid]
        if not node.inputs:
            level[nid] = 0
        else:
            level[nid] = 1 + max(lev(u) for u in node.inputs)
        return level[nid]
    for nid in order:
        lev(nid)

    # Map IN/OUT rows to even rows; internal nodes start below I/O rows
    used_inputs = sorted(nl.inputs)
    used_outputs = sorted(nl.outputs)
    max_io = max([0] + used_inputs + used_outputs) if (used_inputs or used_outputs) else 0
    base_internal_row = 2*max_io + 2

    row_for_in: Dict[int,int] = {idx: 2*idx for idx in used_inputs}
    row_for_out: Dict[int,int] = {idx: 2*idx for idx in used_outputs}

    # Node positions
    pos: Dict[NodeId, Point] = {}

    # Columns spaced by level; leave borders at x=0 and x=w-1
    max_level = max(level.values()) if level else 0
    w = 2 + 3*(max_level+1) + 2  # internal columns + margins + right border

    # Assign rows greedily (spaced by 2 for easy detours)
    next_free_row = base_internal_row
    for nid in order:
        n = nl.nodes[nid]
        if n.kind == 'IN':
            pos[nid] = Point(0, row_for_in[n.index])
        elif n.kind == 'OUT':
            pos[nid] = Point(w-1, row_for_out[n.index])
        else:
            pos[nid] = Point(2 + 3*level[nid], next_free_row)
            next_free_row += 2

    # Grid height
    h = max([p.y for p in pos.values()] + [0]) + 3
    g = Grid(w, h)

    # Stamp border pixels
    for idx in used_inputs:
        g.set(0, row_for_in[idx], BLUE)
    for idx in used_outputs:
        g.set(w-1, row_for_out[idx], BLUE)

    # --- Routing helpers (non-committing probes first) ---
    def is_routable_cell(x:int,y:int) -> bool:
        c = g.get(x,y)
        return c in (WHITE, BLUE, BLACK)

    def path_clear_hline(y:int, x1:int, x2:int) -> bool:
        if x1>x2: x1,x2=x2,x1
        for x in range(x1, x2+1):
            if not is_routable_cell(x,y):
                return False
        return True

    def path_clear_vline(x:int, y1:int, y2:int) -> bool:
        if y1>y2: y1,y2=y2,y1
        for y in range(y1, y2+1):
            if not is_routable_cell(x,y):
                return False
        return True

    def draw_hline(y:int, x1:int, x2:int):
        if x1>x2: x1,x2=x2,x1
        for x in range(x1, x2+1):
            c = g.get(x,y)
            if c == WHITE:
                g.set(x,y,BLUE)
            elif c in (BLUE, BLACK):
                pass
            else:
                raise RuntimeError("Collision with gate; routing failed (hline)")

    def draw_vline(x:int, y1:int, y2:int):
        if y1>y2: y1,y2=y2,y1
        for y in range(y1, y2+1):
            c = g.get(x,y)
            if c == WHITE:
                g.set(x,y,BLUE)
            elif c in (BLUE, BLACK):
                pass
            else:
                raise RuntimeError("Collision with gate; routing failed (vline)")

    def branch_at(x:int,y:int):
        c = g.get(x,y)
        if c in (WHITE, BLUE):
            g.set(x,y,BLACK)

    def route(sx:int, sy:int, tx:int, ty:int) -> Tuple[str,int,int]:
        """Route from (sx,sy) to (tx,ty) trying HV, then VH, then small detours.
        Returns (pattern, bx, by) where pattern in {'HV','VH'} and (bx,by) is the bend point."""
        # 1) Try HV
        if path_clear_hline(sy, sx, tx) and path_clear_vline(tx, sy, ty):
            draw_hline(sy, sx, tx)
            draw_vline(tx, sy, ty)
            return ('HV', tx, sy)
        # 2) Try VH
        if path_clear_vline(sx, sy, ty) and path_clear_hline(ty, sx, tx):
            draw_vline(sx, sy, ty)
            draw_hline(ty, sx, tx)
            return ('VH', sx, ty)
        # 3) Try vertical detours (up/down by a few rows)
        for d in range(1, 6):
            for midy in (sy+d, sy-d):
                if 0 <= midy < g.h:
                    if path_clear_vline(sx, sy, midy) and path_clear_hline(midy, sx, tx) and path_clear_vline(tx, midy, ty):
                        draw_vline(sx, sy, midy)
                        draw_hline(midy, sx, tx)
                        draw_vline(tx, midy, ty)
                        return ('VHV', tx, midy)
        # 4) Try horizontal detours (left/right by a few cols)
        for d in range(1, 6):
            for midx in (sx+d, sx-d):
                if 0 <= midx < g.w:
                    if path_clear_hline(sy, sx, midx) and path_clear_vline(midx, sy, ty) and path_clear_hline(ty, midx, tx):
                        draw_hline(sy, sx, midx)
                        draw_vline(midx, sy, ty)
                        draw_hline(ty, midx, tx)
                        return ('HVH', midx, ty)
        raise RuntimeError("Routing failed: no path without crossing gates")

    # Place gates and route their inputs
    for nid in order:
        node = nl.nodes[nid]
        p = pos[nid]
        if node.kind in ('NOT','AND','OR'):
            # Place gate pixel
            gate_color = RED if node.kind=='NOT' else (YELLOW if node.kind=='AND' else GREEN)
            g.set(p.x, p.y, gate_color)
            # Ensure right neighbor (gate output) exists
            if not is_routable_cell(p.x+1, p.y):
                raise RuntimeError("Right neighbor blocked at gate output")
            draw_hline(p.y, p.x, p.x+1)

            ins = node.inputs or []
            if node.kind == 'NOT' and len(ins)!=1:
                raise RuntimeError("NOT expects 1 input")
            if node.kind in ('AND','OR') and len(ins)!=2:
                raise RuntimeError("AND/OR expect 2 inputs")

            targets: List[Tuple[NodeId, Tuple[int,int]]] = []
            if node.kind == 'NOT':
                targets.append((ins[0], (p.x-1, p.y)))  # approach from left
            else:
                targets.append((ins[0], (p.x-1, p.y)))   # left
                targets.append((ins[1], (p.x, p.y-1)))   # top

            for src, (tx,ty) in targets:
                sp = pos[src]
                sx, sy = sp.x, sp.y
                if nl.nodes[src].kind in ('IN','NOT','AND','OR'):
                    sx += 1  # take output to the right of source
                pattern, bx, by = route(sx, sy, tx, ty)
                # make junctions at bend and at the destination neighbor cell
                branch_at(bx, by)
                branch_at(tx, ty)
            # Extend a bit to the right to ease downstream connections
            if p.x+2 < g.w:
                draw_hline(p.y, p.x+1, p.x+2)

    # Connect outputs to their driving nets
    for nid in order:
        node = nl.nodes[nid]
        if node.kind == 'OUT':
            p = pos[nid]
            src = node.inputs[0]
            sp = pos[src]
            sx, sy = sp.x, sp.y
            if nl.nodes[src].kind in ('IN','NOT','AND','OR'):
                sx += 1
            tx, ty = p.x-1, p.y
            pattern, bx, by = route(sx, sy, tx, ty)
            branch_at(bx, by)
            branch_at(tx, ty)
            draw_hline(ty, tx, p.x)

    # Ensure each input drives at least one step right
    for nid in order:
        node = nl.nodes[nid]
        if node.kind == 'IN':
            p = pos[nid]
            draw_hline(p.y, p.x, p.x+1)

    return g

# --------------------------- Validation ---------------------------

def validate(grid: Grid):
    w, h = grid.w, grid.h
    # Gate neighbor count rules and right neighbor presence
    for y in range(h):
        for x in range(w):
            c = grid.get(x,y)
            if c in (RED, YELLOW, GREEN):
                nbrs = []
                for dx,dy in DIRS:
                    nx, ny = x+dx, y+dy
                    if 0<=nx<w and 0<=ny<h and grid.get(nx,ny) != WHITE:
                        nbrs.append((nx,ny))
                if c == RED:
                    if len(nbrs) != 2:
                        raise ValueError(f"NOT gate at {x},{y} must have exactly 2 non-white neighbors")
                    if not (x+1 < w and grid.get(x+1,y) != WHITE):
                        raise ValueError(f"NOT gate at {x},{y} must have right neighbor")
                else:
                    if len(nbrs) != 3:
                        raise ValueError(f"AND/OR gate at {x},{y} must have exactly 3 non-white neighbors")
                    if not (x+1 < w and grid.get(x+1,y) != WHITE):
                        raise ValueError(f"Gate at {x},{y} must have right neighbor")
    # Border checks: only even rows used, others white
    # Left border inputs
    for y in range(h):
        c = grid.get(0,y)
        if c != WHITE:
            if y % 2 != 0:
                raise ValueError(f"Left border non-white at odd row {y}")
            if c != BLUE:
                raise ValueError(f"Left border used row {y} must be BLUE")
    # Right border outputs
    for y in range(h):
        c = grid.get(w-1,y)
        if c != WHITE:
            if y % 2 != 0:
                raise ValueError(f"Right border non-white at odd row {y}")
            if c != BLUE:
                raise ValueError(f"Right border used row {y} must be BLUE")

# --------------------------- Main ---------------------------

def main(argv: List[str]):
    if len(argv) != 3:
        print("Usage: python3 place.py <input.formulas> <output.ppm>")
        sys.exit(2)
    in_path, out_path = argv[1], argv[2]
    try:
        text = open(in_path, 'r').read()
    except Exception as e:
        print(f"Error reading input: {e}")
        sys.exit(2)

    try:
        toks = Lexer(text).tokens()
        assigns = Parser(toks).parse()
        nl = build_netlist(assigns)
        grid = place_and_route(nl)
        validate(grid)
        grid.write_ppm(out_path)
    except (SyntaxError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)
