# netlist.py
"""
Netlist Build (DAG)
-------------------
Converts a *semantically valid* AST into a directed acyclic graph (DAG) of logic nodes.

Node types: IN, NOT, AND, OR, OUT
Edges: (src_node_id -> dst_node_id)
Aliases: identifiers do not create nodes; they point to an existing node id.
Forward refs: handled via per-kind registries for IN/OUT plus assignment order for identifiers.

Expected AST (minimal interface):
- Program(assignments: list[Assignment])
- Assignment(target: Target, expr: Expr)
- Targets:
    - Ident(name: str)
    - OutRef(index: int)     # out[k]
- Expr:
    - InRef(index: int)      # in[j]
    - IdentRef(name: str)    # previously-defined identifier target
    - Not(expr)
    - And(exprs: list[Expr]) # len >= 2
    - Or(exprs: list[Expr])  # len >= 2

If your parser uses different class names/fields, adapt the tiny adapter at the bottom or
wrap your AST to match this interface before calling build_netlist().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any

# ----------------------------
# Netlist data structures
# ----------------------------

@dataclass
class Node:
    nid: int
    op: str             # 'IN','NOT','AND','OR','OUT'
    index: Optional[int] = None  # for IN/OUT: the j or k index
    inputs: List[int] = field(default_factory=list)  # src node ids

@dataclass
class Netlist:
    nodes: Dict[int, Node]
    edges: List[Tuple[int, int]]
    inputs_used: Set[int]
    outputs_assigned: Set[int]
    name_to_nid: Dict[str, int]      # identifier aliases -> nid
    in_index_to_nid: Dict[int, int]  # IN j -> nid
    out_index_to_nid: Dict[int, int] # OUT k -> nid

    def topo_order(self) -> List[int]:
        """Return a topological order of node ids (Kahn)."""
        indeg = {nid: 0 for nid in self.nodes}
        for s, d in self.edges:
            indeg[d] += 1
        Q = [nid for nid, deg in indeg.items() if deg == 0]
        order = []
        i = 0
        while i < len(Q):
            u = Q[i]; i += 1
            order.append(u)
            for s, d in list(self._edges_from(u)):
                indeg[d] -= 1
                if indeg[d] == 0:
                    Q.append(d)
        if len(order) != len(self.nodes):
            raise RuntimeError("Cycle detected in netlist (should not happen for valid AST).")
        return order

    def _edges_from(self, u: int):
        for s, d in self.edges:
            if s == u:
                yield (s, d)

# ----------------------------
# Minimal AST protocol (duck-typed)
# ----------------------------

# We ship tiny AST helpers so the checker can build trees without your parser.
@dataclass
class Ident:       # target: x
    name: str

@dataclass
class OutRef:      # target: out[k]
    index: int

@dataclass
class InRef:       # expr atom: in[j]
    index: int

@dataclass
class IdentRef:    # expr atom: x
    name: str

@dataclass
class Not:
    expr: Any

@dataclass
class And:
    exprs: List[Any]

@dataclass
class Or:
    exprs: List[Any]

@dataclass
class Assignment:
    target: Any     # Ident | OutRef
    expr: Any       # expression

@dataclass
class Program:
    assignments: List[Assignment]


# ----------------------------
# Builder
# ----------------------------

class NetlistBuilder:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Tuple[int, int]] = []
        self.inputs_used: Set[int] = set()
        self.outputs_assigned: Set[int] = set()

        self._next_id = 1
        self.name_to_nid: Dict[str, int] = {}       # aliases
        self.in_index_to_nid: Dict[int, int] = {}   # in[j] -> nid
        self.out_index_to_nid: Dict[int, int] = {}  # out[k] -> nid

    # ---------- public API ----------
    def build_netlist(self, program: Program) -> Netlist:
        """Build a DAG from a semantically valid AST Program."""
        for asn in program.assignments:
            src = self._emit_expr(asn.expr)  # -> nid

            # Targets do not create WIRE nodes.
            if isinstance(asn.target, Ident):
                self.name_to_nid[asn.target.name] = src
            elif isinstance(asn.target, OutRef):
                out_nid = self._get_or_create_out(asn.target.index)
                self._connect(src, out_nid)
                self.outputs_assigned.add(asn.target.index)
            else:
                raise TypeError(f"Unsupported target type: {type(asn.target)}")

        return Netlist(
            nodes=self.nodes,
            edges=self.edges,
            inputs_used=self.inputs_used,
            outputs_assigned=self.outputs_assigned,
            name_to_nid=self.name_to_nid,
            in_index_to_nid=self.in_index_to_nid,
            out_index_to_nid=self.out_index_to_nid,
        )

    # ---------- expression emission ----------
        # ---------- expression emission ----------
    def _emit_expr(self, e: Any) -> int:
        if isinstance(e, InRef):
            self.inputs_used.add(e.index)
            return self._get_or_create_in(e.index)

        if isinstance(e, IdentRef):
            try:
                return self.name_to_nid[e.name]
            except KeyError as ex:
                raise KeyError(
                    f"Identifier '{e.name}' not defined before use."
                ) from ex

        if isinstance(e, OutRef):
            try:
                # RHS out[k] must refer to an already-assigned OUT node.
                return self.out_index_to_nid[e.index]
            except KeyError as ex:
                raise KeyError(
                    f"Output 'out[{e.index}]' not defined before use."
                ) from ex

        if isinstance(e, Not):
            c = self._emit_expr(e.expr)
            n = self._add_node("NOT")
            self._connect(c, n)
            return n

        if isinstance(e, And):
            if len(e.exprs) < 2:
                raise ValueError("AND must have 2+ operands.")
            inputs = [self._emit_expr(x) for x in e.exprs]
            n = self._add_node("AND")
            for src in inputs:
                self._connect(src, n)
            return n

        if isinstance(e, Or):
            if len(e.exprs) < 2:
                raise ValueError("OR must have 2+ operands.")
            inputs = [self._emit_expr(x) for x in e.exprs]
            n = self._add_node("OR")
            for src in inputs:
                self._connect(src, n)
            return n

        # Allow parenthesized expressions to be passed through unchanged by parsers.
        # If a parser wraps parentheses in a dedicated class, adapt here.
        raise TypeError(f"Unsupported expression type: {type(e)}")


        # Allow parenthesized expressions to be passed through unchanged by parsers.
        # If a parser wraps parentheses in a dedicated class, adapt here.
        raise TypeError(f"Unsupported expression type: {type(e)}")

    # ---------- node helpers ----------
    def _add_node(self, op: str, index: Optional[int] = None) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(nid=nid, op=op, index=index, inputs=[])
        return nid

    def _connect(self, src: int, dst: int) -> None:
        self.nodes[dst].inputs.append(src)
        self.edges.append((src, dst))

    def _get_or_create_in(self, j: int) -> int:
        if j in self.in_index_to_nid:
            return self.in_index_to_nid[j]
        nid = self._add_node('IN', index=j)
        self.in_index_to_nid[j] = nid
        return nid

    def _get_or_create_out(self, k: int) -> int:
        if k in self.out_index_to_nid:
            return self.out_index_to_nid[k]
        nid = self._add_node('OUT', index=k)
        self.out_index_to_nid[k] = nid
        return nid


# Convenience function
def build_netlist(program: Program) -> Netlist:
    return NetlistBuilder().build_netlist(program)


# If run directly, do a tiny demo (XOR build using the minimal AST classes).
if __name__ == "__main__":
    # either = in[0] or in[1];
    # both   = in[0] and in[1];
    # out[0] = either and (not both);
    prog = Program(assignments=[
        Assignment(Ident("either"), Or([InRef(0), InRef(1)])),
        Assignment(Ident("both"), And([InRef(0), InRef(1)])),
        Assignment(OutRef(0), And([IdentRef("either"), Not(IdentRef("both"))])),
    ])
    net = build_netlist(prog)
    print(f"nodes: {len(net.nodes)}, edges: {len(net.edges)}")
    for nid in sorted(net.nodes):
        nd = net.nodes[nid]
        print(f"{nid}: {nd.op}({nd.index}) <- {nd.inputs}")
