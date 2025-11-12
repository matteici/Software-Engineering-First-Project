# topological.py
# Step 6) Topological Order: produce a valid node ordering (sources -> sinks) and detect cycles.
# Supports both Kahn's algorithm (BFS-like) and DFS-based ordering.
# Designed to be standalone and easy to plug into the existing Netlist DAG builder.

from collections import deque, defaultdict
from typing import Dict, Iterable, List, Tuple, Callable, Hashable, Optional, Set

Node = Hashable
Edge = Tuple[Node, Node]
Adj = Dict[Node, List[Node]]

class CycleError(Exception):
    """Raised when a cycle is detected in the dependency graph."""
    def __init__(self, message: str, cycle_nodes: Optional[List[Node]] = None):
        super().__init__(message)
        self.cycle_nodes = cycle_nodes or []

def _normalize_graph(nodes: Iterable[Node], edges: Iterable[Edge]) -> Tuple[Set[Node], Adj, Adj]:
    """Return (all_nodes, out_adj, in_adj) from arbitrary nodes/edges lists."""
    all_nodes: Set[Node] = set(nodes)
    out_adj: Adj = defaultdict(list)
    in_adj: Adj = defaultdict(list)
    for u, v in edges:
        all_nodes.add(u)
        all_nodes.add(v)
        out_adj[u].append(v)
        in_adj[v].append(u)
        # Ensure keys for isolated nodes exist as well
        if v not in out_adj:
            out_adj[v] = out_adj[v]
        if u not in in_adj:
            in_adj[u] = in_adj[u]
    # Ensure isolated nodes appear in both maps
    for n in all_nodes:
        _ = out_adj[n]
        _ = in_adj[n]
    return all_nodes, out_adj, in_adj

def topo_kahn(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    key: Optional[Callable[[Node], object]] = None
) -> List[Node]:
    """
    Kahn's algorithm (stable & deterministic if you pass a `key`).
    - nodes: iterable of nodes
    - edges: iterable of (u, v) directed edges (u -> v)
    - key: optional sorting key function for ready-set (e.g., x-coordinate for left→right)
    Returns a list of nodes in topological order.
    Raises CycleError if a cycle is present.
    """
    all_nodes, out_adj, in_adj = _normalize_graph(nodes, edges)
    indeg = {n: len(in_adj[n]) for n in all_nodes}
    ready = [n for n in all_nodes if indeg[n] == 0]
    if key is not None:
        ready.sort(key=key)
    else:
        # Deterministic default: sort by string representation
        ready.sort(key=lambda x: str(x))

    q = deque(ready)
    order: List[Node] = []
    while q:
        # We pop left, but before each pop we ensure the deque is sorted by key to keep stability
        if len(q) > 1 and key is not None:
            # keep stable order even as new nodes are appended
            tmp = sorted(list(q), key=key)
            q.clear()
            q.extend(tmp)
        elif len(q) > 1 and key is None:
            tmp = sorted(list(q), key=lambda x: str(x))
            q.clear()
            q.extend(tmp)

        u = q.popleft()
        order.append(u)
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(all_nodes):
        # Find a cycle: nodes with indeg>0 are in cycles
        cyclic = [n for n in all_nodes if indeg[n] > 0]
        raise CycleError("Cycle detected in graph (Kahn).", cyclic)

    return order

def topo_dfs(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    key: Optional[Callable[[Node], object]] = None
) -> List[Node]:
    """
    DFS-based topological sort with back-edge cycle detection.
    If `key` is provided, children are visited in sorted(key(child)) order (deterministic).
    Returns topo order (sources→sinks).
    Raises CycleError on cycles.
    """
    all_nodes, out_adj, _ = _normalize_graph(nodes, edges)
    color: Dict[Node, int] = {n: 0 for n in all_nodes}  # 0=white,1=gray,2=black
    stack: List[Node] = []
    order_rev: List[Node] = []
    cycle_trace: List[Node] = []

    def dfs(u: Node):
        nonlocal cycle_trace
        color[u] = 1
        stack.append(u)

        children = out_adj[u]
        if key is not None:
            children = sorted(children, key=key)
        else:
            children = sorted(children, key=lambda x: str(x))

        for v in children:
            if color[v] == 0:
                dfs(v)
            elif color[v] == 1:
                # Found a back-edge u->v, extract a cycle from stack
                try:
                    i = stack.index(v)
                    cycle_trace = stack[i:] + [v]
                except ValueError:
                    cycle_trace = [v, u]
                raise CycleError("Cycle detected in graph (DFS).", cycle_trace)

        stack.pop()
        color[u] = 2
        order_rev.append(u)

    # Visit all components
    roots = list(all_nodes)
    if key is not None:
        roots.sort(key=key)
    else:
        roots.sort(key=lambda x: str(x))
    for r in roots:
        if color[r] == 0:
            dfs(r)

    order_rev.reverse()
    return order_rev

def verify_topological_order(order: List[Node], edges: Iterable[Edge]) -> bool:
    """Utility: verify that `order` respects all dependencies in `edges`."""
    pos = {n: i for i, n in enumerate(order)}
    for u, v in edges:
        if pos[u] >= pos[v]:
            return False
    return True

# Convenience API for this project:
def topo_order(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    method: str = "kahn",
    key: Optional[Callable[[Node], object]] = None
) -> List[Node]:
    """
    Unified entry point.
    method: "kahn" (default) or "dfs"
    key: optional sorting key to encourage left→right (e.g., key=lambda n: x_pos[n])
    """
    if method == "kahn":
        return topo_kahn(nodes, edges, key=key)
    elif method == "dfs":
        return topo_dfs(nodes, edges, key=key)
    else:
        raise ValueError("method must be 'kahn' or 'dfs'")
