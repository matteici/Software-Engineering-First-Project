# check_topological.py
# Self-checks for Step 6 topological order.
# Run: python3 check_topological.py

from topological import topo_order, verify_topological_order, CycleError

def run():
    passed = 0
    failed = 0

    def ok(name, cond=True):
        nonlocal passed, failed
        if cond:
            print(f"[PASS] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name}")
            failed += 1

    # 1) Simple chain A->B->C
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C")]
    order = topo_order(nodes, edges, method="kahn")
    ok("simple chain respects deps", verify_topological_order(order, edges))

    # 2) Diamond: A -> {B, C} -> D
    nodes = ["A", "B", "C", "D"]
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    order = topo_order(nodes, edges, method="kahn")
    ok("diamond respects deps", verify_topological_order(order, edges))

    # 3) Multiple sources: {A, B} -> C
    nodes = ["A", "B", "C"]
    edges = [("A", "C"), ("B", "C")]
    # Provide a key to stabilize order lexicographically (already default).
    order = topo_order(nodes, edges, method="kahn", key=lambda n: n)
    cond = verify_topological_order(order, edges) and set(order[:2]) == {"A", "B"}
    ok("multiple sources ordered first", cond)

    # 4) Disconnected: E alone, plus A->B
    nodes = ["A", "B", "E"]
    edges = [("A", "B")]
    order = topo_order(nodes, edges, method="kahn", key=lambda n: n)
    cond = verify_topological_order(order, edges) and set(order) == {"A", "B", "E"}
    ok("disconnected node included", cond)

    # 5) Self-loop cycle
    nodes = ["X"]
    edges = [("X", "X")]
    try:
        topo_order(nodes, edges, method="kahn")
        ok("self-loop cycle detected (kahn)", False)
    except CycleError:
        ok("self-loop cycle detected (kahn)", True)

    # 6) 3-cycle A->B->C->A
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    try:
        topo_order(nodes, edges, method="dfs")
        ok("3-cycle detected (dfs)", False)
    except CycleError:
        ok("3-cycle detected (dfs)", True)

    # 7) Custom left→right priority with x-positions
    nodes = ["IN0", "N1", "N2", "OUT0"]
    edges = [("IN0", "N1"), ("N1", "N2"), ("N2", "OUT0")]
    x_pos = {"IN0": 0, "N1": 1, "N2": 2, "OUT0": 3}
    order = topo_order(nodes, edges, key=lambda n: (x_pos.get(n, 1_000_000), str(n)))
    cond = verify_topological_order(order, edges) and order == ["IN0", "N1", "N2", "OUT0"]
    ok("key enforces left→right flavor", cond)

    print(f"\n[SUMMARY] {passed} passed, {failed} failed (total {passed+failed})")

if __name__ == "__main__":
    run()
