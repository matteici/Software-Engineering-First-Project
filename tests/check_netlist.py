# check_netlist.py
"""
Lightweight checker for netlist.py

Usage:
  python3 check_netlist.py
It prints a PASS/FAIL summary.
"""

from netlist import (
    Program, Assignment, Ident, OutRef, InRef, IdentRef,
    Not, And, Or, build_netlist
)

def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)

def test_xor_build():
    # either = in[0] or in[1];
    # both   = in[0] and in[1];
    # out[0] = either and (not both);
    prog = Program(assignments=[
        Assignment(Ident("either"), Or([InRef(0), InRef(1)])),
        Assignment(Ident("both"), And([InRef(0), InRef(1)])),
        Assignment(OutRef(0), And([IdentRef("either"), Not(IdentRef("both"))])),
    ])
    net = build_netlist(prog)

    # Basic shape checks
    assert_true(0 in net.inputs_used and 1 in net.inputs_used, "inputs 0 and 1 should be used")
    assert_true(0 in net.outputs_assigned, "out[0] must be assigned")
    # Find OUT[0]
    out0 = None
    for n in net.nodes.values():
        if n.op == 'OUT' and n.index == 0:
            out0 = n
            break
    assert_true(out0 is not None, "Missing OUT[0] node")
    assert_true(len(out0.inputs) == 1, "OUT should have exactly one input edge")

    # Trace that OUT input is an AND
    and_nid = out0.inputs[0]
    and_node = net.nodes[and_nid]
    assert_true(and_node.op == 'AND' and len(and_node.inputs) == 2, "Final gate must be AND with 2 inputs")

    # One input should be from OR(either), another from NOT(both)
    in1, in2 = and_node.inputs
    a = net.nodes[in1]
    b = net.nodes[in2]
    ops = {a.op, b.op}
    assert_true(ops == {'OR', 'NOT'}, "AND inputs should be OR(either) and NOT(both)")

    # Check that OR takes the two INs 0 and 1 (order irrelevant)
    or_node = a if a.op == 'OR' else b
    in_nids = {net.nodes[x].op for x in or_node.inputs}
    assert_true(in_nids == {'IN'}, "OR inputs must be IN nodes")

    # Check that NOT's input is the AND('both')
    not_node = a if a.op == 'NOT' else b
    assert_true(len(not_node.inputs) == 1, "NOT must have 1 input")
    both_and = net.nodes[not_node.inputs[0]]
    assert_true(both_and.op == 'AND', "NOT should be fed by AND(both)")

    print("[PASS] xor netlist build")

def test_identifier_alias_no_wire():
    # x = in[0] or in[1];
    # y = not x;
    # out[0] = y;
    prog = Program(assignments=[
        Assignment(Ident("x"), Or([InRef(0), InRef(1)])),
        Assignment(Ident("y"), Not(IdentRef("x"))),
        Assignment(OutRef(0), IdentRef("y")),
    ])
    net = build_netlist(prog)

    # OUT[0] should have input from a NOT node; no extra WIRE node for alias
    out0 = next(n for n in net.nodes.values() if n.op == 'OUT' and n.index == 0)
    assert_true(len(out0.inputs) == 1, "OUT should have 1 input")
    src = net.nodes[out0.inputs[0]]
    assert_true(src.op == 'NOT', "Alias should point directly to producing node (no WIRE)")
    print("[PASS] alias connects directly (no wire)")

def test_forward_reuse_structural():
    # a = in[0] and in[1];
    # out[0] = not a;
    # out[1] = a;
    prog = Program(assignments=[
        Assignment(Ident("a"), And([InRef(0), InRef(1)])),
        Assignment(OutRef(0), Not(IdentRef("a"))),
        Assignment(OutRef(1), IdentRef("a")),
    ])
    net = build_netlist(prog)
    out0 = next(n for n in net.nodes.values() if n.op == 'OUT' and n.index == 0)
    out1 = next(n for n in net.nodes.values() if n.op == 'OUT' and n.index == 1)
    # The same 'a' AND node should feed both branches (shared producer)
    a_from_out1 = net.nodes[out1.inputs[0]]
    not_from_out0 = net.nodes[out0.inputs[0]]
    a_from_not = net.nodes[not_from_out0.inputs[0]]
    assert_true(a_from_out1.nid == a_from_not.nid, "AND(a) node must be shared")
    print("[PASS] common subexpr reused via alias")

def run_all():
    tests = [
        test_xor_build,
        test_identifier_alias_no_wire,
        test_forward_reuse_structural,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
    total = len(tests)
    print(f"\n[SUMMARY] {passed} passed, {total - passed} failed (total {total})")

if __name__ == "__main__":
    run_all()
