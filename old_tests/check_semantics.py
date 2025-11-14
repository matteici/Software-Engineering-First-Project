#!/usr/bin/env python3
# check_semantics.py
# Lightweight test runner for semantic.py

import sys
from semantic import check_semantics, SemanticError

PASS = 0
FAIL = 0

def expect_ok(name, assignments):
    global PASS, FAIL
    try:
        check_semantics(assignments)
        print(f"[PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"[FAIL] {name}: expected OK, got {type(e).__name__}: {e}")
        FAIL += 1

def expect_error(name, assignments, contains: str):
    global PASS, FAIL
    try:
        check_semantics(assignments)
        print(f"[FAIL] {name}: expected error containing '{contains}', got OK")
        FAIL += 1
    except SemanticError as e:
        msg = str(e)
        if contains in msg:
            print(f"[PASS] {name}")
            PASS += 1
        else:
            print(f"[FAIL] {name}: wrong error.\n  wanted ~ '{contains}'\n  got       '{msg}'")
            FAIL += 1
    except Exception as e:
        print(f"[FAIL] {name}: unexpected exception type {type(e).__name__}: {e}")
        FAIL += 1


def main():
    # Helpers for brevity
    ID  = lambda n: ('id', n)
    OUT = lambda k: ('out', k)

    IN      = lambda j: ('in', j)
    RID     = lambda n: ('ref_id', n)
    ROUT    = lambda k: ('ref_out', k)
    NOT     = lambda e: ('not', e)
    AND     = lambda *xs: ('and', list(xs))
    OR      = lambda *xs: ('or', list(xs))
    PAREN   = lambda e: ('paren', e)

    # ----------------------------
    # Valid programs
    # ----------------------------
    expect_ok("basic valid chain",
              [
                  (ID('a'), OR(IN(0), IN(1))),
                  (OUT(0), AND(RID('a'), NOT(IN(1)))),
                  (ID('b'), OR(ROUT(0), IN(2))),  # uses out[0] after defined
              ])

    expect_ok("out referenced after def multiple times",
              [
                  (OUT(2), OR(IN(0), IN(1))),
                  (ID('x'), AND(ROUT(2), IN(3))),
                  (ID('y'), OR(ROUT(2), RID('x'))),
              ])

    expect_ok("parentheses + nested",
              [
                  (ID('p'), PAREN(AND(IN(0), NOT(IN(1))))),
                  (OUT(1), OR(RID('p'), IN(2))),
              ])

    # ----------------------------
    # Invalid: used before defined
    # ----------------------------
    expect_error("id used before def",
                 [
                     (OUT(0), OR(RID('a'), IN(0))),  # 'a' not defined yet
                 ],
                 "used before definition")

    expect_error("out used before def",
                 [
                     (ID('x'), OR(ROUT(1), IN(0))),  # out[1] not defined yet
                 ],
                 "out[1] used before definition")

    # ----------------------------
    # Invalid: reassignment
    # ----------------------------
    expect_error("name reassigned",
                 [
                     (ID('a'), IN(0)),
                     (ID('a'), IN(1)),
                 ],
                 "reassigned")

    expect_error("duplicate out assignment",
                 [
                     (OUT(0), IN(0)),
                     (OUT(0), IN(1)),
                 ],
                 "out[0] reassigned")

    # ----------------------------
    # Invalid: self-use in own assignment
    # ----------------------------
    expect_error("self-use identifier",
                 [
                     (ID('a'), OR(RID('a'), IN(0))),
                 ],
                 "used in its own assignment")

    expect_error("self-use out",
                 [
                     (OUT(2), OR(ROUT(2), IN(0))),
                 ],
                 "used in its own assignment")

    # ----------------------------
    # More edge cases
    # ----------------------------
    expect_error("forward ref through chain (out)",
                 [
                     (ID('x'), OR(ROUT(5), IN(0))),  # ref to out[5] first
                     (OUT(5), IN(1)),
                 ],
                 "out[5] used before definition")

    expect_ok("longer dependency chain",
              [
                  (ID('a'), IN(0)),
                  (ID('b'), AND(RID('a'), NOT(IN(1)))),
                  (OUT(0), OR(RID('b'), IN(2))),
                  (ID('c'), AND(RID('b'), ROUT(0))),
                  (OUT(1), OR(RID('c'), IN(3))),
              ])

    # Summary and exit code
    total = PASS + FAIL
    print(f"\n[SUMMARY] {PASS} passed, {FAIL} failed (total {total})")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
