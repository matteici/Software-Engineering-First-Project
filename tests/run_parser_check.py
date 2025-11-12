#!/usr/bin/env python3
import sys, argparse, traceback
from typing import Iterable, List, Tuple, Optional, Any

# parser expects (kind, value) tuples
from parser import parse, ParseError, assignment_to_str, Program

# --- import lexer (may expose objects or tuples, and may yield a generator) ---
try:
    import lexer
    HAVE_LEXER = True
except Exception as e:
    HAVE_LEXER = False
    _lexer_import_err = e

TokenTuple = Tuple[str, Optional[str]]

def _materialize(x: Iterable[Any]) -> List[Any]:
    # Handles generators or lists
    return list(x)

def _get_attr_any(obj: Any, *names: str, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def _as_tuple(tok: Any) -> TokenTuple:
    """
    Convert a token object/tuple into (kind, value).
    Supported forms:
      - tuple/list: (kind, value) or (kind,) -> value None
      - object with attrs: .type/.kind/.typ and .value/.val/.lexeme/.text (best-effort)
    """
    # Already a 2-tuple?
    if isinstance(tok, (tuple, list)):
        if len(tok) == 2:
            k, v = tok
            return (str(k), v)
        if len(tok) == 1:
            return (str(tok[0]), None)
        # more than 2: take first two
        k, v = tok[0], tok[1]
        return (str(k), v)

    # Object style
    kind = _get_attr_any(tok, "type", "kind", "typ", "t")
    val  = _get_attr_any(tok, "value", "val", "lexeme", "text", "v")
    if kind is None:
        # As a last resort, try str(tok) and default value None
        return (str(tok), None)
    return (str(kind), val)

def _normalize_tokens(tokens_iterable: Iterable[Any]) -> List[TokenTuple]:
    raw = _materialize(tokens_iterable)
    norm = [_as_tuple(t) for t in raw]
    return norm

def _last_kind(tokens: List[TokenTuple]) -> Optional[str]:
    if not tokens:
        return None
    return tokens[-1][0]

def parse_text(text: str) -> Program:
    toks_raw = lexer.tokenize(text)            # may be generator OR list
    toks = _normalize_tokens(toks_raw)         # ensure list[(kind,value)]
    if _last_kind(toks) != "EOF":
        toks.append(("EOF", None))
    return parse(toks)

def parse_file(path: str) -> Program:
    with open(path, "r", encoding="utf-8") as f:
        return parse_text(f.read())

def run_self_tests(verbose: bool = False) -> int:
    ok = fail = 0

    def check_valid(code: str, label: str):
        nonlocal ok, fail
        try:
            prog = parse_text(code)
            first = assignment_to_str(prog.assignments[0]) if prog.assignments else "<no-assignments>"
            print(f"[PASS] {label}: {first}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            if verbose: traceback.print_exc()
            fail += 1

    def check_invalid(code: str, label: str):
        nonlocal ok, fail
        try:
            _ = parse_text(code)
            print(f"[FAIL] {label}: expected failure but parsed successfully")
            fail += 1
        except Exception:
            print(f"[PASS] {label}: correctly rejected")
            ok += 1

    check_valid("a = in[0] or in[1];", "valid OR with two operands")
    check_invalid("a = in[0] or in[1]", "missing semicolon")
    check_invalid("out[x] = in[0];", "bad out[] index (non-number)")
    check_invalid("a = in[0] and;", "AND with one operand")
    check_invalid("a = or in[0];", "OR with one operand")
    check_invalid("a = not in[0] or in[1];", "not with non-atomic (needs parentheses)")

    print(f"\nSelf-test summary: {ok} passed, {fail} failed.")
    return 0 if fail == 0 else 1

def main():
    if not HAVE_LEXER:
        print("Error: couldn't import your lexer module 'lexer'.", file=sys.stderr)
        print(f"Import error: {_lexer_import_err}", file=sys.stderr)
        sys.exit(2)

    ap = argparse.ArgumentParser(description="Parse a file with the placement-language parser.")
    ap.add_argument("source", nargs="?", help="Path to source file")
    ap.add_argument("--self-test", action="store_true", help="Run built-in parser checks")
    ap.add_argument("--verbose", action="store_true", help="Show tracebacks on failures")
    ap.add_argument("--tokens", action="store_true", help="Print first 10 normalized tokens")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(run_self_tests(verbose=args.verbose))

    if not args.source:
        print("No source provided. Use --self-test or pass a file path.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            src = f.read()
        toks = _normalize_tokens(lexer.tokenize(src))
        if _last_kind(toks) != "EOF":
            toks.append(("EOF", None))
        if args.tokens:
            print(f"[INFO] normalized {len(toks)} tokens; last kind = {_last_kind(toks)}")
            for t in toks[:10]:
                print("   ", t)
        prog = parse(toks)
        print(f"[OK] Parsed program with {len(prog.assignments)} assignment(s).")
        for a in prog.assignments:
            print("  " + assignment_to_str(a))
        sys.exit(0)
    except FileNotFoundError:
        print(f"Error: input file not found: {args.source}", file=sys.stderr)
        sys.exit(1)
    except ParseError as e:
        print(f"ParseError: {e}", file=sys.stderr)
        if args.verbose: traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose: traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
