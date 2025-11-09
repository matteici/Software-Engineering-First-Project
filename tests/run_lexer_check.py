# check_lexer.py
# Run with: python run_lexer_check.py
from lexer import tokenize, LexError

def case(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
    except LexError as e:
        print(f"[FAIL] {name}: unexpected LexError: {e}")
    except Exception as e:
        print(f"[FAIL] {name}: unexpected error: {e}")

def expect_tokens(src, want):
    got = [(t.kind, t.value) for t in tokenize(src)]
    assert got == want, f"\n got={got}\nwant={want}"

def expect_lexerror(src, must_contain=None):
    try:
        list(tokenize(src))
        raise AssertionError("expected LexError, got success")
    except LexError as e:
        if must_contain:
            assert must_contain in str(e), f"error missing '{must_contain}': {e}"

def main():
    case("basic sequence", lambda: expect_tokens(
        "x = in[0] or in[1]; # hi",
        [("IDENT","x"),("EQUAL",None),("IN",None),("LBRACK",None),("NUMBER","0"),("RBRACK",None),
         ("OR",None),("IN",None),("LBRACK",None),("NUMBER","1"),("RBRACK",None),("SEMI",None),("EOF",None)]
    ))
    case("keywords vs identifiers", lambda: expect_tokens(
        "not1 = not not_var;",
        [("IDENT","not1"),("EQUAL",None),("NOT",None),("IDENT","not_var"),("SEMI",None),("EOF",None)]
    ))
    case("comments", lambda: expect_tokens(
        "# c\n a=in[0]; # t\n",
        [("IDENT","a"),("EQUAL",None),("IN",None),("LBRACK",None),("NUMBER","0"),("RBRACK",None),("SEMI",None),("EOF",None)]
    ))
    case("negative index rejected", lambda: expect_lexerror("in[-1]", "-"))
    case("weird char rejected",   lambda: expect_lexerror("a$=0;", "$"))

if __name__ == "__main__":
    main()
