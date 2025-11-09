from dataclasses import dataclass
from typing import Iterator, Optional

KEYWORDS = {"in": "IN", "out": "OUT", "not": "NOT", "and": "AND", "or": "OR"}

@dataclass
class Token:
    kind: str
    value: Optional[str]
    line: int
    col: int

class LexError(Exception):
    pass

def tokenize(source: str) -> Iterator[Token]:
    i, n = 0, len(source)
    line, col = 1, 1

    def peek() -> str:
        return source[i] if i < n else ""

    def advance() -> str:
        nonlocal i, line, col
        ch = source[i]
        i += 1
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1
        return ch

    while i < n:
        ch = peek()

        # Comments
        if ch == "#":
            while i < n and advance() != "\n":
                pass
            continue

        # Whitespace
        if ch in " \t\r\n":
            advance()
            continue

        # Single-char specials
        if ch in "()[]=;":
            start_line, start_col = line, col
            t = advance()
            mapping = {
                "(": "LPAREN", ")": "RPAREN",
                "[": "LBRACK", "]": "RBRACK",
                "=": "EQUAL",  ";": "SEMI",
            }
            yield Token(mapping[t], None, start_line, start_col)
            continue

        # Number (no sign per spec)
        if ch.isdigit():
            start_line, start_col = line, col
            buf = []
            while i < n and peek().isdigit():
                buf.append(advance())
            yield Token("NUMBER", "".join(buf), start_line, start_col)
            continue

        # Word (identifier or keyword)
        if ch.isalpha() or ch == "_":
            start_line, start_col = line, col
            buf = []
            while i < n and (peek().isalnum() or peek() == "_"):
                buf.append(advance())
            word = "".join(buf)
            kind = KEYWORDS.get(word, "IDENT")
            yield Token(kind, word if kind == "IDENT" else None, start_line, start_col)
            continue

        # Anything else is invalid → catches '-' so negatives are rejected here
        raise LexError(f"Invalid character '{ch}' at line {line}, col {col}")

    yield Token("EOF", None, line, col)
