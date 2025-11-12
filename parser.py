from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Union

Token = Tuple[str, Optional[str]]

@dataclass(frozen=True)
class Identifier: name: str
@dataclass(frozen=True)
class In: index: int
@dataclass(frozen=True)
class Out: index: int

Expr = Union["Not","And","Or",Identifier,In,Out]

@dataclass(frozen=True)
class Not: expr: Expr
@dataclass(frozen=True)
class And: parts: List[Expr]
@dataclass(frozen=True)
class Or: parts: List[Expr]

@dataclass(frozen=True)
class Target:
    ident: Optional[str]=None
    out_index: Optional[int]=None
    def is_ident(self)->bool: return self.ident is not None
    def is_out(self)->bool: return self.out_index is not None

@dataclass(frozen=True)
class Assignment: target: Target; expr: Expr
@dataclass(frozen=True)
class Program: assignments: List[Assignment]

class ParseError(Exception): pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens; self.i = 0
        self.assigned_idents: Set[str] = set()
        self.assigned_outs: Set[int] = set()

    def _peek(self)->Token:
        return self.toks[self.i] if self.i < len(self.toks) else ("EOF", None)
    def _advance(self)->Token:
        t = self._peek(); self.i += 1; return t
    def _match(self, kind:str)->Token:
        t = self._peek()
        if t[0]!=kind: self._error(f"expected {kind}, got {t[0]}")
        return self._advance()
    def _accept(self, kind:str)->bool:
        if self._peek()[0]==kind: self._advance(); return True
        return False
    def _error(self, msg:str)->None:
        t = self._peek(); raise ParseError(f"Parse error near token {t!r}: {msg}")

    def parse_program(self)->Program:
        assigns: List[Assignment] = []
        while True:
            if self._peek()[0]=="EOF":
                if not assigns: self._error("empty program (at least one assignment required)")
                break
            assigns.append(self.parse_assignment())
        return Program(assigns)

    def parse_assignment(self)->Assignment:
        target = self.parse_target()
        self._match("EQUAL")
        expr = self.parse_expr()
        self._match("SEMI")
        if target.is_ident():
            name = target.ident  # type: ignore
            if name in self.assigned_idents: self._error(f"identifier '{name}' is assigned more than once")
            self.assigned_idents.add(name)  # type: ignore
        else:
            k = target.out_index  # type: ignore
            if k in self.assigned_outs: self._error(f"out[{k}] is assigned more than once")
            self.assigned_outs.add(k)  # type: ignore
        return Assignment(target, expr)

    def parse_target(self)->Target:
        t = self._peek()
        if t[0]=="IDENT":
            name = self._advance()[1]; assert name is not None
            return Target(ident=name)
        if t[0]=="OUT":
            self._advance(); self._match("LBRACK")
            num = self._match("NUMBER")[1]; idx = self._parse_nonneg_int(num)
            self._match("RBRACK"); return Target(out_index=idx)
        self._error("target must be an identifier or out[<number>]")

    def parse_expr(self)->Expr:
        if self._peek()[0]=="NOT": return self.parse_negation()
        first = self.parse_paren_expr()
        if self._peek()[0]=="AND":
            parts=[first]
            while self._accept("AND"): parts.append(self.parse_paren_expr())
            if len(parts)<2: self._error("AND requires at least two operands")
            return And(parts)
        if self._peek()[0]=="OR":
            parts=[first]
            while self._accept("OR"): parts.append(self.parse_paren_expr())
            if len(parts)<2: self._error("OR requires at least two operands")
            return Or(parts)
        return first

    def parse_paren_expr(self)->Expr:
        if self._accept("LPAREN"):
            e = self.parse_expr()
            if not self._accept("RPAREN"): self._error("missing ')' to close '('")
            return e
        return self.parse_element()

    def parse_negation(self)->Not:
        self._match("NOT")
        return Not(self.parse_paren_expr())

    def parse_element(self)->Expr:
        t = self._peek()
        if t[0]=="IN":
            self._advance(); self._match("LBRACK")
            num = self._match("NUMBER")[1]; idx = self._parse_nonneg_int(num)
            self._match("RBRACK"); return In(idx)
        if t[0]=="IDENT":
            name = self._advance()[1]; assert name is not None
            if name not in self.assigned_idents: self._error(f"identifier '{name}' used before assignment")
            return Identifier(name)
        if t[0]=="OUT":
            self._advance(); self._match("LBRACK")
            num = self._match("NUMBER")[1]; idx = self._parse_nonneg_int(num)
            self._match("RBRACK")
            if idx not in self.assigned_outs: self._error(f"out[{idx}] used before assignment")
            return Out(idx)
        self._error("expected an element (identifier, out[k], or in[j]) or a parenthesized expression")

    @staticmethod
    def _parse_nonneg_int(lexeme: Optional[str])->int:
        if lexeme is None or not lexeme.isdigit(): raise ParseError(f"invalid number token: {lexeme!r}")
        return int(lexeme)

def parse(tokens: List[Token])->Program:
    p = Parser(tokens); prog = p.parse_program()
    if p._peek()[0]!="EOF": raise ParseError("extra tokens after the last assignment")
    return prog

def expr_to_str(e: Expr)->str:
    if isinstance(e, Identifier): return e.name
    if isinstance(e, In): return f"in[{e.index}]"
    if isinstance(e, Out): return f"out[{e.index}]"
    if isinstance(e, Not):
        sub = e.expr
        if isinstance(sub,(Identifier,In,Out)): return f"not {expr_to_str(sub)}"
        return f"not ({expr_to_str(sub)})"
    if isinstance(e, And): return " and ".join(_maybe_paren(p) for p in e.parts)
    if isinstance(e, Or): return " or ".join(_maybe_paren(p) for p in e.parts)
    raise TypeError(f"unknown expr {e}")

def _maybe_paren(e: Expr)->str:
    return expr_to_str(e) if isinstance(e,(Identifier,In,Out)) else f"({expr_to_str(e)})"

def assignment_to_str(a: Assignment)->str:
    left = a.target.ident if a.target.is_ident() else f"out[{a.target.out_index}]"
    return f"{left} = {expr_to_str(a.expr)};"

__all__ = ["Token","Identifier","In","Out","Not","And","Or","Target","Assignment","Program",
           "ParseError","parse","expr_to_str","assignment_to_str"]
