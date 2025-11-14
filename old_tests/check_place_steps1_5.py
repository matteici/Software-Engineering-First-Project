#!/usr/bin/env python3
"""
check_place_steps1_5.py
End-to-end checks for steps 1–5 of the PLACE pipeline, using place.py.

It:
  - Sets PLACE_STOP_BEFORE_RENDER=1 so routing/layout is skipped.
  - Expects a valid PPM (P3 or P6) on success cases.
  - Confirms common syntax/semantic failures return nonzero.
"""

import os
import sys
import tempfile
import textwrap
import subprocess
from pathlib import Path

PY = sys.executable
ENV = dict(os.environ, PLACE_STOP_BEFORE_RENDER="1")

def run(cmd, cwd=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=ENV,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def write(path: Path, s: str):
    path.write_text(textwrap.dedent(s).lstrip(), encoding="utf-8")

def is_ppm_ok(ppm_path: Path) -> bool:
    if not ppm_path.exists():
        return False
    try:
        head = ppm_path.read_bytes()[:2].decode(errors="ignore")
    except Exception:
        return False
    return head in ("P3", "P6")

def PASS(name): print(f"[PASS] {name}")
def FAIL(name, msg): print(f"[FAIL] {name}:\n  {msg}")

def main():
    total = 0
    ok = 0
    place_py = Path("place.py")
    if not place_py.exists():
        print("ERROR: place.py not found in current directory.")
        sys.exit(2)

    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)

        # 1) Success XOR
        total += 1
        try:
            src = td / "xor.txt"
            out = td / "xor.ppm"
            write(src, """
                either = in[0] or in[1];
                both = in[0] and in[1];
                out[0] = either and (not both);
            """)
            rc, so, se = run([PY, str(place_py), str(src), str(out)])
            if rc == 0 and is_ppm_ok(out):
                PASS("success XOR -> rc=0 & PPM exists")
                ok += 1
            else:
                FAIL("success XOR", f"rc={rc}, stdout={so!r}, stderr={se!r}, ppm_exists={out.exists()}")
        except Exception as e:
            FAIL("success XOR (exception)", str(e))

        # 2) Missing input file
        total += 1
        out2 = td / "nope.ppm"
        rc, so, se = run([PY, str(place_py), str(td / "nope.txt"), str(out2)])
        if rc != 0:
            PASS("missing input file -> rc!=0")
            ok += 1
        else:
            FAIL("missing input file", f"expected rc!=0, got rc=0; stdout={so!r}")

        # 3) Syntax error: missing semicolon
        total += 1
        src3 = td / "syntax.txt"; out3 = td / "syntax.ppm"
        write(src3, "x = in[0] or in[1]\n")
        rc, so, se = run([PY, str(place_py), str(src3), str(out3)])
        if rc != 0:
            PASS("syntax error (missing ;) -> rc!=0")
            ok += 1
        else:
            FAIL("syntax error", f"expected rc!=0, got rc=0; stdout={so!r}")

        # 4) Semantic: use-before-def
        total += 1
        src4 = td / "use_before_def.txt"; out4 = td / "use_before_def.ppm"
        write(src4, """
            out[0] = a and in[0];
            a = in[0];
        """)
        rc, so, se = run([PY, str(place_py), str(src4), str(out4)])
        if rc != 0:
            PASS("semantic: use-before-def -> rc!=0")
            ok += 1
        else:
            FAIL("semantic: use-before-def", f"expected rc!=0, got rc=0; stdout={so!r}")

        # 5) Semantic: duplicate out assignment
        total += 1
        src5 = td / "dup_out.txt"; out5 = td / "dup_out.ppm"
        write(src5, """
            out[0] = in[0];
            out[0] = in[1];
        """)
        rc, so, se = run([PY, str(place_py), str(src5), str(out5)])
        if rc != 0:
            PASS("semantic: duplicate out assignment -> rc!=0")
            ok += 1
        else:
            FAIL("semantic: duplicate out assignment", f"expected rc!=0, got rc=0; stdout={so!r}")

        # 6) Valid chain with parentheses
        total += 1
        src6 = td / "chain.txt"; out6 = td / "chain.ppm"
        write(src6, """
            a = in[0];
            b = (a or in[1]);
            c = (not b);
            out[1] = a and (not c);
        """)
        rc, so, se = run([PY, str(place_py), str(src6), str(out6)])
        if rc == 0 and is_ppm_ok(out6):
            PASS("valid chain with parentheses -> rc=0 & PPM exists")
            ok += 1
        else:
            FAIL("valid chain with parentheses", f"rc={rc}, stdout={so!r}, stderr={se!r}")

    print(f"\n[SUMMARY] {ok} passed, {total - ok} failed (total {total})")
    sys.exit(0 if ok == total else 1)

if __name__ == "__main__":
    main()
