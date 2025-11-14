#!/usr/bin/env python3
"""
Quick, dependency-free smoke tests for io_utils.read_and_validate.
Run from project root:
    python run_io_utils_check.py
Exits 0 on PASS, 1 on any failure.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

try:
    from io_utils import read_and_validate
except Exception as e:
    print(f"[FATAL] Could not import io_utils.read_and_validate: {e}", file=sys.stderr)
    sys.exit(1)

GREEN = "[PASS]"
RED = "[FAIL]"

def run_case(name, func):
    try:
        func()
        print(f"{GREEN} {name}")
        return True
    except AssertionError as e:
        print(f"{RED}  {name} -> {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"{RED}  {name} (unexpected error) -> {e}", file=sys.stderr)
        return False

def case_missing_input(tmp):
    """
    Expect return code 1 when input file is missing.
    """
    out = tmp / "out.ppm"
    rc = read_and_validate(f"nope.txt {out}")
    assert rc == 1, f"Expected rc=1, got {rc}"

def case_wrong_ext(tmp):
    """
    Expect return code 1 when output doesn't end with .ppm.
    """
    inp = tmp / "in.txt"
    inp.write_text("hello\n", encoding="utf-8")
    out = tmp / "out.png"
    rc = read_and_validate(f"{inp} {out}")
    assert rc == 1, f"Expected rc=1, got {rc}"

def case_unwritable_dir(tmp):
    """
    Expect return code 1 when output directory is not writable.
    """
    inp = tmp / "in.txt"
    inp.write_text("hello\n", encoding="utf-8")

    locked = tmp / "locked"
    locked.mkdir()
    # Remove write bit; still allow read/execute so path is visible
    locked.chmod(0o555)

    try:
        out = locked / "out.ppm"
        rc = read_and_validate(f"{inp} {out}")
        assert rc == 1, f"Expected rc=1, got {rc}"
    finally:
        # Restore so tmp cleanup works on all OSes
        locked.chmod(0o755)

def case_success(tmp):
    """
    Happy path: should write a minimal valid PPM and return 0.
    """
    inp = tmp / "in.txt"
    out = tmp / "out.ppm"
    inp.write_text("x = in[0];\n", encoding="utf-8")

    rc = read_and_validate(f"{inp} {out}")
    assert rc == 0, f"Expected rc=0, got {rc}"
    assert out.exists(), "Output PPM not created"

    text = out.read_text(encoding="ascii")
    assert text.startswith("P3\n"), "PPM header missing"
    assert "1 1" in text and "255" in text, "PPM size/maxval incorrect"
    last = [ln for ln in text.splitlines() if ln.strip()][-1].split()
    assert last == ["255", "255", "255"], "Pixel not white (255 255 255)"

def main():
    # Work in an isolated temp directory
    tmp_root = Path(tempfile.mkdtemp(prefix="io_utils_check_"))
    try:
        results = []
        results.append(run_case("missing input file -> rc=1", lambda: case_missing_input(tmp_root)))
        results.append(run_case("wrong output extension -> rc=1", lambda: case_wrong_ext(tmp_root)))
        results.append(run_case("unwritable output directory -> rc=1", lambda: case_unwritable_dir(tmp_root)))
        results.append(run_case("success path writes minimal PPM -> rc=0", lambda: case_success(tmp_root)))

        ok = all(results)
        if not ok:
            print("\nSome checks failed. See details above.", file=sys.stderr)
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
