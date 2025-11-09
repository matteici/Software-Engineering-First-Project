import os
import sys
import shlex

def _eprintln(msg: str) -> None:
    print(msg, file=sys.stderr)

def _write_minimal_ppm(path: str) -> None:
    """Write a valid 1×1 white PPM placeholder."""
    content = "P3\n1 1\n255\n255 255 255\n"
    with open(path, "w", encoding="ascii") as f:
        f.write(content)

def read_and_validate(text: str) -> int:
    """
    Parse CLI-like text ('a.txt a.ppm'), validate IO, and
    write a minimal placeholder PPM. Return 0 on success, non-zero on error.
    """
    try:
        args = shlex.split(text)
    except ValueError as e:
        _eprintln(f"Error: could not parse arguments: {e}")
        return 2

    if len(args) != 2:
        _eprintln("Usage: python3 place.py {formulas} {image}")
        _eprintln("Example: python3 place.py circuit.txt circuit.ppm")
        return 2

    formulas, image = args

    # --- Input validation
    if not os.path.isfile(formulas):
        _eprintln(f"Error: input file not found: {formulas}")
        return 1
    if not os.access(formulas, os.R_OK):
        _eprintln(f"Error: input file is not readable: {formulas}")
        return 1

    # --- Output validation
    if not image.lower().endswith(".ppm"):
        _eprintln(f"Error: output image must be a .ppm file: {image}")
        return 1
    out_dir = os.path.dirname(image) or "."
    if not os.path.isdir(out_dir):
        _eprintln(f"Error: output directory does not exist: {out_dir}")
        return 1
    if not os.access(out_dir, os.W_OK):
        _eprintln(f"Error: output directory is not writable: {out_dir}")
        return 1

    # --- IO check
    try:
        with open(formulas, "r", encoding="utf-8") as f:
            f.read(256)
    except Exception as e:
        _eprintln(f"Error: failed to read input file: {e}")
        return 1

    # --- Write placeholder
    try:
        _write_minimal_ppm(image)
    except Exception as e:
        _eprintln(f"Error: failed to write output PPM '{image}': {e}")
        return 1

    return 0
