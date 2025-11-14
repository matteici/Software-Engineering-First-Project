#!/usr/bin/env python3
"""
run_tests.py

This helper script walks through all `.txt` files in a specified directory (or the current
working directory if none is provided), invokes the simplified `place.py` program on
each file, and reports which tests fail to run successfully.

Usage:
  python3 run_tests.py [directory]

If a directory is provided, the script scans that directory for files ending in
`.txt`. Otherwise, it scans the current working directory. For each test file
found, it constructs an output path by replacing the `.txt` extension with
`.ppm`, invokes `place.py` with the test file and the output path, and records
the result. After all tests have been processed, it prints a summary of
which tests succeeded and which failed.

Tests are considered **failed** if `place.py` exits with a non‑zero status code
for that input. The script does not examine or keep the generated images.
"""

from __future__ import annotations

import os
import subprocess
import sys


def run_test(test_path: str, place_path: str, output_dir: str) -> int:
    """Run a single test file through `place.py` and return the exit code.

    Args:
        test_path: The path to the `.txt` test file.
        place_path: The path to the `place.py` script.
        output_dir: Directory where output images should be written.

    Returns:
        The exit code from the `place.py` invocation. A zero return code
        indicates success.
    """
    base_name = os.path.basename(test_path)
    name_without_ext = os.path.splitext(base_name)[0]
    out_file = os.path.join(output_dir, f"{name_without_ext}.ppm")
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct command: python3 place.py <input> <output>
    cmd = [sys.executable, place_path, test_path, out_file]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:
        # If an exception occurs when spawning, treat as failure (exit code 1)
        print(f"Error running {base_name}: {exc}")
        return 1
    return proc.returncode


def find_tests(directory: str) -> list[str]:
    """Return a sorted list of `.txt` files in the given directory."""
    tests = []
    for entry in os.listdir(directory):
        if entry.lower().endswith('.txt') and os.path.isfile(os.path.join(directory, entry)):
            tests.append(entry)
    tests.sort()
    return tests


def main(argv: list[str]) -> None:
    # Determine the directory to search for tests. If provided, use that; otherwise, use cwd.
    if len(argv) > 2:
        print(f"Usage: python3 {argv[0]} [tests_directory]")
        sys.exit(1)
    tests_dir = argv[1] if len(argv) == 2 else os.getcwd()
    tests_dir = os.path.abspath(tests_dir)
    if not os.path.isdir(tests_dir):
        print(f"Error: {tests_dir} is not a directory")
        sys.exit(1)

    # Locate the place.py script relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    place_path = os.path.join(script_dir, 'place.py')
    if not os.path.isfile(place_path):
        print(f"Error: could not find place.py at {place_path}")
        sys.exit(1)

    tests = find_tests(tests_dir)
    if not tests:
        print(f"No .txt test files found in {tests_dir}")
        sys.exit(0)

    failures = []
    successes = 0
    for test in tests:
        test_path = os.path.join(tests_dir, test)
        # Use a temporary output directory inside the test directory to collect images
        output_dir = os.path.join(tests_dir, 'test_outputs')
        ret = run_test(test_path, place_path, output_dir)
        status = "PASSED" if ret == 0 else f"FAILED (exit {ret})"
        print(f"Test {test}: {status}")
        if ret != 0:
            failures.append(test)
        else:
            successes += 1

    print()  # blank line for readability
    if failures:
        print(f"{len(failures)} out of {len(tests)} tests failed:")
        for fail in failures:
            print(f" - {fail}")
    else:
        print(f"All {len(tests)} tests passed successfully.")


if __name__ == '__main__':
    main(sys.argv)