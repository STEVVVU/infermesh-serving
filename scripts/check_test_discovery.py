#!/usr/bin/env python
"""Fail if a unittest directory discovers fewer tests than expected.

`python -m unittest discover` exits 0 when it finds *zero* tests, so a broken
path, a renamed file, or a discovery pattern that stops matching turns a CI
test step into a silent no-op that still shows a green check. This guard makes
that failure mode loud: CI states how many tests it expects, and a shortfall is
an error.

Usage:
    python scripts/check_test_discovery.py <start-dir> <minimum-tests>
"""

from __future__ import annotations

import sys
import unittest


def count_tests(suite: unittest.TestSuite | unittest.TestCase) -> int:
    """Counts leaf test cases, descending through nested suites."""
    if isinstance(suite, unittest.TestCase):
        return 1
    return sum(count_tests(child) for child in suite)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <start-dir> <minimum-tests>", file=sys.stderr)
        return 2
    start_dir, minimum_raw = argv[1], argv[2]
    try:
        minimum = int(minimum_raw)
    except ValueError:
        print(f"minimum-tests must be an integer, got {minimum_raw!r}", file=sys.stderr)
        return 2

    suite = unittest.defaultTestLoader.discover(start_dir)

    # A discovery *error* (an unimportable test module) is reported by the
    # loader as a synthetic failing test, so it would satisfy a plain count.
    # Surface it directly instead.
    if unittest.defaultTestLoader.errors:
        for error in unittest.defaultTestLoader.errors:
            print(error, file=sys.stderr)
        print(f"FAIL: test discovery in {start_dir} reported errors", file=sys.stderr)
        return 1

    discovered = count_tests(suite)
    if discovered < minimum:
        print(
            f"FAIL: discovered {discovered} test(s) in {start_dir}, "
            f"expected at least {minimum}. A test runner that finds nothing "
            f"passes vacuously; check the path and discovery pattern.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: discovered {discovered} test(s) in {start_dir} (minimum {minimum})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
