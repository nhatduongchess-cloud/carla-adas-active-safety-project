"""Unit-test package.

This file is not optional. Without it `tests` is a namespace package, and
Python's import system only falls back to a namespace portion after scanning
the *whole* path for a regular package — so any `tests/__init__.py` that
happens to sit in the environment's site-packages wins over this directory,
and every `tests.test_*` import fails with a misleading error. One such
package is installed in the project's CARLA virtual environment.
"""
