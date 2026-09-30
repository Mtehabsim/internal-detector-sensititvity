"""Command-line entry point for mtkaudit.gradsafe.official; see that module for the design.

    python scripts/detectors/gradsafe_driver.py --help
"""
from mtkaudit.gradsafe import official as _m

if __name__ == "__main__":
    _m.main()
