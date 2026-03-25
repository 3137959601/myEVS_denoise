"""Compatibility wrapper.

Core implementation lives in `myevs.io.openeb_csv_to_evtq`.
This script remains for historical/quick usage.
"""

from myevs.io.openeb_csv_to_evtq import main


if __name__ == "__main__":
    raise SystemExit(main())
