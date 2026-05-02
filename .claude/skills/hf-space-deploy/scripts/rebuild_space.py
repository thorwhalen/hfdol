#!/usr/bin/env python3
"""rebuild_space.py — factory-reboot an existing HF Space (no upload).

Use after publishing a new patch version of the package, when the Space's
Dockerfile pin (e.g. ~=0.1.1) will pick it up but the cached `pip install`
layer needs to be busted.

USAGE:
    python rebuild_space.py --space OWNER/NAME
    python rebuild_space.py --space OWNER/NAME --no-wait
"""

from __future__ import annotations

import argparse
import sys

from hfdol.deploy import factory_reboot, wait_for_build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--space", required=True, help='HF Space repo, e.g. "thorwhalen/typola".')
    ap.add_argument("--no-wait", action="store_true", help="Fire reboot and exit.")
    ap.add_argument("--timeout", type=int, default=900, help="Seconds to wait for build.")
    args = ap.parse_args()

    factory_reboot(args.space)

    if args.no_wait:
        print(f"Watch progress at: https://huggingface.co/spaces/{args.space}/logs")
        return

    print("Polling build status (every 15s)...")
    result = wait_for_build(args.space, timeout=args.timeout)
    if result.succeeded:
        space_url = f"https://{args.space.replace('/', '-')}.hf.space/"
        print(f"\nSpace is live: {space_url}")
        return
    if result.timed_out:
        print(f"\nTimeout after {result.elapsed_seconds}s (still {result.final_stage}).")
        sys.exit(2)
    print(f"\nBuild failed: stage={result.final_stage}")
    print(f"Logs: https://huggingface.co/spaces/{args.space}/logs")
    sys.exit(1)


if __name__ == "__main__":
    main()
