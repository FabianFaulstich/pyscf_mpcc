#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from pyscf.mpcc import laplace_quadrature


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create and validate MPCC Laplace quadrature cache files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-output",
        metavar="FILE",
        help="Parse stdout from a laplace-minimax driver/test run.",
    )
    source.add_argument(
        "--from-init-table",
        action="store_true",
        help="Use laplace-minimax data/init_para.txt pretabulated parameters.",
    )
    parser.add_argument(
        "--root",
        default="external/laplace-minimax",
        help="Path to the laplace-minimax checkout for --from-init-table.",
    )
    parser.add_argument("--nlap", type=int, help="Number of Laplace points.")
    parser.add_argument("--ymin", type=float, help="Minimum denominator.")
    parser.add_argument("--ymax", type=float, help="Maximum denominator.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file containing exponents, weights, interval, and metadata.",
    )
    parser.add_argument(
        "--ngrid",
        type=int,
        default=10000,
        help="Number of logarithmic grid points for validation.",
    )
    args = parser.parse_args(argv)

    if args.from_output:
        quad = laplace_quadrature.from_laplace_minimax_output(args.from_output)
    else:
        missing = [
            name for name in ("nlap", "ymin", "ymax") if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "--from-init-table requires " + ", ".join(f"--{m}" for m in missing)
            )
        quad = laplace_quadrature.from_init_table(
            args.root, args.ymin, args.ymax, args.nlap
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quad.save(output)
    stats = quad.validate(args.ngrid)

    print(f"wrote {output}")
    print(f"source: {quad.source}")
    print(f"nlap: {quad.nlap}")
    print(f"interval: [{quad.ymin:.16e}, {quad.ymax:.16e}]")
    print(f"table_errmax: {quad.errmax:.6e}")
    print(f"validation_max_abs: {stats['max_abs']:.6e}")
    print(f"validation_max_rel: {stats['max_rel']:.6e}")
    print(f"validation_rms_abs: {stats['rms_abs']:.6e}")
    print(f"validation_rms_rel: {stats['rms_rel']:.6e}")


if __name__ == "__main__":
    sys.exit(main())
