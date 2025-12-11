import argparse

def get_parser():
    parser = argparse.ArgumentParser(
            description = "Run MPCC"
            )
    parser.add_argument("molecule", type=str)
    parser.add_argument("basis", type=str)
    parser.add_argument("rank_reduced", type=str)
    parser.add_argument("--rank", type=float, default=None)
    parser.add_argument("--results", type = str, default = None)
    parser.add_argument("--scan",type=str,required=True, choices=["Lvv","Lov","Lov_Lvv","Lov_fix_Loo_1","Lvv_fix_Loo_1","Lov_Lvv_fix_Loo_1"],help="which tensor to scan:Lov etc")
    return parser

def parse_arg():
    parser = get_parser()
    args = parser.parse_args()

    rank_reduced = args.rank_reduced.lower() in ["true", "1", "yes"]

    if rank_reduced and args.rank is None:
        parser.error("When rank_reduced=True you must provide --rank")

    rank_value = args.rank if rank_reduced else None

    return args.molecule, args.basis, rank_reduced,rank_value, args.results, args.scan
