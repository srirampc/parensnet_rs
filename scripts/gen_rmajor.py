import argparse

import h5py
import numpy as np
import anndata as ad


def main(args):
    in_file: str = args.in_file
    out_file: str = args.out_file
    adx = ad.read_h5ad(in_file)
    tadx = adx.X.T
    with h5py.File(out_file, "w") as fwx:
        fwx.create_dataset('X', data=tadx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input ")
    parser.add_argument(
        "-o",
        "--out_file",
        type=str,
        required=True,
        help="Output File: Path to output file",
    )
    parser.add_argument(
        "-i",
        "--in_file",
        type=str,
        required=True,
        help="Path to Input File",
    )
    args = parser.parse_args()
    main(args)
