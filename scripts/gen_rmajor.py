import argparse

import anndata as ad
import h5py


def main(args):
    in_file: str = args.in_file
    out_file: str = args.out_file
    adx = ad.read_h5ad(in_file)
    if adx.X is not None:
        tadx = adx.X.T  # pyright: ignore[reportAttributeAccessIssue]
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
