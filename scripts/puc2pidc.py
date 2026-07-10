import argparse

import h5py
import numpy as np
import numpy.typing as npt
import scipy


def load_puc(in_file: str, ngenes: int) -> npt.NDArray:
    puc_scores = np.zeros((ngenes, ngenes))
    with h5py.File(in_file) as hfx:
        puc_index: npt.NDArray = hfx["/data/index"][:]  # pyright: ignore[reportIndexIssue, reportAssignmentType]
        puc_values: npt.NDArray = hfx["/data/puc"][:]  # pyright: ignore[reportIndexIssue, reportAssignmentType]
        for (i, j), value in zip(puc_index, puc_values):
            puc_scores[i, j] = value
            puc_scores[j, i] = value
    return puc_scores


def puc2pidc(puc_scores: npt.NDArray) -> npt.NDArray:
    pidc_scores = np.zeros(puc_scores.shape)


def main(args):
    ngenes: int = args.ngenes
    in_file: str = args.in_file
    out_file: str = args.out_file
    puc_matrix: npt.NDArray = load_puc(in_file, ngenes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input ")
    parser.add_argument(
        "-n",
        "--ngenes",
        type=int,
        required=True,
        help="Number of edges",
    )
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
