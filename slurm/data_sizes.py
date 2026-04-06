import sys

import h5py
import numpy as np

TO_GB = 1024 * 1024 * 1024


def main(nodes_file):
    with h5py.File(nodes_file) as hfx:
        hdim = hfx["data/hist_dim"][:]
        jvdims = np.outer(hdim, hdim)
        print(f"DIM :: Size {hdim.shape}, JV {jvdims.shape}")
        print(f"DIM :: max {np.max(hdim)}, mean {np.mean(hdim)}, median {np.median(hdim)}")
        tri_jvdims = np.triu(jvdims, k=1)
        jv_size = np.sum(tri_jvdims)
        jv_mem = (jv_size * 4) / TO_GB
        print(f"JV :: Size {jv_size}, Memory {jv_mem}")
        lmr_size = np.sum(hdim) * len(hdim)
        lmr_mem = (lmr_size * 20) / TO_GB
        print(f"LMR :: Size {lmr_size}, Memory {lmr_mem}")


if __name__ == "__main__":
    main(sys.argv[1])
