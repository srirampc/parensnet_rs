PBMC5K_CAT = ["0", "1", "2", "3", "6", "7", "8", "9", "11", "12", "16", "18"]

RSC12K_CAT = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "29",
    "30",
]


# Cluster Sizes
# [10904, 2893, 4509, 3777, 980, 3478, 3684, 2377, 2359, 3474, 1658, 2451,
#  6652, 3141, 3497, 1092, 4304, 2854, 3874, 3126, 644, 1861, 2609, 1567,
#  4587, 1040, 1055, 1747, 646, 1225, 1178, 997]


RSC20K_CAT = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "31",
    "32",
    "33",
]


CFG_FORMAT_5K = """
h5ad_file: "./data/pbmc_scrna/0020K/adata.20K.5K.C{c}.h5ad"
row_major_h5_file: "/data/pbmc_scrna/0020K/adata.20K.5K.C{c}.rmajor.h5"
misi_data_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.misi.h5"
hist_data_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.hist.h5"
puc_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.lmr_puc.h5"
pidc_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.lmr_pidc.h5"
#
nrounds: 0
nsamples: 0
nobs: 0
nvars: 0
mode:
  - "{stage}"
  #- "samples_ranges"
  #- "puc2pidc"
tbase: "2"
save_nodes: False
save_node_pairs: True
"""

COMPLETE_CFG_FORMAT_5K = """
h5ad_file: "./data/pbmc_scrna/0020K/adata.20K.5K.C{c}.h5ad"
row_major_h5_file: "/data/pbmc_scrna/0020K/adata.20K.5K.C{c}.rmajor.h5"
misi_data_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.misi.h5"
hist_data_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.hist.h5"
puc_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.lmr_puc.h5"
pidc_file: "./out/pbmc20k.5K/adata.20k.5K.C{c}.lmr_pidc.h5"
#
nrounds: 0
nsamples: 0
nobs: 0
nvars: 0
mode:
  - hist_nodes
  - hist2misi_dist
  - puc_lmr_dist
  #- "samples_ranges"
  #- "puc2pidc"
tbase: "2"
save_nodes: False
save_node_pairs: True
"""



CFG_FORMAT_20K = """h5ad_file: "./data/pbmc_scrna/rsc.20k/adata_lei.100k.20k.C{c}.h5ad"
row_major_h5_file: "./data/pbmc_scrna/rsc.20k/adata_lei.100k.20k.C{c}.rmajor.h5"
misi_data_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.hist_misidata.C{c}.h5"
hist_data_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.nodes.C{c}.h5"
puc_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.hist_lmr_puc.C{c}.h5"
pidc_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.hist_lmr_pidc.C{c}.h5"
#
nrounds: 0
nsamples: 0
nobs: 0
nvars: 0
mode:
  - {stage}
  #- "hist_dist"
  #- "samples_ranges"
  #- "puc2pidc"
tbase: "2"
save_nodes: False
save_node_pairs: True
"""

COMPLETE_CFG_FORMAT_20K = """h5ad_file: "./data/pbmc_scrna/rsc.20k/adata_lei.100k.20k.C{c}.h5ad"
row_major_h5_file: "./data/pbmc_scrna/rsc.20k/adata_lei.100k.20k.C{c}.rmajor.h5"
misi_data_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.hist_misidata.C{c}.h5"
hist_data_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.nodes.C{c}.h5"
puc_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.hist_lmr_puc.C{c}.h5"
pidc_file: "./out/pbmc100K.20K/cluster/adata.100K.20K.hist_lmr_pidc.C{c}.h5"
#
nrounds: 0
nsamples: 0
nobs: 0
nvars: 0
mode:
  - hist_nodes
  - hist2misi_dist
  - puc_lmr_dist
  #- "samples_ranges"
  #- "puc2pidc"
tbase: "2"
save_nodes: False
save_node_pairs: True
"""


SCRIPT_FORMAT_20K = """#!/bin/bash
#SBATCH -Jdcomplete_pbmc{c}_c100Kg20K_p1024n64         # Job name
#SBATCH -Agts-saluru8-coda20          # Tracking account
#SBATCH -N64                           # Number of nodes required
#SBATCH --ntasks-per-node 24          # Number of cores/node required
#SBATCH --mem=192G              # Memory per core
#SBATCH -t0:30:00                     # Duration of the job
#SBATCH -phive                        # Queue name
#SBATCH -oslurm/log/cluster_pucn/%x.log            # Combined output and error

# Change to working directory
cd "$SLURM_SUBMIT_DIR" || exit
SCRIPT="$HOME/dev/parensnet_rs/slurm/run_puc.sh"
CFG_LOC="$HOME/dev/parensnet_rs/config/cluster_pucn"

$SCRIPT -c "$CFG_LOC/dcomplete_pbmc_c100Kg20Kl{c}.yml" -p 1024 -n 16
"""


SCRIPT1280_FORMAT_20K = """#!/bin/bash
#SBATCH -Jdcomplete_pbmc{c}_c100Kg20K_p1280n64         # Job name
#SBATCH -Agts-saluru8-coda20          # Tracking account
#SBATCH -N64                           # Number of nodes required
#SBATCH --ntasks-per-node 24          # Number of cores/node required
#SBATCH --mem=192G              # Memory per core
#SBATCH -t0:30:00                     # Duration of the job
#SBATCH -phive                        # Queue name
#SBATCH -oslurm/log/cluster_pucn/%x.log            # Combined output and error

# Change to working directory
cd "$SLURM_SUBMIT_DIR" || exit
SCRIPT="$HOME/dev/parensnet_rs/slurm/run_puc.sh"
CFG_LOC="$HOME/dev/parensnet_rs/config/cluster_pucn"

$SCRIPT -c "$CFG_LOC/dcomplete_pbmc_c100Kg20Kl{c}.yml" -p 1280 -n 20
"""


def gen5k():
    STAGES_5K = ["hist_dist", "misi_dist", "puc_lmr_dist"]
    PREFIXES_5K = ["dhist", "dmisi", "dpuc"]
    for stg, prefix in zip(STAGES_5K, PREFIXES_5K):
        for cx in PBMC5K_CAT:
            fname = f"./config/cluster_pucn/{prefix}_pbmc_c20Kg5Kl{cx}.yml"
            cfg_contents = CFG_FORMAT_5K.format(c=cx, stage=stg)
            with open(fname, "w") as fx:
                fx.write(cfg_contents)


def gen20k_stages():
    STAGES_20K = ["hist_nodes", "hist2misi_dist", "puc_lmr_dist"]
    PREFIXES_20K = ["dnodes", "dnodes2misi", "dpuc"]
    for stg, prefix in zip(STAGES_20K, PREFIXES_20K):
        for cx in RSC20K_CAT:
            cfg_contents = CFG_FORMAT_20K.format(c=cx, stage=stg)
            fname = f"./config/cluster_pucn/{prefix}_pbmc_c100Kg20Kl{cx}.yml"
            with open(fname, "w") as fx:
                fx.write(cfg_contents)


def gen20k_complete():
    for cx in RSC20K_CAT:
        cfg_contents = COMPLETE_CFG_FORMAT_20K.format(c=cx)
        fname = f"./config/cluster_pucn/dcomplete_pbmc_c100Kg20Kl{cx}.yml"
        with open(fname, "w") as fx:
            fx.write(cfg_contents)


def gen20k_complete_scripts():
    for cx in RSC20K_CAT:
        script_contents = SCRIPT_FORMAT_20K.format(c=cx)
        fname = f"./slurm/scripts/cluster_pucn/dcomplete_pbmc{cx}_c100Kg20K_p1024n64.sh"
        with open(fname, "w") as fx:
            fx.write(script_contents)


def gen20k_complete1280_scripts():
    for cx in RSC20K_CAT:
        script_contents = SCRIPT1280_FORMAT_20K.format(c=cx)
        fname = f"./slurm/scripts/cluster_pucn/dcomplete_pbmc{cx}_c100Kg20K_p1280n64.sh"
        with open(fname, "w") as fx:
            fx.write(script_contents)


if __name__ == "__main__":
    # gen20k_complete()
    # gen20k_complete_scripts()
    gen20k_complete1280_scripts()
