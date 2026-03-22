import glob
import logging
import pathlib
import re

import polars as pl

log = logging.getLogger(__name__)


def read_timer_data(log_fname: str):
    with open(log_fname) as fptr:
        lines = [
            x.strip().split("]")[-1].strip() for x in fptr.readlines() if "TIMER" in x
        ]
        return [lx.split(",")[1:] for lx in lines]


def parse_file_name(log_fname: str):
    lpath = pathlib.Path(log_fname).name.replace(".log", "").split("_")
    run_name = lpath[0]
    data_name = lpath[1]
    data_sizes = re.split(r"[cg]", lpath[-2])
    run_procs = re.split(r"[pn]", lpath[-1])
    ncells = data_sizes[-2]
    ngenes = data_sizes[-1]
    nprocs = run_procs[-2]
    nnodes = run_procs[-1]
    return [run_name, data_name, ncells, ngenes, nprocs, nnodes]


def build_data_frame(
    meta_props,
    rows,
    col_names=["Max", "Min", "Avg", "Phase"],
    meta_names=["Run", "Dataset", "NCELLS", "NGENES", "NP", "N"],
):
    nrows = len(rows)
    ncols = len(rows[0])
    df_data = {}
    for ix, cname in zip(range(ncols), col_names):
        df_data[cname] = [rx[ix] for rx in rows]
    for mn, mpx in zip(meta_names, meta_props):
        df_data[mn] = [mpx for _ in range(nrows)]
    return pl.DataFrame(df_data)


def log_data_frame(lfx: str):
    data_rows = read_timer_data(lfx)
    if len(data_rows) == 0:
        log.log(logging.WARN, f"No rows in {lfx}")
        return None
    meta_props = parse_file_name(lfx)
    log.log(logging.DEBUG, f"META : {meta_props}")
    # log.log(logging.DEBUG, "\n".join(str(x) for x in data_rows))
    ldf = build_data_frame(meta_props, data_rows)
    log.log(logging.DEBUG, "%s", str(ldf))
    return ldf


def main(log_dir: str, pattern: str, out_csv: str):
    logging.basicConfig(level=logging.INFO)
    log_files = glob.glob(f"{log_dir}/{pattern}")
    nfiles = len(log_files)
    df_list = []
    for ix, lfx in enumerate(log_files):
        log.log(logging.INFO, f"Pocessing {ix}/{nfiles}: {lfx}")
        ldf = log_data_frame(lfx)
        if ldf is not None and not ldf.is_empty():
            df_list.append(ldf)
    full_df: pl.DataFrame = pl.concat(df_list)
    log.log(logging.INFO, "Final data framed: %s", str(full_df))
    full_df.write_csv(out_csv)


if __name__ == "__main__":
    # main("slurm/log/", "*.log", "slurm/log_times.csv")
    main("slurm/log/cluster_pucn/", "*.log", "slurm/cluster_pucn_log_times.csv")
