import argparse
import os
import pathlib
import shutil

HEADER = """#!/bin/bash
#SBATCH -J{name}                       # Job name
#SBATCH -A{account}                    # Tracking account
#SBATCH -N{nodes}                      # Number of nodes required
#SBATCH --ntasks-per-node {ntasks}     # Number of cores/node required
#SBATCH --mem={mem}                    # Memory per node
#SBATCH -t{hours}:{mins:02d}:00        # Duration of the job
#SBATCH -phive                         # Queue name
#SBATCH -oslurm/log/{log_dir}/%x.log      # Combined output and error

# Change to working directory
cd "$SLURM_SUBMIT_DIR" || exit
SCRIPT="{source_dir}/slurm/run_puc_srun.sh"
CFG_LOC="{source_dir}/config/pucn/"

"""

RUN_CMD = """
$SCRIPT -c "$CFG_LOC/{config_file}" -p {np}
"""


class ScalingRunGenerator:
    def __init__(self, account, source_dir, log_dir):
        self.account = account
        self.source_dir = source_dir
        self.log_dir = log_dir

    def header_for(self, nproc, name, hours, mins):
        return HEADER.format(
            name=name,
            account=self.account,
            ntasks=self.tasks_per_node(nproc),
            mem=self.node_mem(nproc),
            nodes=self.nodes(nproc),
            hours=hours,
            mins=mins,
            log_dir=self.log_dir,
            source_dir=self.source_dir,
        )

    def command_for(self, nproc, config_file):
        return RUN_CMD.format(config_file=config_file, np=nproc)

    def proc_commands(self, name, hours, mins, config_file, nruns, run_procs):
        def job_name(nproc):
            nodes = self.nodes(nproc)
            return f"{name}_p{nproc}n{nodes}"

        def file_name(nproc):
            jname = job_name(nproc)
            return f"{jname}.sh"

        def gen_command(nproc):
            header = self.header_for(nproc, job_name(nproc), hours, mins)
            cmd = self.command_for(nproc, config_file)
            run_cmd = "".join([cmd for _ in range(nruns)])
            return header + run_cmd

        return {x: (file_name(x), gen_command(x)) for x in run_procs}

    def node_cores(self) -> int:
        return 24

    def nodes(self, nproc):
        if nproc == 1:
            return 1
        if nproc == 2:
            return 2
        if nproc <= 64:
            return 4
        return int(nproc / 16)

    def tasks_per_node(self, nproc):
        if nproc <= 2:
            return 1
        if nproc <= 32:
            return int(nproc / 4)
        return 16

    def node_mem(self, nproc) -> str:
        if nproc == 1:
            return "750GB"
        if nproc == 2:
            return "360GB"
        return "192GB"

    def pow2_scaling(self, hours, mins, config_file, out_dir, nruns):
        run_procs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        name = pathlib.Path(config_file).stem
        cmds_dict = self.proc_commands(
            name, hours, mins, config_file, nruns, run_procs)
        print("Generating Files:  ", [fx for (fx, _) in cmds_dict.values()])
        for f_name, script_str in cmds_dict.values():
            with open(f"{out_dir}/{f_name}", "w") as fptr:
                fptr.write(script_str)

    def slurm_configs(self, hours, mins, cfg_files, out_dir, nruns, num_procs):
        print("Weak Scaling Config Files : ", cfg_files)
        for c_file in cfg_files:
            name = pathlib.Path(c_file).stem
            cmds_dict = self.proc_commands(
                name, hours, mins, c_file, nruns, num_procs
            )
            print("Generating Files:  ", [
                  fx for (fx, _) in cmds_dict.values()])
            for f_name, script_str in cmds_dict.values():
                with open(f"{out_dir}/{f_name}", "w") as fptr:
                    fptr.write(script_str)


def data_prep(args):
    source_dir = os.path.expandvars(args.source_dir)
    dest_dir = f"{source_dir}/slurm/scripts/prep/"
    sgen = ScalingRunGenerator(args.account, args.source_dir, "prep")
    shutil.rmtree(dest_dir, ignore_errors=True)
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    config_files = [
        "dhist_lung_c100Kg1K.yml",
        "dhist_lung_c100Kg3K.yml",
        "dhist_lung_c100Kg5K.yml",
        "dhist_lung_c100Kg8K.yml",
        "dhist_lung_c100Kg10K.yml",
        "dhist_lung_c100Kg12K.yml",
        "dhist_lung_c100Kg15K.yml",
        "dhist_lung_c100Kg18K.yml",
    ]
    sgen.slurm_configs(1, 00, config_files, dest_dir, 3, [512])


def strong_scaling(args, dest_dir):
    source_dir = os.path.expandvars(args.source_dir)
    dest_dir = f"{source_dir}/slurm/scripts/strong/"
    sgen = ScalingRunGenerator(args.account, args.source_dir, "strong")
    shutil.rmtree(dest_dir, ignore_errors=True)
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    sgen.pow2_scaling(1, 00, "dmisipuc_lung_c100Kg5K.yml", dest_dir, 3)


def weak_scaling(args, dest_dir):
    source_dir = os.path.expandvars(args.source_dir)
    dest_dir = f"{source_dir}/slurm/scripts/weak/"
    sgen = ScalingRunGenerator(args.account, args.source_dir, "weak")
    shutil.rmtree(dest_dir, ignore_errors=True)
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    config_files = [
        "dmisipuc_lung_c100Kg1K.yml",
        "dmisipuc_lung_c100Kg3K.yml",
        "dmisipuc_lung_c100Kg5K.yml",
        "dmisipuc_lung_c100Kg8K.yml",
        "dmisipuc_lung_c100Kg10K.yml",
        "dmisipuc_lung_c100Kg12K.yml",
        "dmisipuc_lung_c100Kg15K.yml",
        "dmisipuc_lung_c100Kg18K.yml",
    ]
    sgen.slurm_configs(1, 00, config_files, dest_dir, 3, [256])


def big_datasets(args, dest_dir):
    source_dir = os.path.expandvars(args.source_dir)
    dest_dir = f"{source_dir}/slurm/scripts/big/"
    sgen = ScalingRunGenerator(args.account, args.source_dir, "big")
    shutil.rmtree(dest_dir, ignore_errors=True)
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    olmisi_config_files = [
        "olmisi_pbmc_c800Kg20K.yml",
        "olmisi_pbmc_c800Kg30K.yml",
    ]
    puc_config_files = [
        "dpuc_pbmc_c800Kg20K.yml",
        "dpuc_pbmc_c800Kg30K.yml",
    ]
    sgen.slurm_configs(1, 00, olmisi_config_files, dest_dir, 3, [1024, 2048])
    sgen.slurm_configs(1, 00, puc_config_files, dest_dir, 3, [1024, 2048])


def create_output_dirs(out_dir):
    out_dirs = [
        "pbmc20K.5K",
        "rsc.10k",
        "rsc.12k",
        "rsc.15k",
        "rsc.18k",
        "rsc.1k",
        "rsc.3k",
        "rsc.5k",
        "rsc.8k",
        "pbmc800K.20K",
        "pbmc800K.30K",
    ]
    for rdir in out_dirs:
        dest_dir = f"{out_dir}/{rdir}/"
        pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)


def main(args):
    create_output_dirs(args.out_dir)
    data_prep(
        args,
    )
    strong_scaling(
        args,
    )
    weak_scaling(
        args,
    )
    big_datasets(
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process files from a source directory"
    )
    parser.add_argument(
        "-a", "--account", type=str, required=True, help="Account For Slurm"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="./out/",
        help="Output Directory: Relative to the Source Code Directory",
    )
    parser.add_argument(
        "-s",
        "--source_dir",
        type=str,
        default="$HOME/dev/parensnet_rs/",
        help="Path to source directory",
    )
    args = parser.parse_args()
    main(args)
