import abc
import pathlib
from abc import abstractmethod

from typing_extensions import override

HEADER = """#!/bin/bash
#SBATCH -J{name}         # Job name
#SBATCH -Agts-saluru8-coda20          # Tracking account
#SBATCH -N{nodes}                       # Number of nodes required
#SBATCH --ntasks-per-node 16          # Number of cores/node required
#SBATCH --mem-per-cpu=12G             # Memory per core
#SBATCH -t{hours}:{mins:02d}:00          # Duration of the job
#SBATCH -phive                        # Queue name
#SBATCH -oslurm/log/%x.log            # Combined output and error

# Change to working directory
cd "$SLURM_SUBMIT_DIR" || exit
SCRIPT="$HOME/dev/parensnet_rs/slurm/run_puc.sh"
CFG_LOC="$HOME/dev/parensnet_rs/config/pucn/"

"""

RUN_CMD = """
$SCRIPT -c "$CFG_LOC/{config_file}" -p {np}
"""

RUN_CMD_NPR = """
$SCRIPT -c "$CFG_LOC/{config_file}" -p {np} -n {npr}
"""


def header(name, nodes, hours, mins):
    return HEADER.format(name=name, nodes=nodes, hours=hours, mins=mins)


def command(config_file, np, npr):
    if npr is None:
        return RUN_CMD.format(config_file=config_file, np=np)
    else:
        return RUN_CMD_NPR.format(config_file=config_file, np=np, npr=npr)


class ScalingRun(abc.ABC):
    @abstractmethod
    def node_limit(self) -> int:
        return 0

    @abstractmethod
    def node_cores(self) -> int:
        return 0

    @abstractmethod
    def run_procs(self) -> list[int]:
        return []

    def nodes_for(self, nproc):
        return 1 if nproc <= self.node_limit() else int(nproc / self.node_limit())

    def header_for(self, nproc, name, hours, mins):
        return header(name, self.nodes_for(nproc), hours, mins)

    def command_for(self, nproc, config_file):
        return command(
            config_file, nproc, self.node_limit() if nproc > self.node_cores() else None
        )

    def generate_commands(self, name, hours, mins, config_file):
        def job_name(nproc):
            nodes = self.nodes_for(nproc)
            return f"{name}_p{nproc}n{nodes}"

        def file_name(nproc):
            jname = job_name(nproc)
            return f"{jname}.sh"

        def gen_command(nproc):
            return self.header_for(
                nproc, job_name(nproc), hours, mins
            ) + self.command_for(nproc, config_file)

        return {x: (file_name(x), gen_command(x)) for x in self.run_procs()}

    def generate_sbatches(self, hours, mins, config_file, out_dir):
        name = pathlib.Path(config_file).stem
        cmds_dict = self.generate_commands(name, hours, mins, config_file)
        print("Generating Files:  ", [fx for (fx, _) in cmds_dict.values()])
        for f_name, script_str in cmds_dict.values():
            with open(f"{out_dir}/{f_name}", "w") as fptr:
                fptr.write(script_str)


class Pow2Scaling(ScalingRun):
    @override
    def run_procs(self) -> list[int]:
        return [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    @override
    def node_limit(self) -> int:
        return 16

    @override
    def node_cores(self) -> int:
        return 24


class Pow32Scaling(ScalingRun):
    @override
    def run_procs(self) -> list[int]:
        return [3, 8, 24, 72, 216, 648, 1944, 5832]

    @override
    def node_limit(self) -> int:
        return 24

    @override
    def node_cores(self) -> int:
        return 24


def main():
    # Pow2Scaling().generate_sbatches(1, 0, "misi_pbmc_c20Kg0K500.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "misi_pbmc_c20Kg5K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "misi_pbmc_c100Kg5K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "misi_pbmc_c100Kg10K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "misi_pbmc_c100Kg12K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "lpuc_pbmc_c20Kg5K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "lpuc_pbmc_c100Kg5K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "lpuc_pbmc_c100Kg10K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "lpuc_pbmc_c100Kg12K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "dhist_pbmc_c20Kg5K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "dmisi_pbmc_c20Kg5K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(0, 30, "dmisi_pbmc_c100Kg1K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(0, 30, "dmisi_pbmc_c100Kg3K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dmisi_pbmc_c100Kg5K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dmisi_pbmc_c100Kg8K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dmisi_pbmc_c100Kg10K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dmisi_pbmc_c100Kg12K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "dpuc_pbmc_c20Kg5K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(0, 30, "dpuc_pbmc_c100Kg1K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(0, 30, "dpuc_pbmc_c100Kg3K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dpuc_pbmc_c100Kg5K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dpuc_pbmc_c100Kg8K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dpuc_pbmc_c100Kg10K.yml", "./slurm/scripts/")
    Pow2Scaling().generate_sbatches(1, 00, "dpuc_pbmc_c100Kg12K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "dhist_pbmc_c100Kg5K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "dhist_pbmc_c100Kg10K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "dhist_pbmc_c100Kg12K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(1, 0, "dnodes2misi_pbmc_c100Kg5K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(2, 0, "dnodes2misi_pbmc_c100Kg10K.yml", "./slurm/scripts/")
    # Pow2Scaling().generate_sbatches(2, 0, "dnodes2misi_pbmc_c100Kg12K.yml", "./slurm/scripts/")


if __name__ == "__main__":
    main()
