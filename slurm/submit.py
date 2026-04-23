import argparse
import os
import subprocess
import sys
import typing as t

type JobType = t.Literal["hist", "strong", "weak", "big"]

STRONG_SCALING = {
    "misipuc": [
        "dmisipuc_lung_c100Kg5K_p1n1.sh",
        "dmisipuc_lung_c100Kg5K_p2n2.sh",
        "dmisipuc_lung_c100Kg5K_p4n2.sh",
        "dmisipuc_lung_c100Kg5K_p8n2.sh",
        "dmisipuc_lung_c100Kg5K_p16n4.sh",
        "dmisipuc_lung_c100Kg5K_p32n2.sh",
        "dmisipuc_lung_c100Kg5K_p64n4.sh",
        "dmisipuc_lung_c100Kg5K_p128n8.sh",
        "dmisipuc_lung_c100Kg5K_p256n16.sh",
        "dmisipuc_lung_c100Kg5K_p512n32.sh",
        "dmisipuc_lung_c100Kg5K_p1024n64.sh",
    ],
}

WEAK_SCALING = {
    "dmisipuc": [
        "dmisipuc_lung_c100Kg1K_p256n64.sh",
        "dmisipuc_lung_c100Kg3K_p256n64.sh",
        "dmisipuc_lung_c100Kg5K_p256n64.sh",
        "dmisipuc_lung_c100Kg8K_p256n64.sh",
        "dmisipuc_lung_c100Kg10K_p256n64.sh",
        "dmisipuc_lung_c100Kg12K_p256n64.sh",
        "dmisipuc_lung_c100Kg15K_p256n64.sh",
        "dmisipuc_lung_c100Kg18K_p256n64.sh",
    ],
    "dhist": [
        "dhist_lung_c100Kg1K_p256n64.sh",
        "dhist_lung_c100Kg3K_p256n64.sh",
        "dhist_lung_c100Kg5K_p256n64.sh",
        "dhist_lung_c100Kg8K_p256n64.sh",
        "dhist_lung_c100Kg10K_p256n64.sh",
        "dhist_lung_c100Kg12K_p256n64.sh",
        "dhist_lung_c100Kg15K_p256n64.sh",
        "dhist_lung_c100Kg18K_p256n64.sh",
    ],
}

BIG_SCALING = {
    "dmisipuc": [
        "olmisi_pbmc_c800Kg20K_p1024n64.sh",
        "dpuc_pbmc_c800Kg20K_p1024n64.sh",
        "olmisi_pbmc_c800Kg30K_p1024n64.sh",
        "dpuc_pbmc_c800Kg30K_p1024n64.sh",
        "olmisi_pbmc_c800Kg20K_p2048n128.sh",
        "dpuc_pbmc_c800Kg20K_p2048n128.sh",
        "olmisi_pbmc_c800Kg30K_p2048n128.sh",
        "dpuc_pbmc_c800Kg30K_p2048n128.sh",
    ],
}

STRONG_SCRIPT_DIR = "./slurm/scripts/strong/"
BIG_SCRIPT_DIR = "./slurm/scripts/big/"
WEAK_SCRIPT_DIR = "./slurm/scripts/weak/"
PREP_SCRIPT_DIR = "./slurm/scripts/prep/"


def submit_jobs(script_dir: str, job_list: list[str]):
    print(job_list)
    first_script = f"{script_dir}/{job_list[0]}"
    first_result = subprocess.run(
        ["sbatch", first_script], stdout=subprocess.PIPE)
    print(first_result, ":", first_result.stdout)
    assert first_result.stdout.startswith(b"Submitted batch job")
    parent = first_result.stdout.strip().split()[-1]
    parent = parent.decode("utf-8")
    for script in job_list[1:]:
        script_path = f"{script_dir}/{script}"
        if not os.path.isfile(script_path):
            print(f"Missing file {script_path}")
            return 1
    for script in job_list[1:]:
        script_path = f"{script_dir}/{script}"
        parent_arg = f"afterany:{parent}"
        script_result = subprocess.run(
            ["sbatch", "-d", parent_arg, script_path], stdout=subprocess.PIPE
        )
        print(script_path, parent_arg, ":", str(script_result.stdout))
        assert script_result.stdout.startswith(b"Submitted batch job")
        parent = script_result.stdout.strip().split()[-1]
        parent = parent.decode("utf-8")


def main(job: JobType):
    match job:
        case "hist":
            submit_jobs(PREP_SCRIPT_DIR, STRONG_SCALING["dhist"])
        case "strong":
            submit_jobs(STRONG_SCRIPT_DIR, STRONG_SCALING["dmisipuc"])
        case "weak":
            submit_jobs(WEAK_SCRIPT_DIR, WEAK_SCALING["dmisipuc"])
        case "big":
            submit_jobs(BIG_SCRIPT_DIR, BIG_SCALING["dmisipuc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process files from a source directory"
    )
    parser.add_argument(
        "-j", "--job", required=True, choices=["hist", "strong", "weak", "big"]
    )
    args = parser.parse_args()
    main(args.job)
