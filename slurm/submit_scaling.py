#!/bin/python
import subprocess

P20K5K = {
    "misi": [
        "misi_pbmc_c20Kg5K_p2n1.sh",
        "misi_pbmc_c20Kg5K_p4n1.sh",
        "misi_pbmc_c20Kg5K_p8n1.sh",
        "misi_pbmc_c20Kg5K_p16n1.sh",
        "misi_pbmc_c20Kg5K_p32n2.sh",
        "misi_pbmc_c20Kg5K_p64n4.sh",
        "misi_pbmc_c20Kg5K_p128n8.sh",
        "misi_pbmc_c20Kg5K_p256n16.sh",
        "misi_pbmc_c20Kg5K_p512n32.sh",
        # "misi_pbmc_c20Kg5K_p1024n64.sh",
    ],
    "lpuc": [
        "lpuc_pbmc_c20Kg5K_p2n1.sh",
        "lpuc_pbmc_c20Kg5K_p4n1.sh",
        "lpuc_pbmc_c20Kg5K_p8n1.sh",
        "lpuc_pbmc_c20Kg5K_p16n1.sh",
        "lpuc_pbmc_c20Kg5K_p32n2.sh",
        "lpuc_pbmc_c20Kg5K_p64n4.sh",
        "lpuc_pbmc_c20Kg5K_p128n8.sh",
        "lpuc_pbmc_c20Kg5K_p256n16.sh",
        "lpuc_pbmc_c20Kg5K_p512n32.sh",
        "lpuc_pbmc_c20Kg5K_p1024n64.sh",
    ],
    "dhist": [
        "dhist_pbmc_c20Kg5K_p2n1.sh",
        "dhist_pbmc_c20Kg5K_p4n1.sh",
        "dhist_pbmc_c20Kg5K_p8n1.sh",
        "dhist_pbmc_c20Kg5K_p16n1.sh",
        "dhist_pbmc_c20Kg5K_p32n2.sh",
        "dhist_pbmc_c20Kg5K_p64n4.sh",
        "dhist_pbmc_c20Kg5K_p128n8.sh",
        "dhist_pbmc_c20Kg5K_p256n16.sh",
        "dhist_pbmc_c20Kg5K_p512n32.sh",
        "dhist_pbmc_c20Kg5K_p1024n64.sh",
    ],
    "dmisi": [
        "dmisi_pbmc_c20Kg5K_p2n1.sh",
        "dmisi_pbmc_c20Kg5K_p4n1.sh",
        "dmisi_pbmc_c20Kg5K_p8n1.sh",
        "dmisi_pbmc_c20Kg5K_p16n1.sh",
        "dmisi_pbmc_c20Kg5K_p32n2.sh",
        "dmisi_pbmc_c20Kg5K_p64n4.sh",
        "dmisi_pbmc_c20Kg5K_p128n8.sh",
        "dmisi_pbmc_c20Kg5K_p256n16.sh",
        "dmisi_pbmc_c20Kg5K_p512n32.sh",
        "dmisi_pbmc_c20Kg5K_p1024n64.sh",
    ],

    "dpuc": [
        "dpuc_pbmc_c20Kg5K_p2n1.sh",
        "dpuc_pbmc_c20Kg5K_p4n1.sh",
        "dpuc_pbmc_c20Kg5K_p8n1.sh",
        "dpuc_pbmc_c20Kg5K_p16n1.sh",
        "dpuc_pbmc_c20Kg5K_p32n2.sh",
        "dpuc_pbmc_c20Kg5K_p64n4.sh",
        "dpuc_pbmc_c20Kg5K_p128n8.sh",
        "dpuc_pbmc_c20Kg5K_p256n16.sh",
        "dpuc_pbmc_c20Kg5K_p512n32.sh",
        "dpuc_pbmc_c20Kg5K_p1024n64.sh",
    ],
}

P100K5K = {
    "misi": [
        # "misi_pbmc_c100Kg5K_p2n1.sh",
        # "misi_pbmc_c100Kg5K_p4n1.sh",
        # "misi_pbmc_c100Kg5K_p8n1.sh",
        # "misi_pbmc_c100Kg5K_p16n1.sh",
        "misi_pbmc_c100Kg5K_p32n2.sh",
        "misi_pbmc_c100Kg5K_p64n4.sh",
        "misi_pbmc_c100Kg5K_p128n8.sh",
        "misi_pbmc_c100Kg5K_p256n16.sh",
        "misi_pbmc_c100Kg5K_p512n32.sh",
        "misi_pbmc_c100Kg5K_p1024n64.sh",
    ],
    "lpuc": [
        # "lpuc_pbmc_c100Kg5K_p2n1.sh",
        # "lpuc_pbmc_c100Kg5K_p4n1.sh",
        # "lpuc_pbmc_c100Kg5K_p8n1.sh",
        # "lpuc_pbmc_c100Kg5K_p16n1.sh",
        "lpuc_pbmc_c100Kg5K_p32n2.sh",
        "lpuc_pbmc_c100Kg5K_p64n4.sh",
        "lpuc_pbmc_c100Kg5K_p128n8.sh",
        "lpuc_pbmc_c100Kg5K_p256n16.sh",
        "lpuc_pbmc_c100Kg5K_p512n32.sh",
        "lpuc_pbmc_c100Kg5K_p1024n64.sh",
    ],
    "dhist": [
        # "dhist_pbmc_c100Kg5K_p2n1.sh",
        # "dhist_pbmc_c100Kg5K_p4n1.sh",
        # "dhist_pbmc_c100Kg5K_p8n1.sh",
        # "dhist_pbmc_c100Kg5K_p16n1.sh",
        "dhist_pbmc_c100Kg5K_p32n2.sh",
        "dhist_pbmc_c100Kg5K_p64n4.sh",
        "dhist_pbmc_c100Kg5K_p128n8.sh",
        "dhist_pbmc_c100Kg5K_p256n16.sh",
        "dhist_pbmc_c100Kg5K_p512n32.sh",
        "dhist_pbmc_c100Kg5K_p1024n64.sh",
    ],
    "dnodes2misi": [
        # "dnodes2misi_pbmc_c100Kg5K_p2n1.sh",
        # "dnodes2misi_pbmc_c100Kg5K_p4n1.sh",
        # "dnodes2misi_pbmc_c100Kg5K_p8n1.sh",
        # "dnodes2misi_pbmc_c100Kg5K_p16n1.sh",
        "dnodes2misi_pbmc_c100Kg5K_p32n2.sh",
        "dnodes2misi_pbmc_c100Kg5K_p64n4.sh",
        "dnodes2misi_pbmc_c100Kg5K_p128n8.sh",
        "dnodes2misi_pbmc_c100Kg5K_p256n16.sh",
        "dnodes2misi_pbmc_c100Kg5K_p512n32.sh",
        "dnodes2misi_pbmc_c100Kg5K_p1024n64.sh",
    ],
}


P100K10K = {
    "misi": [
        # "misi_pbmc_c100Kg10K_p2n1.sh",
        # "misi_pbmc_c100Kg10K_p4n1.sh",
        # "misi_pbmc_c100Kg10K_p8n1.sh",
        # "misi_pbmc_c100Kg10K_p16n1.sh",
        # "misi_pbmc_c100Kg10K_p32n2.sh",
        # "misi_pbmc_c100Kg10K_p64n4.sh",
        "misi_pbmc_c100Kg10K_p128n8.sh",
        "misi_pbmc_c100Kg10K_p256n16.sh",
        "misi_pbmc_c100Kg10K_p512n32.sh",
        "misi_pbmc_c100Kg10K_p1024n64.sh",
    ],
    "lpuc": [
        # "misi_pbmc_c100Kg10K_p2n1.sh",
        # "misi_pbmc_c100Kg10K_p4n1.sh",
        # "misi_pbmc_c100Kg10K_p8n1.sh",
        # "misi_pbmc_c100Kg10K_p16n1.sh",
        # "misi_pbmc_c100Kg10K_p32n2.sh",
        # "misi_pbmc_c100Kg10K_p64n4.sh",
        "lpuc_pbmc_c100Kg10K_p128n8.sh",
        "lpuc_pbmc_c100Kg10K_p256n16.sh",
        "lpuc_pbmc_c100Kg10K_p512n32.sh",
        "lpuc_pbmc_c100Kg10K_p1024n64.sh",
    ],
    "dnodes2misi": [
        # "dnodes2misi_pbmc_c100Kg10K_p2n1.sh",
        # "dnodes2misi_pbmc_c100Kg10K_p4n1.sh",
        # "dnodes2misi_pbmc_c100Kg10K_p8n1.sh",
        # "dnodes2misi_pbmc_c100Kg10K_p16n1.sh",
        # "dnodes2misi_pbmc_c100Kg10K_p32n2.sh",
        # "dnodes2misi_pbmc_c100Kg10K_p64n4.sh",
        "dnodes2misi_pbmc_c100Kg10K_p128n8.sh",
        "dnodes2misi_pbmc_c100Kg10K_p256n16.sh",
        "dnodes2misi_pbmc_c100Kg10K_p512n32.sh",
        "dnodes2misi_pbmc_c100Kg10K_p1024n64.sh",
    ],
    "dmisi": [
        # "dmisi_pbmc_c100Kg10K_p2n1.sh",
        # "dmisi_pbmc_c100Kg10K_p4n1.sh",
        # "dmisi_pbmc_c100Kg10K_p8n1.sh",
        # "dmisi_pbmc_c100Kg10K_p16n1.sh",
        # "dmisi_pbmc_c100Kg10K_p32n2.sh",
        # "dmisi_pbmc_c100Kg10K_p64n4.sh",
        "dmisi_pbmc_c100Kg10K_p128n8.sh",
        "dmisi_pbmc_c100Kg10K_p256n16.sh",
        "dmisi_pbmc_c100Kg10K_p512n32.sh",
        "dmisi_pbmc_c100Kg10K_p1024n64.sh",
    ],
    "dpuc": [
        # "dpuc_pbmc_c100Kg10K_p2n1.sh",
        # "dpuc_pbmc_c100Kg10K_p4n1.sh",
        # "dpuc_pbmc_c100Kg10K_p8n1.sh",
        # "dpuc_pbmc_c100Kg10K_p16n1.sh",
        # "dpuc_pbmc_c100Kg10K_p32n2.sh",
        # "dpuc_pbmc_c100Kg10K_p64n4.sh",
        "dpuc_pbmc_c100Kg10K_p128n8.sh",
        "dpuc_pbmc_c100Kg10K_p256n16.sh",
        "dpuc_pbmc_c100Kg10K_p512n32.sh",
        "dpuc_pbmc_c100Kg10K_p1024n64.sh",
    ],

}

P100K12K = {
    "misi": [
        # "misi_pbmc_c100Kg12K_p2n1.sh",
        # "misi_pbmc_c100Kg12K_p4n1.sh",
        # "misi_pbmc_c100Kg12K_p8n1.sh",
        # "misi_pbmc_c100Kg12K_p16n1.sh",
        # "misi_pbmc_c100Kg12K_p32n2.sh",
        # "misi_pbmc_c100Kg12K_p64n4.sh",
        "misi_pbmc_c100Kg12K_p128n8.sh",
        "misi_pbmc_c100Kg12K_p256n16.sh",
        "misi_pbmc_c100Kg12K_p512n32.sh",
        "misi_pbmc_c100Kg12K_p1024n64.sh",
    ],
    "lpuc": [
        # "lpuc_pbmc_c100Kg12K_p2n1.sh",
        # "lpuc_pbmc_c100Kg12K_p4n1.sh",
        # "lpuc_pbmc_c100Kg12K_p8n1.sh",
        # "lpuc_pbmc_c100Kg12K_p16n1.sh",
        # "lpuc_pbmc_c100Kg12K_p32n2.sh",
        # "lpuc_pbmc_c100Kg12K_p64n4.sh",
        "lpuc_pbmc_c100Kg12K_p128n8.sh",
        "lpuc_pbmc_c100Kg12K_p256n16.sh",
        "lpuc_pbmc_c100Kg12K_p512n32.sh",
        "lpuc_pbmc_c100Kg12K_p1024n64.sh",
    ],
    "dnodes2misi": [
        # "dnodes2misi_pbmc_c100Kg12K_p2n1.sh",
        # "dnodes2misi_pbmc_c100Kg12K_p4n1.sh",
        # "dnodes2misi_pbmc_c100Kg12K_p8n1.sh",
        # "dnodes2misi_pbmc_c100Kg12K_p16n1.sh",
        # "dnodes2misi_pbmc_c100Kg12K_p32n2.sh",
        # "dnodes2misi_pbmc_c100Kg12K_p64n4.sh",
        "dnodes2misi_pbmc_c100Kg12K_p128n8.sh",
        "dnodes2misi_pbmc_c100Kg12K_p256n16.sh",
        "dnodes2misi_pbmc_c100Kg12K_p512n32.sh",
        "dnodes2misi_pbmc_c100Kg12K_p1024n64.sh",
    ],
    "dmisi": [
        # "dmisi_pbmc_c100Kg12K_p2n1.sh",
        # "dmisi_pbmc_c100Kg12K_p4n1.sh",
        # "dmisi_pbmc_c100Kg12K_p8n1.sh",
        # "dmisi_pbmc_c100Kg12K_p16n1.sh",
        # "dmisi_pbmc_c100Kg12K_p32n2.sh",
        # "dmisi_pbmc_c100Kg12K_p64n4.sh",
        "dmisi_pbmc_c100Kg12K_p128n8.sh",
        "dmisi_pbmc_c100Kg12K_p256n16.sh",
        "dmisi_pbmc_c100Kg12K_p512n32.sh",
        "dmisi_pbmc_c100Kg12K_p1024n64.sh",
    ],
    "dpuc": [
        # "dpuc_pbmc_c100Kg12K_p2n1.sh",
        # "dpuc_pbmc_c100Kg12K_p4n1.sh",
        # "dpuc_pbmc_c100Kg12K_p8n1.sh",
        # "dpuc_pbmc_c100Kg12K_p16n1.sh",
        # "dpuc_pbmc_c100Kg12K_p32n2.sh",
        # "dpuc_pbmc_c100Kg12K_p64n4.sh",
        "dpuc_pbmc_c100Kg12K_p128n8.sh",
        "dpuc_pbmc_c100Kg12K_p256n16.sh",
        "dpuc_pbmc_c100Kg12K_p512n32.sh",
        "dpuc_pbmc_c100Kg12K_p1024n64.sh",
    ],
}


SCRIPT_DIR = "./slurm/scripts/"


def submit_jobs(job_list: list[str]):
    print(job_list)
    first_script = f"{SCRIPT_DIR}/{job_list[0]}"
    first_result = subprocess.run(["sbatch", first_script], stdout=subprocess.PIPE)
    print(first_result, ":", first_result.stdout)
    assert first_result.stdout.startswith(b"Submitted batch job")
    parent = first_result.stdout.strip().split()[-1]
    parent = parent.decode("utf-8")
    for script in job_list[1:]:
        script_path = f"{SCRIPT_DIR}/{script}"
        parent_arg = f"afterany:{parent}"
        script_result = subprocess.run(
            ["sbatch", "-d", parent_arg, script_path], stdout=subprocess.PIPE
        )
        print(script_path, parent_arg, ":", str(script_result.stdout))
        assert script_result.stdout.startswith(b"Submitted batch job")
        parent = script_result.stdout.strip().split()[-1]
        parent = parent.decode("utf-8")


def main():
    # Run from the root directory
    # submit_jobs(P20K5K["misi"])
    # submit_jobs(P100K5K["misi"])
    # submit_jobs(P100K10K["misi"])
    # submit_jobs(P100K12K["misi"])
    # submit_jobs(P20K5K["lpuc"])
    # submit_jobs(P100K5K["lpuc"])
    # submit_jobs(P100K10K["lpuc"])
    # submit_jobs(P100K12K["lpuc"])
    #
    # submit_jobs(P20K5K["dhist"])
    # submit_jobs(P20K5K["dmisi"])
    # submit_jobs(P20K5K["dpuc"])
    # submit_jobs(P100K5K["dhist"])
    # submit_jobs(P100K5K["dnodes2misi"])
    # submit_jobs(P100K10K["dnodes2misi"])
    # submit_jobs(P100K12K["dnodes2misi"])
    # submit_jobs(P100K12K["dmisi"])
    # submit_jobs(P100K10K["dmisi"])
    submit_jobs(P100K10K["dpuc"])
    submit_jobs(P100K12K["dpuc"])


if __name__ == "__main__":
    main()
