#!/bin/bash
mkdir -p ./out/pbmc20K.5K/
echo "Build distributions"
./slurm/run_puc.sh -c config/pucn/dhist_pbmc_c20Kg5K.yml -p 64 2>&1 | tee -a log/dhist_pbmc_c20Kg5K_p64n1.log

echo "Running for 64 cores"
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 64 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p64n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 64 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p64n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 64 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p64n1.log

echo "Running for 32 cores"
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 32 2>&1 | tee log/olmisipuc_pbmc_c20Kg5K_p32n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 32 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p32n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 32 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p32n1.log

echo "Running for 16 cores"
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 16 2>&1 | tee log/olmisipuc_pbmc_c20Kg5K_p16n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 16 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p16n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 16 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p16n1.log

echo "Running for 8 cores"
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 8 2>&1 | tee log/olmisipuc_pbmc_c20Kg5K_p8n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 8 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p8n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 8 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p8n1.log


echo "Running for 4 cores"
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 4 2>&1 | tee log/olmisipuc_pbmc_c20Kg5K_p4n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 4 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p4n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 4 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p4n1.log


echo "Running for 2 cores"
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 2 2>&1 | tee log/olmisipuc_pbmc_c20Kg5K_p2n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 2 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p2n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 2 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p2n1.log

echo "Running for 1 cores"
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 1 2>&1 | tee log/olmisipuc_pbmc_c20Kg5K_p1n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 1 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p1n1.log
./slurm/run_puc.sh -c config/pucn/olmisipuc_pbmc_c20Kg5K.yml -p 1 2>&1 | tee -a log/olmisipuc_pbmc_c20Kg5K_p1n1.log
