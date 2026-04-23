#!/bin/bash

mkdir -p data/

wget -O data/pbmc20K.zip https://zenodo.org/records/19670636/files/pbmc20K.zip?download=1
wget -O data/rsc_data.zip https://zenodo.org/records/19668587/files/rsc_data.zip?download=1

unzip data/pbmc20K.zip -d data/
unzip data/rsc_data.zip -d data/
