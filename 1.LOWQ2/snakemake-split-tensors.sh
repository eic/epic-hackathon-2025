#!/bin/bash
source /home/simong/EIC/epic/install/bin/thisepic.sh
#source /home/simong/EIC/EICrecon/bin/eicrecon-this.sh

#snakemake --cores 8 targets_lowq2.edm4eic.root features_lowq2.edm4eic.root submit_lowq2.edm4eic.root   --rerun-incomplete

snakemake --cores 8 features_lowq2.edm4eic.root   --rerun-incomplete
