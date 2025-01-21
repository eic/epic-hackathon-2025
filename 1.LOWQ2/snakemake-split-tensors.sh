#!/bin/bash
source /home/simong/EIC/epic/install/bin/thisepic.sh
#source /home/simong/EIC/EICrecon/bin/eicrecon-this.sh

snakemake --cores 4 /scratch/EIC/EICreconTensors/targets.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/features.eicrecon.tree.edm4eic.root   --rerun-incomplete
