#!/bin/bash
#source /home/simong/EIC/epic/install/bin/thisepic.sh
#source /home/simong/EIC/EICrecon/bin/eicrecon-this.sh

snakemake --cores 2 /scratch/EIC/EICreconTensors/tensors_8.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_9.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_10.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_11.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_12.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_13.eicrecon.tree.edm4eic.root  --rerun-incomplete
