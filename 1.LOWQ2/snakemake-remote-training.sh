#!/bin/bash
#source /home/simong/EIC/epic/install/bin/thisepic.sh
#source /home/simong/EIC/EICrecon/bin/eicrecon-this.sh

snakemake --cores 2 /scratch/EIC/EICreconTensors/tensors_20.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_21.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_22.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_23.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_24.eicrecon.tree.edm4eic.root /scratch/EIC/EICreconTensors/tensors_25.eicrecon.tree.edm4eic.root  --rerun-incomplete
