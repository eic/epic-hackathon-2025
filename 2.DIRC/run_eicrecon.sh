#!/usr/bin/env bash

export JANA_PLUGIN_PATH=EICrecon/prefix/lib/EICrecon/plugins
export LD_LIBRARY_PATH=EICrecon/prefix/lib:$LD_LIBRARY_PATH

source /opt/detector/epic-24.12.0/bin/thisepic.sh epic_inner_detector

INPUT_FILENAME="$1"
shift
OUTPUT_FILENAME="$(basename "${INPUT_FILENAME%%.eicrecon.edm4eic.root}.hackathon.edm4eic.root")"

exec EICrecon/prefix/bin/eicrecon \
    "${INPUT_FILENAME}" \
    -Ppodio:output_file="${OUTPUT_FILENAME}" \
    -Ppodio:output_collections=DIRCBarrelParticleIDDIRCInput_features,DIRCBarrelParticleIDTrackInput_features,DIRCBarrelParticleIDPIDTarget,DIRCBarrelParticleIDOutput_probability_tensor,ReconstructedChargedWithRealDIRCParticles,DIRCBarrelParticleIDs \
    $@
