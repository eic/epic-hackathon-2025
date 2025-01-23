#!/usr/bin/env bash

set -xue

if [ ! -d EICrecon ]; then
  git clone -b dirc_2025 https://github.com/eic/EICrecon.git
fi
cd EICrecon/
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=$PWD/prefix -DCMAKE_INSTALL_RPATH=$PWD/prefix/lib/EICrecon/plugins/ -DCMAKE_BUILD_TYPE=Release
cmake --build build -j `nproc`
cmake --install build
