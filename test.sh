#!/bin/bash
cd "$(dirname "$0")" # for GitHub Workspaces
sudo apt-get update
sudo apt-get install -y cmake g++ libeigen3-dev libboost-math-dev
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "------------------------------------------------"
./examples/market_crash_analysis