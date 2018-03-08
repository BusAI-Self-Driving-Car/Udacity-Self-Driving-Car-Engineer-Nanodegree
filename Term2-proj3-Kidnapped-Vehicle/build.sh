#!/bin/bash
# Script to build all components from scratch, using the maximum available CPU power
#
# Given parameters are passed over to CMake.
# Examples:
#    * ./build_all.sh -DCMAKE_BUILD_TYPE=Debug
#    * ./build_all.sh VERBOSE=1
#
# Written by Tiffany Huang, 12/14/2016
#

# Go into the directory where this bash script is contained.
cd `dirname $0`

export PATH=~/Documents/udas/software/cmake-3.6.0-Linux-x86_64/bin:$PATH
export PATH=~/Documents/udas/software/make-4.2:$PATH

# Symlink to a newer GCC (required by this project)
sudo ln -s -f /usr/bin/gcc-6 /usr/bin/gcc
sudo ln -s -f /usr/bin/g++-6 /usr/bin/g++

echo ""
echo "CMAKE: $(which cmake)"
echo "MAKE: $(which make)"
echo ""
echo "Created new GCC / G++ symlinks for gXX-6:"
echo "GCC: $(ls -l $(which gcc))"
echo "G++: $(ls -l $(which g++))"
echo ""

# Compile code.
mkdir -p build
cd build
cmake ..
make -j `nproc` $*

# Revert original symlinks
sudo ln -s -f /usr/bin/gcc-4.8 /usr/bin/gcc
sudo ln -s -f /usr/bin/g++-4.8 /usr/bin/g++

echo ""
echo "Restoring original GCC / G++ symlinks ..."
echo "GCC: $(ls -l $(which gcc))"
echo "G++: $(ls -l $(which g++))"
echo ""

