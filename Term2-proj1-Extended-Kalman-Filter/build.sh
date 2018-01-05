#!/bin/bash

rm -r build
mkdir build
cd build

export PATH=~/Documents/udas/software/cmake-3.6.0-Linux-x86_64/bin:$PATH
export PATH=~/Documents/udas/software/make-4.2:$PATH

# Symlink to a newer GCC (required by this project)
sudo rm -f /usr/bin/gcc; sudo ln -s /usr/bin/gcc-6 /usr/bin/gcc
sudo rm -f /usr/bin/g++; sudo ln -s /usr/bin/g++-6 /usr/bin/g++

echo "CMAKE: $(which cmake)"
echo ""
echo "MAKE: $(which make)"
echo ""
echo "GCC: $(ls -l $(which gcc))"
echo "G++: $(ls -l $(which g++))"
echo ""

cmake .. && make

# Revert original symlinks
sudo rm -f /usr/bin/gcc; sudo ln -s /usr/bin/gcc-4.8 /usr/bin/gcc
sudo rm -f /usr/bin/g++; sudo ln -s /usr/bin/g++-4.8 /usr/bin/g++

echo ""
echo "Restoring original GCC / G++ symlinks ..."
echo "GCC: $(ls -l $(which gcc))"
echo "G++: $(ls -l $(which g++))"
echo ""

cd ..
