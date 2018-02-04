#!/bin/bash

CMAKE_BUILD_TYPE=$1
if [[ -z $CMAKE_BUILD_TYPE || ($CMAKE_BUILD_TYPE != "Debug" && $CMAKE_BUILD_TYPE != "Release") ]]; then
	echo "Please specify build-type: Debug/Release"
	exit 1
else
	echo "Specified build-type: $CMAKE_BUILD_TYPE"
fi

rm -r build
mkdir build
cd build

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

cmake  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE .. && make

# Revert original symlinks
sudo ln -s -f /usr/bin/gcc-4.8 /usr/bin/gcc
sudo ln -s -f /usr/bin/g++-4.8 /usr/bin/g++

echo ""
echo "Restoring original GCC / G++ symlinks ..."
echo "GCC: $(ls -l $(which gcc))"
echo "G++: $(ls -l $(which g++))"
echo ""

cd ..
