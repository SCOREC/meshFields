# meshFields
GPU friendly unstructured mesh field storage and interface

## build dependencies

The following commands were tested on a SCOREC workstation running RHEL7 with a
Nvidia Turing GPU.

`cd` to a working directory that will contain *all* your source code (including
this directory) and build directories.  That directory is referred to as `root`
in the following bash scripts.

Create a file named `envRhel7_turing.sh` with the following contents:

```
export root=$PWD 
module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core 
module use /opt/scorec/spack/v0154_2/lmod/linux-rhel7-x86_64/Core 
module load gcc/10.1.0 cmake cuda/11.4

function getname() {
  name=$1
  machine=`hostname -s`
  buildSuffix=${machine}-cuda
  echo "build-${name}-${buildSuffix}"
}
export kk=$root/`getname kokkos`/install
export oh=$root/`getname omegah`/install
export cab=$root/`getname cabana`/install
CMAKE_PREFIX_PATH=$kk:$kk/lib64/cmake:$oh:$cab:$CMAKE_PREFIX_PATH

cm=`which cmake`
echo "cmake: $cm"
echo "kokkos install dir: $kk"
```


Create a file named `buildAll_turing.sh` with the following contents:

```
#!/bin/bash -e

#kokkos
cd $root
#tested with kokkos develop@9dff8cc
git clone -b develop git@github.com:kokkos/kokkos.git
mkdir -p $kk
cd $_/..
cmake ../kokkos \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ENABLE_CUDA=on \
  -DKokkos_ENABLE_CUDA_LAMBDA=on \
  -DKokkos_ENABLE_DEBUG=on \
  -DKokkos_ENABLE_PROFILING=on \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j 24 install

#omegah
cd $root
git clone git@github.com:sandialabs/omega_h.git
[ -d $oh ] && rm -rf ${oh%%install}
mkdir -p $oh 
cd ${oh%%install}
cmake ../omega_h \
  -DCMAKE_INSTALL_PREFIX=$oh \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=on \
  -DOmega_h_CUDA_ARCH=75 \
  -DOmega_h_USE_MPI=OFF  \
  -DBUILD_TESTING=on  \
  -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_PREFIX=$kk/lib64/cmake
make VERBOSE=1 -j8 install
ctest

#cabana
cd $root
git clone git@github.com:ECP-copa/Cabana.git cabana
[ -d $cab ] && rm -rf ${cab%%install}
mkdir -p $cab
cd ${cab%%install}
cmake ../cabana \
  -DCMAKE_BUILD_TYPE="Debug" \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DCabana_ENABLE_MPI=OFF \
  -DCabana_ENABLE_CAJITA=OFF \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_EXAMPLES=OFF \
  -DCabana_ENABLE_Cuda=ON \
  -DCMAKE_INSTALL_PREFIX=$cab
make -j 24 install
```

Make the script executable:

```
chmod +x buildAll_turing.sh
```


Source the environment script from this work directory:

```
source envRhel7_turing.sh
```

Run the build script:

```
./buildAll_turing.sh
```

## build dependencies for CPU

This build serves to help debugging, check for memory leaks via valgrind, and
run tests for coverage. (GPU code is not registered as having been run by GCOV 
and subsequently LCOV.) These commands can be run in your root directory but you 
will need to re-source the enviornment script and re-run the build script for the 
GPU build when you want to return to it.

`cd` to a working directory that will contain *all* your source code (including
this directory) and build directories.  That directory is referred to as `root`
in the following bash scripts.

Create a file named `envCpu.sh` with the following contents:

```
export root=$PWD
module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core
module use /opt/scorec/spack/v0154_2/lmod/linux-rhel7-x86_64/Core
module load lcov/1.16 py-gcovr/4.2
module load gcc/10.1.0 cmake
module load perl/5.30.3
module load perl-io-compress/2.081
function getname() {
  name=$1
  machine=`hostname -s`
  buildSuffix=${machine}-cpu
  echo "build-${name}-${buildSuffix}"
}
export kkCpu=$root/`getname kokkos`/install
export ohCpu=$root/`getname omegah`/install
export cabCpu=$root/`getname cabana`/install
CMAKE_PREFIX_PATH=$kkCpu:$kkCpu/lib64/cmake:$ohCpu:$cabCpu:$CMAKE_PREFIX_PATH
cm=`which cmake`
echo "cmake: $cm"
echo "kokkos install dir: $kkCpu"

```
Create a file named `buildAllCpu.sh` with the following contents:
(git clone commands for kokkos, cabana, and omegah are commented
out as it is assumed you have created a GPU build previously.)

```
#!/bin/bash -e
#kokkos
cd $root
#tested with kokkos develop@9dff8cc
#git clone -b develop https://github.com/kokkos/kokkos.git
mkdir -p $kkCpu
cd $_/..
cmake ../kokkos \
  -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ENABLE_DEBUG=on \
  -DKokkos_ENABLE_PROFILING=on \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j 24 install
#omegah
cd $root
#git clone https://github.com/sandialabs/omega_h.git
[ -d $ohCpu ] && rm -rf ${ohCpu%%install}
mkdir -p $ohCpu
cd ${ohCpu%%install}
cmake ../omega_h \
  -DCMAKE_INSTALL_PREFIX=$ohCpu \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_MPI=OFF  \
  -DBUILD_TESTING=on  \
  -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_PREFIX=$kkCpu/lib64/cmake
make VERBOSE=1 -j8 install
ctest
#cabana
cd $root
#git clone https://github.com/ECP-copa/Cabana.git cabana
[ -d $cabCpu ] && rm -rf ${cabCpu%%install}
mkdir -p $cabCpu
cd ${cabCpu%%install}
cmake ../cabana \
  -DCMAKE_BUILD_TYPE="Debug" \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCabana_ENABLE_MPI=OFF \
  -DCabana_ENABLE_CAJITA=OFF \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX=$cabCpu
make -j 24 install

```

Make the script executable:

```
chmod +x buildAllCpu.sh
```

Source the environment script from this work directory:

```
source envCpu.sh
```

Run the build script:

```
./buildAllCpu.sh
```

## build meshFields

The following assumes that the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies.

```
cd $root
git clone git@github.com:SCOREC/meshFields
cmake -S meshFields -B build-meshFields-cuda
cmake --build build-meshFields-cuda 
```

## rebuild meshFields

The following assumes that (1) the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies and (2) that `meshFields` was previously built.

Use the following command to rebuild `meshFields` after making changes to its source code.  This command should be run from the `$root` directory.

```
cmake --build build-meshFields-cuda 
```

## run meshFields tests

The following assumes that (1) the environment is already setup (see above) and the
`root` directory is the same directory used to build the dependencies and (2) that
`meshFields` was previously built.

```
cmake --build build-meshFields-cuda --target test
```

Alternatively, you can cd into the build directory and run `ctest` directly:

```
cd build-meshFields-cuda 
ctest
```

## build and run meshFields tests on CPU for coverage

The following assumes that (1) the enviornment is set up for a CPU build
(see above) and the `root` directory is the same directory used to build the 
dependencies and (2) you have already cloned the git repository from creating
a GPU build and building meshFields

The `-D` options for the install directories can be changed to reflect the 
path in your instalation. It will likely be the same except in the case of
`LCOV_SYSTEM_EXCLUDE_PATHS` which should be updated to reflect your file 
structure as it is hard coded.

```
cd $root
cmake -S meshFields -B build-meshFields-cpu -D meshFields_ENABLE_COVERAGE_BUILD=ON -D CMAKE_BUILD_TYPE=Debug -D CABANA_INSTALL_DIR="\${PROJECT_SOURCE_DIR}/../build-cabana-cranium-cuda/*" -D KOKKOS_INSTALL_DIR="\${PROJECT_SOURCE_DIR}/../build-kokkos-cranium-cuda/*" -D OMEGAH_INSTALL_DIR="\${PROJECT_SOURCE_DIR}/../build-omegah-cranium-cuda/*" -D LCOV_SYSTEM_EXCLUDE_PATHS="/opt/scorec/spack/v0154_2/install/linux-rhel7-x86_64/gcc-6.5.0/gcc-10.1.0-tf5jjaditemasrbsl7tz6pnqa6duqwkg/include/c++/*\;/usr/local/cuda-11.4/include/*"


cd build-meshFields-cpu
make
make coverage
```

In your build directory, `build-meshFields-cpu`, there will be a folder called `coverage`. If your
install is not already on your local machine, downloading this folder to your local machine will 
allow the file inside, `index.html`, to be viewed in your browser. This is a visual representation
of testing coverage.


## Timing Data

For timing, we'll use Kokkos SimpleKernelTimer along with some of the other tools listed here:
https://github.com/kokkos/kokkos-tools/wiki

First, clone the repository
```
git clone git@github.com:kokkos/kokkos-tools.git
```
Then, follow the instructions here to compile:
https://github.com/kokkos/kokkos-tools/wiki/SimpleKernelTimer

To run your program with timing data, run the following from your `root` directory
```
export KOKKOS_PROFILE_LIBRARY={PATH_TO_TOOL_DIRECTORY}/kp_kernel_timer.so
/path/to/binary/to/time
```
This will produce normal program output, plus the following line
```
KokkosP: Kernel timing written to {timing output file path}
```

To view the timing data, run the following command
```
{PATH_TO_TOOL_DIRECTORY}/kp_reader {timing output file path}
```
