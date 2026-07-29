#!/bin/bash -x

(
#cdash output root
d=/users/d_zxg06726/nightlyBuilds/meshFields_build
exec > $d/nightly_log.txt 2>&1

source /etc/profile
# source /users/d_zxg06726/.bash_profile

#setup lmod
export PATH=/usr/share/lmod/lmod/libexec:$PATH

#setup spack modules
unset MODULEPATH

module use /opt/scorec/spack/rhel9/v0222_2/lmod/linux-rhel9-x86_64/Core/
module load gcc/13.2.0-4eahhas
module load mpich/4.2.3-62uy3hd
module load cuda/12.6.2-gqq65nw
module load cmake

cd $d
#remove compilation directories created by previous nightly.cmake runs
[ -d build ] && rm -rf build/

#install kokkos
[ ! -d kokkos ] && git clone --depth 1 --branch 5.1.1 https://github.com/Kokkos/kokkos.git kokkos
kkbdir=$d/build-kokkos
[ -d $kkbdir ] && rm -rf $kkbdir
cmake -S kokkos -B $kkbdir \
  -DCMAKE_CXX_COMPILER=$d/kokkos/bin/nvcc_wrapper \
  -DCMAKE_INSTALL_PREFIX=$kkbdir/install \
  -DKokkos_ARCH_AMPER80=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_IMPL_VIEW_LEGACY=ON \
  -DKokkos_ENABLE_DEBUG=ON
cmake --build $kkbdir -j 8 --target install

[ ! -d cabana ] && git clone git@github.com:ECP-copa/Cabana.git cabana
cd cabana && git pull && cd -
bdir=$d/build-cab
[ -d $bdir ] && rm -rf $bdir 
cmake -S cabana -B $bdir \
  -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
  -DCMAKE_CXX_COMPILER=$d/kokkos/bin/nvcc_wrapper \
  -DKokkos_ROOT=$kkbdir/install \
  -DCabana_ENABLE_MPI=OFF \
  -DCabana_ENABLE_CAJITA=OFF \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_EXAMPLES=OFF \
  -DCabana_ENABLE_Cuda=ON \
  -DCMAKE_INSTALL_PREFIX=$bdir/install
cmake --build $bdir -j 8 --target install

#install omega_h
[ ! -d omega_h ] && git clone https://github.com/SCOREC/omega_h.git
cd omega_h && git pull && cd -
[ -d build-omega_h ] && rm -rf build-omega_h
cmake -S omega_h -B build-omega_h \
  -DCMAKE_INSTALL_PREFIX=build-omega_h/install \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=debug \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=ON \
  -DOmega_h_USE_MPI=OFF \
  -DBUILD_TESTING=ON \
  -DCMAKE_CXX_EXTENSIONS=OFF \
  -DKokkos_PREFIX=$kkbdir/install/lib64/cmake
cmake --build build-omega_h -j 8 --target install

touch $d/startedCoreNightly
#run nightly.cmake script
ctest -V --script $d/nightly.cmake
touch $d/doneCoreNightly
)
