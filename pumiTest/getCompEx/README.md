## setup environment

```
module use /opt/scorec/spack/rhel9/v0201_4/lmod/linux-rhel9-x86_64/Core/
module load gcc/12.3.0-iil3lno mpich/4.1.1-xpoyz4t cmake
```

## build pumi

```
git clone git@github.com:SCOREC/core
bdir=$PWD/buildPumi
cmake -S core -B $bdir \
-DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_Fortran_COMPILER=gfortran \
-DSCOREC_CXX_OPTIMIZE=off \
-DCMAKE_INSTALL_PREFIX=$bdir/install
cmake --build $bdir --target install -j 10
```

## build example

```
cmake -S getCompEx -B buildCompEx -DSCOREC_PREFIX=$bdir/install -DCMAKE_CXX_COMPILER=mpicxx
cmake --build buildCompEx
./buildCompEx/getCompEx
```
