Run using g++ IntegratorMfemTest.cpp  -I<path-to-mfem-build> -L<path-to-mfem-build>  -l mfem
in order to run, use -m to specify the mesh file and -o to specify order
mfem version: https://github.com/mfem/mfem/tree/2caa75e35a54df93d19f23655170254556dfc081
For the build:
mkdir <mfem-build-dir> ; cd <mfem-build-dir>
cmake <mfem-source-dir>
make -j 4
