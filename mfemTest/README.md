Run using g++ IntegratorMfemTest.cpp  -I<path-to-mfem-build> -L<path-to-mfem-build>  -l mfem
in order to run, use -m to specify the mesh file, -o to specify order, -s to specify size, -n to specify number of elements in each direction, and -t to specify type of element with 1 for tetrahedron and 0 for triangle.
mfem version: https://github.com/mfem/mfem/tree/2caa75e35a54df93d19f23655170254556dfc081
For the build:
mkdir <mfem-build-dir> ; cd <mfem-build-dir>
cmake <mfem-source-dir>
make -j 4
