## build and run

export CMAKE_PREFIX_PATH=/path/to/kokkos/install/lib64/cmake/Kokkos/:$CMAKE_PREFIX_PATH
cmake -B build -S integrationSandbox
cmake --build build/
ctest --test-dir build
