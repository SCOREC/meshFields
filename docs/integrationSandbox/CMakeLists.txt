# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(integrationSandbox CXX)

# Look for an installed Kokkos
find_package(Kokkos REQUIRED)

# This is the only thing required to set up compiler/linker flags

enable_testing()
add_executable(testCallBack testCallBack.cc)
target_link_libraries(testCallBack Kokkos::kokkos)
add_test(NAME testCallBack COMMAND ./testCallBack)
