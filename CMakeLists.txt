cmake_minimum_required(VERSION 3.13.0)

project(meshFields VERSION 0.1.0 LANGUAGES CXX)

find_package(Omega_h REQUIRED)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "")

find_package(Cabana REQUIRED)

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

add_executable(SliceWrapper test/SliceWrapper.cpp)
target_include_directories(SliceWrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src) #hack - this should be a header-only library
target_link_libraries(SliceWrapper Omega_h::omega_h)
target_link_libraries(SliceWrapper Cabana::cabanacore)
target_compile_definitions(SliceWrapper PUBLIC ENABLE_CABANA)

add_test(sliceWrapper10 ./SliceWrapper 10)
add_test(sliceWrapper32 ./SliceWrapper 32)
add_test(sliceWrapper33 ./SliceWrapper 33)
add_test(sliceWrapper50 ./SliceWrapper 50)
