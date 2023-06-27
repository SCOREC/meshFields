cmake_minimum_required(VERSION 3.13.0)

project(meshFields VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Omega_h REQUIRED)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "")

find_package(Cabana REQUIRED)

set(MESHFIELD_HEADERS
  src/MeshField_Utility.hpp
  src/MeshField_Macros.hpp
  src/KokkosController.hpp
  src/CabanaController.hpp
  src/MeshField.hpp
  src/test.hpp
)

#create header only library
add_library(meshFields INTERFACE)
target_compile_features(meshFields INTERFACE cxx_std_17)
target_include_directories(meshFields
  INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(meshFields INTERFACE Omega_h::omega_h)
target_link_libraries(meshFields INTERFACE Cabana::cabanacore)
target_compile_definitions(meshFields INTERFACE ENABLE_CABANA)

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

#tests
#add_executable(SliceWrapper test/testSliceWrapper.cpp)
#target_link_libraries(SliceWrapper PRIVATE meshFields)

#add_executable(MeshField test/testMeshField.cpp)
#target_link_libraries(MeshField PRIVATE meshFields)

add_executable(Main test/testMain.cpp)
target_link_libraries(Main PRIVATE meshFields)

add_executable(KokkosTests test/testKokkos.cpp)
target_link_libraries(KokkosTests PRIVATE meshFields)

add_executable(CabanaTests test/testCabana.cpp)
target_link_libraries(CabanaTests PRIVATE meshFields)

add_executable(LogicTests test/test.cpp)
target_link_libraries(LogicTests PRIVATE meshFields)

add_test(MainTests ./Main)
add_test(KokkosTests ./KokkosTests)
add_test(CabanaTests ./CabanaTests)
add_test(logicTests ./LogicTests)
#add_test(sliceWrapper50 ./SliceWrapper 50)
#add_test(meshField ./MeshField)

## export the library
set_target_properties(meshFields PROPERTIES PUBLIC_HEADER "${MESHFIELD_HEADERS}")
install(
  TARGETS meshFields
  EXPORT meshFields-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/meshfields-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshFields
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/meshfields-config-version.cmake"
  COMPATIBILITY AnyNewerVersion)

install(FILES
  "${PROJECT_BINARY_DIR}/meshfields-config.cmake"
  "${PROJECT_BINARY_DIR}/meshfields-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshFields)

install(
  EXPORT meshFields-targets
  NAMESPACE meshFields::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshFields)
