cmake_minimum_required(VERSION 3.13.0)

project(meshfields VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MeshFields_USE_Cabana "Build with the Cabana storage backend" OFF)

find_package(Omega_h REQUIRED)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "")

if(MeshFields_USE_Cabana)
  find_package(Cabana 0.7.0 REQUIRED)
endif()

set(MESHFIELD_HEADERS
  src/MeshField_Utility.hpp
  src/MeshField_Macros.hpp
  src/KokkosController.hpp
  src/MeshField_Element.hpp
  src/MeshField_Integrate.hpp
  src/MeshField_Shape.hpp
  src/MeshField_Fail.hpp
  src/MeshField_For.hpp
  src/MeshField_Reduce.hpp
  src/MeshField_Scan.hpp
  src/MeshField_Field.hpp
  src/MeshField.hpp
  "${CMAKE_CURRENT_BINARY_DIR}/MeshField_Config.hpp"
)
if(MeshFields_USE_Cabana)
  list(APPEND MESHFIELD_HEADERS 
    src/CabanaController.hpp
    src/MeshField_SimdFor.hpp)
endif()

set(MESHFIELD_SOURCES
  src/MeshField_Fail.cpp
)

add_library(meshfields ${MESHFIELD_SOURCES})

target_compile_features(meshfields INTERFACE cxx_std_17)
target_include_directories(meshfields
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>" # for MeshField_Config.hpp
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(meshfields INTERFACE Omega_h::omega_h)
if(Kokkos_ENABLE_CUDA)
  target_compile_options(meshfields INTERFACE "--expt-relaxed-constexpr")
endif()

#enable/disable exceptions
option(MeshFields_USE_EXCEPTIONS "Enable throwing exceptions" ON)
message(STATUS "Exceptions: ${MeshFields_USE_EXCEPTIONS}")

#pass cmake options to the source
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/MeshField_Config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/MeshField_Config.hpp")

if(MeshFields_USE_Cabana)
  target_link_libraries(meshfields INTERFACE Cabana::Core)
  target_compile_definitions(meshfields INTERFACE MESHFIELDS_ENABLE_CABANA)
endif()

#support ctags
option(MeshFields_USE_CTAGS "Generate Ctags" OFF)
if(MeshFields_USE_CTAGS)
  #from: https://stackoverflow.com/a/9842046
  set_source_files_properties(tags PROPERTIES GENERATED true)
  add_custom_target(tags
    COMMAND ctags -R --c++-kinds=+p --fields=+iaS --extra=+q
            --exclude=docs --exclude=cmake --exclude=.git
            --exclude=.github .
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_dependencies(meshfields tags)
endif()

# Set options for doxygen documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
)
add_custom_target(doc
  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen" VERBATIM
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile_internal.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_internal @ONLY
)
add_custom_target(docInternal
  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_internal
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating Internal API documentation with Doxygen" VERBATIM
)
endif()

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

#check for valgrind
find_program(VALGRIND_CMD valgrind DOC "Location of the valgrind program")

#tests
function(test_func_impl TEST_NAME)
  set(TEST_STR ${ARGN})
  # need to run as a cmake script to capture assert and other 'system failures'
  # https://cmake.org/cmake/help/latest/prop_test/WILL_FAIL.html#prop_test:WILL_FAIL
  add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_COMMAND} -E env ${TEST_STR})
endfunction(test_func_impl)

function(test_func TEST_NAME)
  test_func_impl(${TEST_NAME} ${ARGN})
  if(TEST ${TEST_NAME})
    set_property(TEST ${TEST_NAME} PROPERTY LABELS "base")
  endif()
endfunction(test_func)

# Unlike test_func, will_fail_test_func assumes the command for the test will fail
function(will_fail_test_func TEST_NAME)
  test_func_impl(${TEST_NAME} ${ARGN})
  set_property(TEST ${TEST_NAME} PROPERTY WILL_FAIL TRUE)
  if(TEST ${TEST_NAME})
    set_property(TEST ${TEST_NAME} PROPERTY LABELS "base")
  endif()
endfunction()

function(will_fail_valgrind_test_func TEST_NAME)
  if(VALGRIND_CMD)
    test_func_impl(${TEST_NAME} ${VALGRIND_CMD} ${ARGN})
    set_property(TEST ${TEST_NAME} PROPERTY
      FAIL_REGULAR_EXPRESSION "Invalid read;Invalid write"
    )
    set_property(TEST ${TEST_NAME} PROPERTY WILL_FAIL TRUE)
    if(TEST ${TEST_NAME})
      set_property(TEST ${TEST_NAME} PROPERTY LABELS "base")
    endif()
  endif()
endfunction()

function(meshfields_add_exe EXE_NAME EXE_SRC)
  add_executable(${EXE_NAME} ${EXE_SRC})
  target_link_libraries(${EXE_NAME} PRIVATE meshfields)
endfunction()

# Creating minimal reproduction of error
meshfields_add_exe(KokkosTests test/testKokkos.cpp)
meshfields_add_exe(SerializationTests test/testSerialize.cpp)
meshfields_add_exe(ElementTests test/testElement.cpp)
meshfields_add_exe(ElementJacobian1d test/testElementJacobian1d.cpp)
meshfields_add_exe(ElementJacobian2d test/testElementJacobian2d.cpp)
meshfields_add_exe(OmegahElementTests test/testOmegahElement.cpp)
meshfields_add_exe(OmegahCoordFieldTest test/testOmegahCoordField.cpp)
meshfields_add_exe(ExceptionTest test/testExceptions.cpp)

if(MeshFields_USE_Cabana)
  meshfields_add_exe(CabanaTests test/testCabana.cpp)
  test_func(CabanaTests ./CabanaTests)
endif()

test_func(KokkosTests ./KokkosTests)
test_func(SerializationTests ./SerializationTests)
test_func(ElementTests ./ElementTests)
test_func(ElementJacobian1d ./ElementJacobian1d)
test_func(ElementJacobian2d ./ElementJacobian2d)
test_func(OmegahElementTests ./OmegahElementTests)
test_func(OmegahCoordFieldTest ./OmegahCoordFieldTest)
if(MeshFields_USE_EXCEPTIONS)
  # exception caught - no error
  test_func(ExceptionTest ./ExceptionTest)
else()
  will_fail_test_func(ExceptionTest ./ExceptionTest)
endif()

#Code Coverage set up -------------------------------------------------------

option(meshfields_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(meshfields_ENABLE_COVERAGE_BUILD)

  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
  include(CodeCoverage)

  append_coverage_compiler_flags()

  set(LCOV_SYSTEM_EXCLUDE_PATHS "" CACHE PATH "Set path to system libraries, c++ compiler and cuda-11.4 by default, to be excluded from lcov analysis")

  if (NOT EXISTS ${Kokkos_DIR})
    message(FATAL_ERROR " Kokkos_DIR was not set or the path does not exist")
  endif()

  if (NOT EXISTS ${Omega_h_DIR})
    message(FATAL_ERROR " Omega_h_DIR was not set or the path does not exist")
  endif()

  set(LCOV_SRC_EXCLUDE_PATHS
      "${Kokkos_DIR}/*"
      "${Omega_h_DIR}/*"
      "${LCOV_SYSTEM_EXCLUDE_PATHS}"
  )

  if (MeshFields_USE_Cabana AND NOT EXISTS ${Cabana_DIR})
    message(FATAL_ERROR " Cabana_DIR was not set or the path does not exist")
  endif()

  if (MeshFields_USE_Cabana)
    list(APPEND LCOV_SRC_EXCLUDE_PATHS "${Cabana_DIR}")
  endif()

  setup_target_for_coverage_lcov(
    NAME coverage
    EXECUTABLE ctest -C ${ROOT_DIR}/CTestTestfile.cmake
    EXCLUDE ${LCOV_SRC_EXCLUDE_PATHS}
  )

endif()

#Code Coverage set up end ------------------------------------------------


## export the library
set_target_properties(meshfields PROPERTIES PUBLIC_HEADER "${MESHFIELD_HEADERS}")
install(
  TARGETS meshfields
  EXPORT meshfields-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/meshfields-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshfields
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/meshfields-config-version.cmake"
  COMPATIBILITY AnyNewerVersion)

install(FILES
  "${PROJECT_BINARY_DIR}/meshfields-config.cmake"
  "${PROJECT_BINARY_DIR}/meshfields-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshfields)

install(
  EXPORT meshfields-targets
  NAMESPACE meshfields::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshfields)
