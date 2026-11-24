cmake_minimum_required(VERSION 3.13.0)

project(meshfields VERSION 1.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckGit.cmake)

option(MeshFields_USE_Cabana "Build with the Cabana storage backend" OFF)

set(MeshFields_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(MeshFields_IS_TOP_LEVEL ON)
endif()
option(MeshFields_ALWAYS_CHECK_GIT
  "Check the git hash on every build instead of only at configure time"
  ${MeshFields_IS_TOP_LEVEL})

find_package(Kokkos REQUIRED)
find_package(Omega_h REQUIRED)
find_package(MPI REQUIRED)

# Verify Omega_h was built with Kokkos enabled
if(NOT Omega_h_USE_Kokkos)
  message(FATAL_ERROR "Omega_h must be built with Kokkos enabled (Omega_h_USE_Kokkos=ON)")
endif()

if(MeshFields_USE_Cabana)
  find_package(Cabana 0.7.0 REQUIRED)
endif()

set(MESHFIELD_HEADERS
  src/MeshField_Defines.hpp
  src/MeshField_Utility.hpp
  src/MeshField_Macros.hpp
  src/KokkosController.hpp
  src/MeshField_Element.hpp
  src/MeshField_LinearAlgebra.hpp
  src/MeshField_MappedElement.hpp
  src/MeshField_Integrate.hpp
  src/MeshField_Shape.hpp
  src/MeshField_ShapeField.hpp
  src/MeshField_Fail.hpp
  src/MeshField_For.hpp
  src/MeshField_Reduce.hpp
  src/MeshField_Scan.hpp
  src/MeshField_Field.hpp
  src/MeshField.hpp
  src/MeshField_SPR_ErrorEstimator.hpp
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
# create a target for in-tree builds that matches the installed target name
add_library(meshfields::meshfields ALIAS meshfields)

target_compile_features(meshfields INTERFACE cxx_std_20)
target_include_directories(meshfields
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>" # for MeshField_Config.hpp
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(meshfields INTERFACE Kokkos::kokkos)
target_link_libraries(meshfields INTERFACE Omega_h::omega_h)

#enable/disable exceptions
option(MeshFields_USE_EXCEPTIONS "Enable throwing exceptions" ON)
message(STATUS "Exceptions: ${MeshFields_USE_EXCEPTIONS}")

# Compute a single integer version for C++ preprocessor use
# MESHFIELDS_VERSION = MAJOR*10000 + MINOR*100 + PATCH
# e.g. 1.0.0 -> 10000
math(EXPR MeshFields_VERSION
     "${PROJECT_VERSION_MAJOR} * 10000 + ${PROJECT_VERSION_MINOR} * 100 + ${PROJECT_VERSION_PATCH}")

#pass cmake options to the source
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/MeshField_Config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/MeshField_Config.hpp")


CheckGitSetup()
target_sources(meshfields PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/MeshField_GetHash.cpp)
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

# Tests use Catch2; retain the existing full-suite option for Spack and users.
include(CTest)
option(MeshFields_IS_TESTING "Build the full test suite" OFF)
option(MeshFields_BUILD_BENCHMARKS "Build the controller benchmark demo" OFF)
option(MeshFields_USE_VALGRIND "Run tests under valgrind" OFF)
if(MeshFields_USE_VALGRIND)
  find_program(VALGRIND_CMD valgrind)
  if(NOT VALGRIND_CMD)
    message(FATAL_ERROR "MeshFields_USE_VALGRIND requires valgrind")
  endif()
endif()

if(BUILD_TESTING OR MeshFields_BUILD_BENCHMARKS)
  find_package(Catch2 3 REQUIRED)
  include(Catch)
  add_library(meshfields_test_main STATIC test/catchMain.cpp)
  target_include_directories(meshfields_test_main PUBLIC test)
  target_link_libraries(meshfields_test_main PUBLIC meshfields Catch2::Catch2)

  function(meshfields_add_test NAME SRC)
    set(label "meshfields::base")
    if(ARGN)
      set(label "${ARGN}")
    endif()
    add_executable(${NAME} ${SRC})
    target_link_libraries(${NAME} PRIVATE meshfields_test_main)
    if(MeshFields_USE_VALGRIND)
      add_test(NAME ${NAME}
               COMMAND ${VALGRIND_CMD} --error-exitcode=1 $<TARGET_FILE:${NAME}>)
      set_tests_properties(${NAME} PROPERTIES LABELS "${label}")
    else()
      # Defer discovery so compiling CUDA tests does not require a GPU.
      catch_discover_tests(${NAME} TEST_PREFIX "${NAME}: "
                           DISCOVERY_MODE PRE_TEST
                           PROPERTIES LABELS "${label}")
    endif()
  endfunction()

  if(BUILD_TESTING)
    meshfields_add_test(ControllerTests test/testController.cpp meshfields::smoke)
    meshfields_add_test(Reproduction test/testReproduction.cpp meshfields::smoke)
    if(MeshFields_IS_TESTING)
      meshfields_add_test(SerializationTests test/testSerialize.cpp)
      meshfields_add_test(ControllerLifetimeTest test/testControllerLifetime.cpp)
      meshfields_add_test(ElementJacobian test/testElementJacobian.cpp)
      meshfields_add_test(Integrator test/testIntegrator.cpp)
      meshfields_add_test(LinearAlgebra test/testLinearAlgebra.cpp)
      meshfields_add_test(Interpolate test/testInterpolate.cpp)
      meshfields_add_test(OmegahMappedElement test/testOmegahMappedElement.cpp)
      if(MeshFields_USE_EXCEPTIONS)
        meshfields_add_test(ExceptionTest test/testExceptions.cpp)
      else()
        add_executable(ExceptionTest test/testExceptions.cpp)
        target_link_libraries(ExceptionTest PRIVATE meshfields_test_main)
        foreach(overload IN ITEMS variadic string)
          # cmake -E env converts the child's abort into an ordinary nonzero
          # exit status, which CTest can check with WILL_FAIL.
          add_test(NAME "ExceptionTest_${overload}_aborts"
                   COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:ExceptionTest>
                           "[${overload}]")
          set_tests_properties("ExceptionTest_${overload}_aborts"
                               PROPERTIES WILL_FAIL TRUE LABELS "meshfields::base")
        endforeach()
      endif()
      meshfields_add_test(OmegahSprAdapt test/testSprThwaitesAdapt.cpp)
      target_compile_definitions(OmegahSprAdapt PRIVATE
        MESHFIELDS_TEST_MESH="${CMAKE_CURRENT_SOURCE_DIR}/meshes/thwaites_basal_effectiveStrain.osh")
      if(Omega_h_USE_MPI)
        add_test(NAME OmegahSprAdapt_p2
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:OmegahSprAdapt>
                         ${MPIEXEC_POSTFLAGS} "[adapt]")
        set_tests_properties(OmegahSprAdapt_p2 PROPERTIES
                             LABELS "meshfields::base" PROCESSORS 2)
      endif()
    endif()
  endif()
  if(MeshFields_USE_Cabana AND (MeshFields_IS_TESTING OR MeshFields_BUILD_BENCHMARKS))
    add_executable(IntegratorPerformance test/testIntegratorPerformance.cpp)
    target_link_libraries(IntegratorPerformance PRIVATE meshfields)
  endif()
  if(MeshFields_BUILD_BENCHMARKS)
    add_executable(ControllerBenchmark test/benchmarkController.cpp)
    target_link_libraries(ControllerBenchmark PRIVATE meshfields_test_main)
    # Run explicitly; benchmarks are not correctness tests or timing gates.
  endif()
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
