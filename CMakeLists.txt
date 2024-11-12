cmake_minimum_required(VERSION 3.13.0)

project(meshFields VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Omega_h REQUIRED)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "")

find_package(Cabana 0.7.0 REQUIRED)

find_package(KokkosKernels REQUIRED)

set(MESHFIELD_HEADERS
  src/MeshField_Utility.hpp
  src/MeshField_Macros.hpp
  src/KokkosController.hpp
  src/CabanaController.hpp
  src/MeshField.hpp
)

#create header only library
add_library(meshFields INTERFACE)
target_compile_features(meshFields INTERFACE cxx_std_17)
target_include_directories(meshFields
  INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(meshFields INTERFACE Omega_h::omega_h)
target_link_libraries(meshFields INTERFACE Cabana::Core)
target_link_libraries(meshFields INTERFACE Kokkos::kokkoskernels)
target_compile_definitions(meshFields INTERFACE ENABLE_CABANA)

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
  add_dependencies(meshFields tags)
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


# Creating minimal reproduction of error
add_executable(Main test/testMain.cpp)
target_link_libraries(Main PRIVATE meshFields)

add_executable(KokkosTests test/testKokkos.cpp)
target_link_libraries(KokkosTests PRIVATE meshFields)

add_executable(CabanaTests test/testCabana.cpp)
target_link_libraries(CabanaTests PRIVATE meshFields)

add_executable(SerializationTests test/testSerialize.cpp)
target_link_libraries(SerializationTests PRIVATE meshFields)

add_executable(QRTests test/testQR.cpp)
target_link_libraries(QRTests PRIVATE meshFields)

test_func(MainTests ./Main)
test_func(KokkosTests ./KokkosTests)
test_func(CabanaTests ./CabanaTests)
test_func(SerializationTests ./SerializationTests)
if(Kokkos_ENABLE_CUDA) #haven't tested with HIP or SYCL
  test_func(QRTestsCUDA ./QRTests)
else()
  #with the serial backend the QR tests fail when run under valgrind
  #see https://github.com/kokkos/kokkos-kernels/issues/2328
  will_fail_valgrind_test_func(QRTestsSerialValgrind ./QRTests)
endif()

#Code Coverage set up -------------------------------------------------------

option(meshFields_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(meshFields_ENABLE_COVERAGE_BUILD)

  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
  include(CodeCoverage)

  append_coverage_compiler_flags()

  set(LCOV_SYSTEM_EXCLUDE_PATHS "" CACHE PATH "Set path to system libraries, c++ compiler and cuda-11.4 by default, to be excluded from lcov analysis")
  
  if (NOT EXISTS ${Cabana_DIR})
    message(FATAL_ERROR " Cabana_DIR was not set or the path does not exist")
  endif()

  if (NOT EXISTS ${Kokkos_DIR})
    message(FATAL_ERROR " Kokkos_DIR was not set or the path does not exist")
  endif()

  if (NOT EXISTS ${KokkosKernels_DIR})
    message(FATAL_ERROR " KokkosKernels_DIR was not set or the path does not exist")
  endif()

  if (NOT EXISTS ${Omega_h_DIR})
    message(FATAL_ERROR " Omega_h_DIR was not set or the path does not exist")
  endif()

  setup_target_for_coverage_lcov(
    NAME coverage                    
    EXECUTABLE ctest -C ${ROOT_DIR}/CTestTestfile.cmake
    EXCLUDE 
      "${CMAKE_CURRENT_LIST_DIR}/docs/*"
      "${CMAKE_CURRENT_LIST_DIR}/.github/*"
      "${CMAKE_CURRENT_BINARY_DIR}/nvcc_internal_extended_lambda_implementation"
      "/tmp/*"
      "${Cabana_DIR}/*"
      "${Kokkos_DIR}/*"
      "${KokkosKernels_DIR}/*"
      "${Omega_h_DIR}/*"
      "${LCOV_SYSTEM_EXCLUDE_PATHS}"  
  )

endif()

#Code Coverage set up end ------------------------------------------------


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
