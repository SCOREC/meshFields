// main for the Catch2 tests: Kokkos and Omega_h are initialized for the
// lifetime of the test run.  Omega_h::Library initializes and finalizes
// Kokkos and removes its arguments, so the rest go to Catch2.
#include "MeshField_TestHarness.hpp"
#include <catch2/catch_session.hpp>

namespace {
Omega_h::Library *lib = nullptr;
}

Omega_h::Library &library() { return *lib; }

int main(int argc, char **argv) {
  Omega_h::Library l(&argc, &argv);
  lib = &l;
  return Catch::Session().run(argc, argv);
}
