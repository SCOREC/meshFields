// Minimal repro for the CabanaController lifetime bug uncovered while
// investigating https://github.com/SCOREC/meshFields/issues/37
//
// CabanaController::makeSlice() returns a Cabana::Slice that is a
// *non-owning* view into the CabanaController's own `aosoa` member
// (unlike KokkosController, whose Kokkos::View slices are refcounted and
// safe to outlive the controller). ShapeField used to store a copy of the
// Controller (ShapeField::meshField), which incidentally kept the
// CabanaController - and therefore its aosoa - alive for the lifetime of
// the returned ShapeField. Once that unused-looking member is removed,
// any ShapeField built from a CabanaController and returned by value from
// a factory (CreateCoordinateField/CreateLagrangeField) holds a dangling
// slice into freed memory.
//
// This test builds a CoordinateField from a CabanaController inside a
// helper function so the local CabanaController goes out of scope before
// the field is used, then writes to the field - reproducing the
// heap-use-after-free ASan catches in testOmegahTri.cpp /
// testOmegahTet.cpp.
#include <iostream>
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

#ifdef MESHFIELDS_ENABLE_CABANA
auto makeField() {
  const MeshField::MeshInfo meshInfo{.numVtx = 4, .dim = 2};
  // the CabanaController built inside CreateCoordinateField is a local
  // variable there; once CreateCoordinateField returns, that controller
  // (and its aosoa) is destroyed unless something else keeps it alive.
  return MeshField::CreateCoordinateField<ExecutionSpace,
                                          MeshField::CabanaController, 2>(
      meshInfo);
}
#endif

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
#ifdef MESHFIELDS_ENABLE_CABANA
    auto coordField = makeField(); // Controller is already out of scope here

    Kokkos::parallel_for(
        "writeCoords", 4, KOKKOS_LAMBDA(const int &i) {
          // heap-use-after-free: writes into a slice whose backing AoSoA
          // storage was freed when the CabanaController in makeField()
          // went out of scope.
          coordField(i, 0, 0, MeshField::Vertex) = (double)i;
          coordField(i, 0, 1, MeshField::Vertex) = (double)i * 2;
        });
    Kokkos::fence();
    std::cout << "no crash detected (bug not reproduced)\n";
#else
    std::cout << "MESHFIELDS_ENABLE_CABANA not set; skipping\n";
#endif
  }
  Kokkos::finalize();
  return 0;
}
