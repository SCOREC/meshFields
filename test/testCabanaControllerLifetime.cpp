// Regression test for the CabanaController lifetime bug uncovered while
// investigating https://github.com/SCOREC/meshFields/issues/37
//
// CabanaController::makeSlice() returns a Cabana::Slice that is a
// *non-owning* view into the CabanaController's own `aosoa` member
// (unlike KokkosController, whose Kokkos::View slices are refcounted and
// safe to outlive the controller). CreateCoordinateField/CreateLagrangeField
// build the Controller as a local variable; the ShapeField they hand back
// used to be kept alive-by-accident via ShapeField::meshField, which stored
// a copy of the Controller. Once that unused-looking member was removed,
// any ShapeField built from a CabanaController and returned by value from
// a factory held a dangling slice into freed memory.
//
// The fix: CabanaController::aosoa is now a std::shared_ptr, and
// CreateCoordinateField/CreateLagrangeField return a FieldWithController
// {ctrlr, field} pair instead of a bare ShapeField, making it the caller's
// explicit responsibility to keep `ctrlr` alive for as long as `field` is
// used. This test builds a CoordinateField from a CabanaController inside a
// helper function, keeps the returned FieldWithController (both members)
// alive in main(), and writes to the field - this must NOT reproduce the
// heap-use-after-free ASan caught in testOmegahTri.cpp / testOmegahTet.cpp.
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
  // returns FieldWithController{ctrlr, field}; the CabanaController built
  // inside CreateCoordinateField is only kept alive because it is copied
  // into the returned struct's `ctrlr` member.
  return MeshField::CreateCoordinateField<ExecutionSpace,
                                          MeshField::CabanaController, 2>(
      meshInfo);
}
#endif

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
#ifdef MESHFIELDS_ENABLE_CABANA
    // fieldWithCtrlr.ctrlr must stay alive for as long as
    // fieldWithCtrlr.field is used.
    auto fieldWithCtrlr = makeField();
    auto coordField = fieldWithCtrlr.field;

    Kokkos::parallel_for(
        "writeCoords", 4, KOKKOS_LAMBDA(const int &i) {
          coordField(i, 0, 0, MeshField::Vertex) = (double)i;
          coordField(i, 0, 1, MeshField::Vertex) = (double)i * 2;
        });
    Kokkos::fence();

    MeshField::LO numErrors = 0;
    Kokkos::parallel_reduce(
        "checkCoords", 4,
        KOKKOS_LAMBDA(const int &i, MeshField::LO &lerrors) {
          if (coordField(i, 0, 0, MeshField::Vertex) != (double)i)
            lerrors += 1;
          if (coordField(i, 0, 1, MeshField::Vertex) != (double)i * 2)
            lerrors += 1;
        },
        numErrors);
    if (numErrors > 0) {
      std::cout << "FAILED: read back incorrect coordinate values\n";
      Kokkos::finalize();
      return 1;
    }
    std::cout << "OK: no use-after-free, values read back correctly\n";
#else
    std::cout << "MESHFIELDS_ENABLE_CABANA not set; skipping\n";
#endif
  }
  Kokkos::finalize();
  return 0;
}
