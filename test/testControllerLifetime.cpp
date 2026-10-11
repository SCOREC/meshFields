// Regression test for https://github.com/SCOREC/meshFields/issues/37
//
// CreateLagrangeField builds its Controller as a local variable.  A Cabana
// slice is a non-owning view into its controller's storage (a Kokkos::View
// slice is reference counted), so once ShapeField stopped storing a copy of
// the controller, a field returned from a function could point into freed
// memory.  The fix: the CabanaController storage is a std::shared_ptr, and
// CreateLagrangeField returns a FieldWithController {ctrlr, field} that the
// caller must keep alive for as long as the field is used.
//
// This test creates a quadratic field inside a function, as the bug required,
// then writes to it and evaluates it.  Under ASan the bug shows as a
// heap-use-after-free.
#include "MeshField_TestHarness.hpp"

using MeshField::LO;
using MeshField::Real;

// the quadratic triangle with vertices 0, 1, 2 and edges 0, 1, 2
struct QuadraticTriangleToField {
  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Triangle};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(LO node, LO comp, LO ent, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Triangle);
    assert(ent == 0);
    const MeshField::Mesh_Topology holderTopo =
        node < 3 ? MeshField::Vertex : MeshField::Edge;
    return {0, comp, node < 3 ? node : node - 3, holderTopo};
  }
};

// returns the field and its controller, which must be kept alive together
template <template <typename...> typename Controller> auto createField() {
  const MeshField::MeshInfo meshInfo{
      .numVtx = 3, .numEdge = 3, .numTri = 1, .dim = 2};
  return MeshField::CreateLagrangeField<ExecutionSpace, Controller, Real, 2, 2,
                                        1>(meshInfo);
}

template <template <typename...> typename Controller> void testLifetime() {
  auto fieldWithCtrlr = createField<Controller>();
  auto field = fieldWithCtrlr.field;
  // f(x, y) = 2x + y on the triangle (0,0), (1,0), (0,1): its vertices and
  // edge midpoints (0.5,0), (0.5,0.5), (0,0.5)
  Kokkos::parallel_for(
      "setDofs", 1, KOKKOS_LAMBDA(const int) {
        field(0, 0, 0, MeshField::Vertex) = 0.0;
        field(1, 0, 0, MeshField::Vertex) = 2.0;
        field(2, 0, 0, MeshField::Vertex) = 1.0;
        field(0, 0, 0, MeshField::Edge) = 1.0;
        field(1, 0, 0, MeshField::Edge) = 1.5;
        field(2, 0, 0, MeshField::Edge) = 0.5;
      });

  MeshField::FieldElement elm(1, field, MeshField::QuadraticTriangleShape(),
                              QuadraticTriangleToField());
  Kokkos::View<Real[1][2]> lc("localCoords");
  Kokkos::deep_copy(lc, 1.0 / 3);
  const auto result = MeshField::evaluate(elm, lc);
  const auto result_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  // f at the centroid (1/3, 1/3)
  report("testLifetime",
         !(std::fabs(result_h(0, 0) - 1.0) <= MeshField::Epsilon));
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  t.forEachController([&]<template <typename...> typename Controller>() {
    testLifetime<Controller>();
  });
  return t.exitCode();
}
