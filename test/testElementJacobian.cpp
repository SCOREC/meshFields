#include "MeshField_TestHarness.hpp"
#include <string>
#include <type_traits>
#include <vector>

using MeshField::LO;
using MeshField::Real;

// a mesh of one simplex whose nodes are vertices 0 to dim
template <size_t dim> struct SimplexToVertexField {
  static constexpr MeshField::Mesh_Topology Topology =
      dim == 1   ? MeshField::Edge
      : dim == 2 ? MeshField::Triangle
                 : MeshField::Tetrahedron;

  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {Topology};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(LO node, LO comp, LO ent, MeshField::Mesh_Topology topo) const {
    assert(topo == Topology);
    assert(ent == 0);
    return {0, comp, node, MeshField::Vertex};
  }
};

template <size_t dim> struct JacobianCase {
  std::string name;
  /** vertex coordinates, [vertex * dim + direction] */
  std::vector<Real> coords;
  /** J[i][j] = dx_j/dxi_i, [i * dim + j] */
  std::vector<Real> jacobian;
  Real determinant;
};

/*
 * Compute the Jacobian, and its determinant, at the centroid of a single
 * linear simplex
 */
template <size_t dim, template <typename...> typename Controller>
void testJacobian(JacobianCase<dim> const &tc) {
  const std::string name =
      "testJacobian(" + std::to_string(dim) + "d, " + tc.name + ")";
  constexpr LO numVtx = dim + 1;
  const MeshField::MeshInfo meshInfo{.numVtx = numVtx, .dim = dim};
  auto coordFieldWithCtrlr =
      MeshField::CreateCoordinateField<ExecutionSpace, Controller, dim>(
          meshInfo);
  auto field = coordFieldWithCtrlr.field;
  Kokkos::View<Real *, Kokkos::HostSpace> coords_h("coords_h",
                                                   tc.coords.size());
  for (size_t i = 0; i < tc.coords.size(); ++i)
    coords_h(i) = tc.coords[i];
  const auto coords =
      Kokkos::create_mirror_view_and_copy(ExecutionSpace(), coords_h);
  Kokkos::parallel_for(
      "setCoords", numVtx, KOKKOS_LAMBDA(const int vtx) {
        for (size_t d = 0; d < dim; ++d)
          field(vtx, 0, d, MeshField::Vertex) = coords(vtx * dim + d);
      });

  using Shape = std::conditional_t<
      dim == 1, MeshField::LinearEdgeShape,
      std::conditional_t<dim == 2, MeshField::LinearTriangleShape,
                         MeshField::LinearTetrahedronShape>>;
  MeshField::FieldElement elm(1, field, Shape(), SimplexToVertexField<dim>());

  Kokkos::View<Real **> lc("localCoords", 1, dim);
  Kokkos::deep_copy(lc, 1.0 / (dim + 1));
  const auto J = MeshField::getJacobians(elm, lc, 1);
  const auto J_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
  if (J_h.extent(0) != 1 || J_h.extent(1) != dim || J_h.extent(2) != dim) {
    report(name + " extents", 1);
    return;
  }
  LO errors = 0;
  for (size_t i = 0; i < dim; ++i)
    for (size_t j = 0; j < dim; ++j)
      if (!(std::fabs(J_h(0, i, j) - tc.jacobian[i * dim + j]) <=
            MeshField::MachinePrecision))
        ++errors;
  // getJacobianDeterminants does not support 1d yet
  if constexpr (dim > 1) {
    const auto det = MeshField::getJacobianDeterminants(elm, J);
    const auto det_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), det);
    if (det_h.extent(0) != 1 ||
        !(std::fabs(det_h(0) - tc.determinant) <= MeshField::MachinePrecision))
      ++errors;
  }
  report(name, errors);
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  t.forEachController([&]<template <typename...> typename Controller>() {
    testJacobian<1, Controller>({"edge", {-1, 1}, {1}, 1});
    testJacobian<2, Controller>(
        {"right triangle", {0, 0, 1, 0, 0, 1}, {1, 0, 0, 1}, 1});
    testJacobian<2, Controller>(
        {"skewed triangle", {0, 0, 5, 1, 3, 4}, {5, 1, 3, 4}, 17});
    testJacobian<3, Controller>({"right tetrahedron",
                                 {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                                 {1, 0, 0, 0, 1, 0, 0, 0, 1},
                                 1});
    // the vertices are ordered left handed, so the determinant is negative
    testJacobian<3, Controller>({"skewed tetrahedron",
                                 {0, 0, 0, 5, 1, 2, 3, 4, 6, 10, 8, 9},
                                 {5, 1, 2, 3, 4, 6, 10, 8, 9},
                                 -59});
  });
  return t.exitCode();
}
