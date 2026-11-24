#include "MeshField_TestHarness.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>
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
  REQUIRE(J_h.extent(0) == 1);
  REQUIRE(J_h.extent(1) == dim);
  REQUIRE(J_h.extent(2) == dim);
  using Catch::Matchers::WithinAbs;
  for (size_t i = 0; i < dim; ++i)
    for (size_t j = 0; j < dim; ++j) {
      CAPTURE(i, j);
      CHECK_THAT(J_h(0, i, j), WithinAbs(tc.jacobian[i * dim + j],
                                         MeshField::MachinePrecision));
    }
  if constexpr (dim > 1) {
    const auto det = MeshField::getJacobianDeterminants(elm, J);
    const auto det_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), det);
    REQUIRE(det_h.extent(0) == 1);
    CHECK_THAT(det_h(0),
               WithinAbs(tc.determinant, MeshField::MachinePrecision));
  } else {
    const auto values = MeshField::evaluate(elm, lc);
    const auto values_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    REQUIRE(values_h.extent(0) == 1);
    CHECK_THAT(values_h(0, 0),
               WithinAbs(0.25 * tc.coords[0] + 0.75 * tc.coords[1],
                         MeshField::MachinePrecision));
  }
}

TEMPLATE_LIST_TEST_CASE("Jacobian of a linear simplex", "[jacobian]",
                        ControllerCases) {
  SECTION("edge") {
    testJacobian<1, TestType::template Controller>({{-1, 1}, {1}, 1});
  }
  SECTION("right triangle") {
    testJacobian<2, TestType::template Controller>(
        {{0, 0, 1, 0, 0, 1}, {1, 0, 0, 1}, 1});
  }
  SECTION("skewed triangle") {
    testJacobian<2, TestType::template Controller>(
        {{0, 0, 5, 1, 3, 4}, {5, 1, 3, 4}, 17});
  }
  SECTION("right tetrahedron") {
    testJacobian<3, TestType::template Controller>(
        {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 1});
  }
  // the vertices are ordered left handed, so the determinant is negative
  SECTION("skewed tetrahedron") {
    testJacobian<3, TestType::template Controller>(
        {{0, 0, 0, 5, 1, 2, 3, 4, 6, 10, 8, 9},
         {5, 1, 2, 3, 4, 6, 10, 8, 9},
         -59});
  }
}
