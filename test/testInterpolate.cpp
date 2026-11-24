#include "MeshField_TestFunctions.hpp"
#include "MeshField_TestHarness.hpp"

using MeshField::Real;

// Create a list of two parametric coordinates for evaluation
constexpr size_t numOffNodePts = 2;
template <size_t dim> Kokkos::View<Real **> offNodeCoords(size_t numElems) {
  Kokkos::View<Real **> lc("localCoords", numElems * numOffNodePts, dim);
  Kokkos::parallel_for(
      "setLocalCoords", numElems, KOKKOS_LAMBDA(const int ent) {
        for (size_t d = 0; d < dim; ++d) {
          lc(ent * numOffNodePts, d) = 1.0 / (dim + 1);
          lc(ent * numOffNodePts + 1, d) = 0.1 * (d + 1);
        }
      });
  return lc;
}

/*
 * Write source(ent, xi) = ent. If ownership is working, we should have the
 * neighboring element with the smallest id listed on every dof holder thats
 * is on a shared entity
 */
template <size_t dim, size_t order, template <typename...> typename Controller>
void testOwnership(
    Omega_h::Mesh &mesh,
    MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf) {
  auto fieldWithCtrlr = omf.template CreateLagrangeField<Real, order, 1>();
  auto field = fieldWithCtrlr.field;
  auto elm = omf.CreateLagrangeElement(field);
  MeshField::setDofByInterpolation(
      elm, KOKKOS_LAMBDA(const int ent, Kokkos::Array<Real, dim> const &) {
        return Kokkos::Array<Real, 1>{static_cast<Real>(ent)};
      });

  MeshField::LO numErrors = 0;
  // the dof holders are the vertices and, for order 2, the edges
  for (int holderDim = 0; holderDim < order; ++holderDim) {
    const auto topo = holderDim == 0 ? MeshField::Vertex : MeshField::Edge;
    const auto up = mesh.ask_up(holderDim, dim);
    const auto a2ab = up.a2ab;
    const auto ab2b = up.ab2b;
    MeshField::LO lerrorsDim = 0;
    Kokkos::parallel_reduce(
        "checkOwnership", mesh.nents(holderDim),
        KOKKOS_LAMBDA(const int e, MeshField::LO &lerrors) {
          auto lowest = ab2b[a2ab[e]];
          for (auto i = a2ab[e] + 1; i < a2ab[e + 1]; ++i)
            lowest = Kokkos::min(lowest, ab2b[i]);
          if (field(e, 0, 0, topo) != lowest)
            ++lerrors;
        },
        lerrorsDim);
    numErrors += lerrorsDim;
  }
  CHECK(numErrors == 0);
}

/*
 * Ensure that a coordinate field created with a map recovers the map applied
 * to the linear coordinate field
 */
template <size_t dim, size_t order, typename CoordinateMap,
          template <typename...> typename Controller>
void testCoordinateMap(
    Omega_h::Mesh &mesh,
    MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf,
    CoordinateMap const &coordMap) {
  auto mappedWithCtrlr =
      omf.template CreateLagrangeCoordinateField<order>(coordMap);
  auto mapped = omf.CreateLagrangeElement(mappedWithCtrlr.field);
  auto linear = omf.CreateLagrangeElement(omf.getCoordField().field);
  const auto lc = offNodeCoords<dim>(mesh.nelems());
  const auto xm = MeshField::evaluate(mapped, lc, numOffNodePts);
  const auto xl = MeshField::evaluate(linear, lc, numOffNodePts);
  MeshField::LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkCoordinateMap", xm.extent(0),
      KOKKOS_LAMBDA(const int pt, MeshField::LO &lerrors) {
        Kokkos::Array<Real, dim> x;
        for (size_t d = 0; d < dim; ++d)
          x[d] = xl(pt, d);
        const auto expected = coordMap(x);
        for (size_t d = 0; d < dim; ++d)
          if (Kokkos::fabs(xm(pt, d) - expected[d]) > MeshField::Epsilon)
            ++lerrors;
      },
      numErrors);
  CHECK(numErrors == 0);
}

TEMPLATE_LIST_TEST_CASE("interpolation", "[interpolate]", MeshCases) {
  constexpr size_t dim = TestType::dim;
  auto mesh = buildMesh<dim>();
  MeshField::OmegahMeshField<ExecutionSpace, dim, TestType::template Controller>
      omf(mesh);
  SECTION("ownership P1") { testOwnership<dim, 1>(mesh, omf); }
  SECTION("ownership P2") { testOwnership<dim, 2>(mesh, omf); }
  SECTION("coordinate map P2 identity") {
    testCoordinateMap<dim, 2>(mesh, omf, MeshField::IdentityCoordinateMap());
  }
  SECTION("coordinate map P1 shear") {
    testCoordinateMap<dim, 1>(mesh, omf, Shear<dim>());
  }
  SECTION("coordinate map P2 shear") {
    testCoordinateMap<dim, 2>(mesh, omf, Shear<dim>());
  }
  SECTION("coordinate map P2 bend") {
    testCoordinateMap<dim, 2>(mesh, omf, Bend<dim>());
  }
}
