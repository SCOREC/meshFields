#include "MeshField_TestFunctions.hpp"
#include "MeshField_TestHarness.hpp"
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <string>
#include <utility>

// the coordinate field's own element
struct CoordinateElement {
  template <typename Omf, typename CoordField>
  static auto make(Omf &omf, CoordField const &coordField) {
    return std::make_pair(0, omf.CreateLagrangeElement(coordField));
  }
};

// a linear field on the coordinate field's geometry
struct MappedCoordinateElement {
  template <typename Omf, typename CoordField>
  static auto make(Omf &omf, CoordField const &coordField) {
    auto fieldWithCtrlr =
        omf.template CreateLagrangeField<MeshField::Real, 1, 1>();
    auto elm = omf.CreateLagrangeElement(fieldWithCtrlr.field,
                                         omf.CreateLagrangeElement(coordField));
    // the controller must outlive the element
    return std::make_pair(std::move(fieldWithCtrlr), elm);
  }
};

// count the integration points
class CountIntegrator : public MeshField::Integrator {
public:
  CountIntegrator(int order) : MeshField::Integrator(order) {}
  void atPoints(Kokkos::View<MeshField::Real **>,
                Kokkos::View<MeshField::Real *>,
                Kokkos::View<MeshField::Real *> dV) override {
    count += dV.extent(0);
  }
  size_t count = 0;
};

/*
 * Count the integration points of each order through an Element
 */
template <size_t dim, template <typename...> typename Controller,
          typename Element>
void testCount(Omega_h::Mesh &mesh, int order) {
  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto [storage, elm] = Element::make(omf, omf.getCoordField().field);
  CountIntegrator integrator(order);
  integrator.process(elm);
  // one point per simplex for order 1, one per vertex for order 2
  INFO("order " << order);
  CHECK(integrator.count == mesh.nelems() * (order == 1 ? 1 : dim + 1));
}

/*
 * Integrate the measure of the mesh with a coordinate field of the given
 * order, created with coordMap, through an Element.
 */
template <size_t dim, template <typename...> typename Controller,
          typename Element, size_t order,
          typename CoordinateMap = MeshField::IdentityCoordinateMap>
void testMeasure(Omega_h::Mesh &mesh, MeshField::Real expected,
                 std::string const &name,
                 CoordinateMap const &coordMap = CoordinateMap()) {
  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto coordFieldWithCtrlr =
      omf.template CreateLagrangeCoordinateField<order>(coordMap);
  auto [storage, elm] = Element::make(omf, coordFieldWithCtrlr.field);
  MeasureIntegrator integrator(order);
  integrator.process(elm);
  INFO(name);
  CHECK_THAT(integrator.measure,
             Catch::Matchers::WithinAbs(expected, MeshField::Epsilon));
}

template <size_t dim, template <typename...> typename Controller,
          typename Element>
void testMeasures(Omega_h::Mesh &mesh) {
  // the unit square and the 1x2x3 box
  const MeshField::Real volume = dim == 2 ? 1.0 : 6.0;
  testMeasure<dim, Controller, Element, 1>(mesh, volume, "linear");
  testMeasure<dim, Controller, Element, 2>(mesh, volume, "quadratic");
  testMeasure<dim, Controller, Element, 2>(mesh, volume, "sheared quadratic",
                                           Shear<dim>());
  if constexpr (dim == 2) {
    const Bulge<dim> bulge;
    testMeasure<dim, Controller, Element, 2>(mesh, 1.0 + bulge.delta / 6.0,
                                             "bulged quadratic", bulge);
  }
}

TEMPLATE_LIST_TEST_CASE("integration point count", "[integrator]", MeshCases) {
  constexpr size_t dim = TestType::dim;
  auto mesh = buildMesh<dim>();
  const int order = GENERATE(1, 2);
  CAPTURE(order);
  SECTION("FieldElement") {
    testCount<dim, TestType::template Controller, CoordinateElement>(mesh,
                                                                     order);
  }
  SECTION("MappedElement") {
    testCount<dim, TestType::template Controller, MappedCoordinateElement>(
        mesh, order);
  }
}

TEMPLATE_LIST_TEST_CASE("integrated measure", "[integrator]", MeshCases) {
  constexpr size_t dim = TestType::dim;
  auto mesh = buildMesh<dim>();
  SECTION("FieldElement") {
    testMeasures<dim, TestType::template Controller, CoordinateElement>(mesh);
  }
  SECTION("MappedElement") {
    testMeasures<dim, TestType::template Controller, MappedCoordinateElement>(
        mesh);
  }
}

using MeshField::Real;
using Powers = Kokkos::Array<int, 3>;

template <typename CoordinateElement>
class MonomialIntegrator : public MeshField::Integrator {
public:
  MonomialIntegrator(int order, CoordinateElement &coords, Powers powers)
      : MeshField::Integrator(order), coords(coords), powers(powers) {}

  void atPoints(Kokkos::View<Real **> p, Kokkos::View<Real *> w,
                Kokkos::View<Real *> dV) override {
    const auto x = MeshField::evaluateStaticLength(coords, p);
    const auto exponents = powers;
    Kokkos::parallel_reduce(
        "integrateMonomial", dV.extent(0),
        KOKKOS_LAMBDA(int i, Real &sum) {
          Real value = 1;
          for (size_t d = 0; d < CoordinateElement::MeshEntDim; ++d)
            for (int j = 0; j < exponents[d]; ++j)
              value *= x(i, d);
          sum += w(i % w.extent(0)) * dV(i) * value;
        },
        integral);
  }

  Real integral = 0;

private:
  CoordinateElement &coords;
  Powers powers;
};

int factorial(int n) {
  int result = 1;
  for (int i = 2; i <= n; ++i)
    result *= i;
  return result;
}

template <typename Case, size_t Order, typename Element>
void checkMonomials(Omega_h::Mesh &mesh, bool referenceSimplex) {
  constexpr auto dim = Case::dim;
  MeshField::OmegahMeshField<ExecutionSpace, dim, Case::template Controller>
      omf(mesh);
  auto field = omf.template CreateLagrangeCoordinateField<Order>(
      MeshField::IdentityCoordinateMap());
  auto coords = omf.CreateLagrangeElement(field.field);
  auto [storage, elm] = Element::make(omf, field.field);
  auto check = [&](int degree, Powers powers, Real expected) {
    CAPTURE(dim, Order, degree, powers[0], powers[1], powers[2]);
    MonomialIntegrator integrator(degree, coords, powers);
    integrator.process(elm);
    CHECK_THAT(integrator.integral,
               Catch::Matchers::WithinAbs(expected, MeshField::Epsilon));
  };
  if (!referenceSimplex) {
    // Integrate x over the unit square or the 1x2x3 box.
    check(Order, {1, 0, 0}, dim == 2 ? 0.5 : 3.0);
    return;
  }
  for (int degree = 1; degree <= Order; ++degree)
    for (int x = 0; x <= degree; ++x)
      for (int z = 0; z <= (dim == 3 ? degree - x : 0); ++z) {
        const int y = degree - x - z;
        const Real expected = Real(factorial(x) * factorial(y) * factorial(z)) /
                              factorial(degree + dim);
        check(degree, {x, y, z}, expected);
      }
}

template <size_t Dim> Omega_h::Mesh buildReferenceSimplex() {
  Omega_h::Mesh mesh(&library());
  if constexpr (Dim == 2)
    Omega_h::build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, Dim,
                                         Omega_h::LOs({0, 1, 2}),
                                         Omega_h::Reals({0, 0, 1, 0, 0, 1}));
  else
    Omega_h::build_from_elems_and_coords(
        &mesh, OMEGA_H_SIMPLEX, Dim, Omega_h::LOs({0, 1, 2, 3}),
        Omega_h::Reals({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}));
  return mesh;
}

template <typename Case, typename Element> void checkMonomialMeshes() {
  SECTION("reference simplex") {
    auto mesh = buildReferenceSimplex<Case::dim>();
    checkMonomials<Case, 1, Element>(mesh, true);
    checkMonomials<Case, 2, Element>(mesh, true);
  }
  SECTION("box mesh") {
    auto mesh = buildMesh<Case::dim>();
    checkMonomials<Case, 1, Element>(mesh, false);
    checkMonomials<Case, 2, Element>(mesh, false);
  }
}

TEMPLATE_LIST_TEST_CASE("integrated monomials", "[integrator]", MeshCases) {
  SECTION("FieldElement") {
    checkMonomialMeshes<TestType, CoordinateElement>();
  }
  SECTION("MappedElement") {
    checkMonomialMeshes<TestType, MappedCoordinateElement>();
  }
}
