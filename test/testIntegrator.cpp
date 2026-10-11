#include "MeshField_TestFunctions.hpp"
#include "MeshField_TestHarness.hpp"
#include <string>
#include <utility>

// the coordinate field's own element
struct CoordinateElement {
  static constexpr const char *name = "FieldElement";
  template <typename Omf, typename CoordField>
  static auto make(Omf &omf, CoordField const &coordField) {
    return std::make_pair(0, omf.CreateLagrangeElement(coordField));
  }
};

// a linear field on the coordinate field's geometry
struct MappedCoordinateElement {
  static constexpr const char *name = "MappedElement";
  template <typename Omf, typename CoordField>
  static auto make(Omf &omf, CoordField const &coordField) {
    auto fieldWithCtrlr =
        omf.template CreateLagrangeField<MeshField::Real, 1, 1>();
    auto elm = omf.CreateLagrangeElement(
        fieldWithCtrlr.field, omf.CreateLagrangeElement(coordField));
    // the controller must outlive the element
    return std::make_pair(std::move(fieldWithCtrlr), elm);
  }
};

// count the integration points
class CountIntegrator : public MeshField::Integrator {
public:
  CountIntegrator(int order) : MeshField::Integrator(order) {}
  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *>,
                Kokkos::View<MeshField::Real *>) override {
    count += p.extent(0);
  }
  size_t count = 0;
};

/*
 * Count the integration points of each order through an Element
 */
template <size_t dim, template <typename...> typename Controller,
          typename Element, size_t order>
void testCount(Omega_h::Mesh &mesh) {
  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto [storage, elm] = Element::make(omf, omf.getCoordField().field);
  CountIntegrator integrator(order);
  integrator.process(elm);
  // one point per simplex for order 1, one per vertex for order 2
  const size_t expected = mesh.nelems() * (order == 1 ? 1 : dim + 1);
  if (!report("testCount(order " + std::to_string(order) + ", " +
                  Element::name + ")",
              integrator.count != expected))
    std::cout << "  count " << integrator.count << " expected " << expected
              << "\n";
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
  const auto err = std::abs(integrator.measure - expected);
  if (!report("testMeasure(" + name + ", " + Element::name + ")",
              err > MeshField::Epsilon))
    std::cout << "  measure " << integrator.measure << " expected "
              << expected << "\n";
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  t.forEachController([&]<template <typename...> typename Controller>() {
    t.forEachMesh([&]<size_t dim>(Omega_h::Mesh &mesh) {
      forEachType<CoordinateElement,
                  MappedCoordinateElement>([&]<typename Element>() {
        testCount<dim, Controller, Element, 1>(mesh);
        testCount<dim, Controller, Element, 2>(mesh);
        // the unit square and the 1x2x3 box
        const MeshField::Real volume = dim == 2 ? 1.0 : 6.0;
        testMeasure<dim, Controller, Element, 1>(mesh, volume, "linear");
        testMeasure<dim, Controller, Element, 2>(mesh, volume, "quadratic");
        testMeasure<dim, Controller, Element, 2>(
            mesh, volume, "sheared quadratic", Shear<dim>());
        if constexpr (dim == 2) {
          const Bulge<dim> bulge;
          testMeasure<dim, Controller, Element, 2>(
              mesh, 1.0 + bulge.delta / 6.0, "bulged quadratic", bulge);
        }
      });
    });
  });
  return t.exitCode();
}
