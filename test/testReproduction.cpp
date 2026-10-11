#include "MeshField_MappedElement.hpp"
#include "MeshField_TestFunctions.hpp"
#include "MeshField_TestHarness.hpp"
#include <string>
#include <utility>

// Interpolate (c+1) f into each component c of a field, for each element type,
// on affine geometry, and check that
//  - the values at the element's nodes are (c+1) f, for any f
//  - (c+1) f  and gradient is reproduced in the element interior when the field's order is at least
//    the degree of f

using MeshField::LO;
using MeshField::Real;

// a FieldElement
struct Lagrange {
  static constexpr const char *name = "FieldElement";
  template <size_t order, size_t numComp, typename Omf, typename GeomElem>
  static auto make(Omf &omf, GeomElem const &) {
    auto fieldWithCtrlr =
        omf.template CreateLagrangeField<Real, order, numComp>();
    auto elm = omf.CreateLagrangeElement(fieldWithCtrlr.field);
    // the controller must outlive the element
    return std::make_pair(std::move(fieldWithCtrlr), elm);
  }
};

// a MappedElement on the geometry
struct Mapped {
  static constexpr const char *name = "MappedElement";
  template <size_t order, size_t numComp, typename Omf, typename GeomElem>
  static auto make(Omf &omf, GeomElem const &geom) {
    auto fieldWithCtrlr =
        omf.template CreateLagrangeField<Real, order, numComp>();
    auto elm = omf.CreateLagrangeElement(fieldWithCtrlr.field, geom);
    return std::make_pair(std::move(fieldWithCtrlr), elm);
  }
};

template <typename Elm>
constexpr bool IsMapped = requires(Elm const &elm) { elm.geom; };

constexpr size_t numOffNodePts = 2;
template <size_t dim> KOKKOS_INLINE_FUNCTION Vec<dim> interiorPoint(size_t pt) {
  Vec<dim> xi;
  for (size_t d = 0; d < dim; ++d)
    xi[d] = pt == 0 ? 1.0 / (dim + 1) : 0.1 * (d + 1);
  return xi;
}

// the number of components of the physical gradient that differ from (c+1)
// grad f; zero for elements without physical gradients
template <typename Elm, typename Coefficients, size_t dim, typename Function>
KOKKOS_INLINE_FUNCTION LO gradientErrors(Elm const &elm, Coefficients const &c,
                                         int ent, Vec<dim> const &xi,
                                         Function const &func,
                                         Vec<dim> const &x) {
  LO errors = 0;
  if constexpr (IsMapped<Elm>) {
    constexpr size_t numComp = Elm::NumComponents;
    const auto g = elm.gradientFromCoefficients(c, elm.geometry(ent), xi);
    const auto expected = gradScaledByComponent<numComp>(func, x);
    for (size_t i = 0; i < dim * numComp; ++i)
      if (!(Kokkos::fabs(g[i] - expected[i]) <= MeshField::Epsilon))
        ++errors;
  }
  return errors;
}

template <size_t dim, template <typename...> typename Controller,
          typename Element, size_t order, size_t numComp, typename Function,
          typename CoordinateMap>
void testReproduction(Omega_h::Mesh &mesh, CoordinateMap const &coordMap,
                      std::string const &mapName) {
  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto coordsWithCtrlr =
      omf.template CreateLagrangeCoordinateField<1>(coordMap);
  const auto geom = omf.CreateLagrangeElement(coordsWithCtrlr.field);
  auto made = Element::template make<order, numComp>(omf, geom);
  auto elm = made.second;
  using Elm = decltype(elm);
  const Function func;

  // reference space interpolation, through the geometry
  MeshField::setDofByInterpolation(
      elm, KOKKOS_LAMBDA(const int ent, Vec<dim> const &xi) {
        return valueScaledByComponent<numComp>(func, geom.getValue(ent, xi));
      });

  const auto shapeFn = [&] {
    if constexpr (IsMapped<Elm>)
      return elm.field.shapeFn;
    else
      return elm.shapeFn;
  }();
  constexpr size_t numNodes = decltype(shapeFn)::numNodes;
  const auto nodeXi = shapeFn.getNodeParametricCoords();
  // f is reproduced everywhere only if the field's order is high enough
  constexpr bool reproduces = Function::degree <= order;
  constexpr size_t numPts = numNodes + (reproduces ? numOffNodePts : 0);

  LO errors = 0;
  Kokkos::parallel_reduce(
      "checkReproduction", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, LO &lerr) {
        const auto c = elm.coefficients(ent);
        const auto cx = geom.coefficients(ent);
        for (size_t pt = 0; pt < numPts; ++pt) {
          Vec<dim> xi;
          if (pt < numNodes) {
            for (size_t d = 0; d < dim; ++d)
              xi[d] = nodeXi[pt * dim + d];
          } else {
            xi = interiorPoint<dim>(pt - numNodes);
          }
          const auto x = geom.valueFromCoefficients(cx, xi);
          const auto v = elm.valueFromCoefficients(c, xi);
          const auto expected = valueScaledByComponent<numComp>(func, x);
          for (size_t i = 0; i < numComp; ++i)
            if (!(Kokkos::fabs(v[i] - expected[i]) <= MeshField::Epsilon))
              ++lerr;
          if (reproduces)
            lerr += gradientErrors(elm, c, ent, xi, func, x);
        }
      },
      errors);
  report("testReproduction(P" + std::to_string(order) + ", " +
             std::to_string(numComp) + " comp, degree " +
             std::to_string(Function::degree) + " f, " + mapName + ", " +
             Element::name + ")",
         errors);
}

template <size_t dim, template <typename...> typename Controller,
          typename Element, typename CoordinateMap>
void testOrders(Omega_h::Mesh &mesh, CoordinateMap const &coordMap,
                std::string const &mapName) {
  using Linear = LinearFunction<dim>;
  using Quadratic = QuadraticFunction<dim>;
  testReproduction<dim, Controller, Element, 1, 1, Linear>(mesh, coordMap,
                                                           mapName);
  testReproduction<dim, Controller, Element, 1, 3, Linear>(mesh, coordMap,
                                                           mapName);
  // the order is too low to reproduce f, so only the nodes are checked
  testReproduction<dim, Controller, Element, 1, 1, Quadratic>(mesh, coordMap,
                                                              mapName);
  testReproduction<dim, Controller, Element, 2, 1, Linear>(mesh, coordMap,
                                                           mapName);
  testReproduction<dim, Controller, Element, 2, 1, Quadratic>(mesh, coordMap,
                                                              mapName);
  testReproduction<dim, Controller, Element, 2, 3, Quadratic>(mesh, coordMap,
                                                              mapName);
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  MeshField::Debug = true;
  t.forEachController([&]<template <typename...> typename Controller>() {
    t.forEachMesh([&]<size_t dim>(Omega_h::Mesh &mesh) {
      forEachType<Lagrange, Mapped>([&]<typename Element>() {
        testOrders<dim, Controller, Element>(
            mesh, MeshField::IdentityCoordinateMap(), "identity");
        testOrders<dim, Controller, Element>(mesh, Shear<dim>(), "shear");
      });
    });
  });
  return t.exitCode();
}
