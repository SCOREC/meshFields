#include "MeshField_MappedElement.hpp"
#include "MeshField_TestFunctions.hpp"
#include "MeshField_TestHarness.hpp"
#include <string>
#include <type_traits>

template <size_t dim, template <typename...> typename Controller>
using OmegahMeshField =
    MeshField::OmegahMeshField<ExecutionSpace, dim, Controller>;

constexpr int numPts = 3;

// points used for testing element values
template <size_t dim> KOKKOS_INLINE_FUNCTION Vec<dim> point(int pt) {
  const MeshField::Real pts[numPts][3] = {
      {0.25, 0.25, 0.25}, {0.1, 0.5, 0.15}, {0.4, 0.1, 0.3}};
  Vec<dim> xi;
  for (size_t d = 0; d < dim; ++d)
    xi[d] = pts[pt][d];
  return xi;
}

// 1. physical gradients on curved geometry agree with central differences
template <size_t dim, size_t FieldOrder,
          template <typename...> typename Controller>
bool testCurvedGradientFiniteDifference(Omega_h::Mesh &mesh,
                                        OmegahMeshField<dim, Controller> &omf) {
  QuadraticFunction<dim> func;
  // sheared, then curved by a smooth offset
  const MeshField::Real s = 0.03;
  auto cfWc = omf.template CreateLagrangeCoordinateField<2>(
      KOKKOS_LAMBDA(Vec<dim> x) {
        x = Shear<dim>()(x);
        auto y = x;
        for (size_t d = 0; d < dim; ++d)
          y[d] += s * Kokkos::sin(3.0 * x[(d + 1) % dim] + d + 1);
        return y;
      });
  auto fieldWc =
      omf.template CreateLagrangeField<MeshField::Real, FieldOrder, 1>();
  auto geomElm = omf.CreateLagrangeElement(cfWc.field);
  auto elm = omf.CreateLagrangeElement(fieldWc.field, geomElm);
  MeshField::setDofByInterpolation(elm, KOKKOS_LAMBDA(Vec<dim> const &x) {
    return valueScaledByComponent<1>(func, x);
  });
  auto coordElm = omf.CreateLagrangeElement(cfWc.field, geomElm);

  const MeshField::Real h = 1e-5;
  MeshField::LO errors = 0;
  Kokkos::parallel_reduce(
      "checkFiniteDifference", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, MeshField::LO &lerr) {
        const auto c = elm.coefficients(ent);
        const auto geo = elm.geometry(ent);
        const auto cx = coordElm.coefficients(ent);
        for (int pt = 0; pt < numPts; ++pt) {
          const auto xi = point<dim>(pt);
          const auto g = elm.gradientFromCoefficients(c, geo, xi);
          for (size_t k = 0; k < dim; ++k) {
            auto xp = xi;
            auto xm = xi;
            xp[k] += h;
            xm[k] -= h;
            const auto dv = (elm.valueFromCoefficients(c, xp)[0] -
                             elm.valueFromCoefficients(c, xm)[0]) /
                            (2 * h);
            const auto posP = coordElm.valueFromCoefficients(cx, xp);
            const auto posM = coordElm.valueFromCoefficients(cx, xm);
            MeshField::Real dvFromGrad = 0;
            for (size_t j = 0; j < dim; ++j)
              dvFromGrad += (posP[j] - posM[j]) / (2 * h) * g[j];
            if (!(Kokkos::fabs(dvFromGrad - dv) <= 1e-7))
              lerr++;
          }
        }
      },
      errors);
  return report("testCurvedGradientFiniteDifference(P" +
                    std::to_string(FieldOrder) + ")",
                errors);
}

// the parametric bubble xi0*xi1: zero at the vertices, and in P2
template <size_t dim>
KOKKOS_INLINE_FUNCTION MeshField::Real bubble(Vec<dim> const &xi) {
  return xi[0] * xi[1];
}

// Interpolates the stored dofs plus (c+1)*bubble onto the nodes of a Lagrange
// RefShape.  The coefficients are for a different basis than the stored dofs,
// and use all of its nodes.
template <typename RefShape> struct InterpolateToShape {
  template <typename FieldElem>
  KOKKOS_INLINE_FUNCTION auto referenceShape(FieldElem const &) const {
    return RefShape{};
  }

  template <typename FieldElem, typename GeomElem>
  KOKKOS_INLINE_FUNCTION auto coefficients(FieldElem const &field,
                                           GeomElem const &, int ent) const {
    constexpr size_t dim = FieldElem::MeshEntDim;
    constexpr size_t numComp = FieldElem::NumComponents;
    const auto dofs = field.coefficients(ent);
    const auto nodes = RefShape{}.getNodeParametricCoords();
    Kokkos::Array<MeshField::Real, RefShape::numNodes * numComp> c;
    for (size_t n = 0; n < RefShape::numNodes; ++n) {
      Vec<dim> xi;
      for (size_t d = 0; d < dim; ++d)
        xi[d] = nodes[n * dim + d];
      const auto v = field.valueFromCoefficients(dofs, xi);
      for (size_t ci = 0; ci < numComp; ++ci)
        c[n * numComp + ci] = v[ci] + (ci + 1) * bubble<dim>(xi);
    }
    return c;
  }
};

// 2. P1 dofs plus a bubble on a P2 reference basis
template <size_t dim, template <typename...> typename Controller>
void testReferenceShape(Omega_h::Mesh &mesh,
                        OmegahMeshField<dim, Controller> &omf) {
  using P2Shape =
      std::conditional_t<dim == 2, MeshField::QuadraticTriangleShape,
                         MeshField::QuadraticTetrahedronShape>;
  constexpr size_t numComp = 2;
  auto cfWc = omf.template CreateLagrangeCoordinateField<1>(Shear<dim>());
  auto fieldWc =
      omf.template CreateLagrangeField<MeshField::Real, 1, numComp>();
  auto geomElm = omf.CreateLagrangeElement(cfWc.field);
  auto p1 = omf.CreateLagrangeElement(fieldWc.field, geomElm);
  const LinearFunction<dim> func;
  MeshField::setDofByInterpolation(p1, KOKKOS_LAMBDA(Vec<dim> const &x) {
    return valueScaledByComponent<numComp>(func, x);
  });
  const MeshField::MappedElement onP2(omf.CreateLagrangeElement(fieldWc.field),
                                      geomElm, InterpolateToShape<P2Shape>{});
  using OnP2 = std::remove_cvref_t<decltype(onP2)>;
  static_assert(std::is_same_v<typename OnP2::ReferenceShape, P2Shape>);

  MeshField::LO errors = 0;
  Kokkos::parallel_reduce(
      "checkReferenceShape", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, MeshField::LO &lerr) {
        const auto c1 = p1.coefficients(ent);
        const auto c2 = onP2.coefficients(ent);
        const auto geo = p1.geometry(ent);
        for (int pt = 0; pt < numPts; ++pt) {
          const auto xi = point<dim>(pt);
          const auto v1 = p1.valueFromCoefficients(c1, xi);
          const auto v2 = onP2.valueFromCoefficients(c2, xi);
          const auto g1 = p1.gradientFromCoefficients(c1, geo, xi);
          const auto g2 = onP2.gradientFromCoefficients(c2, geo, xi);
          // the physical gradient of (c+1)*bubble
          Kokkos::Array<MeshField::Real, dim * numComp> localBubbleGrad;
          for (size_t d = 0; d < dim; ++d)
            for (size_t ci = 0; ci < numComp; ++ci)
              localBubbleGrad[d * numComp + ci] =
                  (ci + 1) * (d == 0 ? xi[1] : (d == 1 ? xi[0] : 0.0));
          const auto bubbleGrad =
              MeshField::IdentityMap{}.gradient(localBubbleGrad, geo);
          for (size_t ci = 0; ci < numComp; ++ci) {
            const auto expected = v1[ci] + (ci + 1) * bubble<dim>(xi);
            if (!(Kokkos::fabs(v2[ci] - expected) <= MeshField::Epsilon))
              lerr++;
          }
          for (size_t i = 0; i < dim * numComp; ++i)
            if (!(Kokkos::fabs(g2[i] - (g1[i] + bubbleGrad[i])) <=
                  MeshField::Epsilon))
              lerr++;
        }
      },
      errors);
  report("testReferenceShape", errors);
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  MeshField::Debug = true;
  t.forEachController([&]<template <typename...> typename Controller>() {
    t.forEachMesh([&]<size_t dim>(Omega_h::Mesh &mesh) {
      OmegahMeshField<dim, Controller> omf(mesh);
      testCurvedGradientFiniteDifference<dim, 1, Controller>(mesh, omf);
      testCurvedGradientFiniteDifference<dim, 2, Controller>(mesh, omf);
      testReferenceShape<dim, Controller>(mesh, omf);
    });
  });
  return t.exitCode();
}
