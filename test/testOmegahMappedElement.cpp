#include "MeshField_MappedElement.hpp"
#include "MeshField_TestFunctions.hpp"
#include "MeshField_TestHarness.hpp"
#include <string>

// MappedElement physical gradients on curved P2 geometry, against central
// differences, on triangles and tetrahedra.  The meshes are sheared so the
// Jacobians are full matrices.
//
// Values and gradients on affine geometry are tested in testReproduction, and
// the measure in testIntegratorMeasure.

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

// physical gradients on curved geometry agree with central differences
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

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  MeshField::Debug = true;
  t.forEachController([&]<template <typename...> typename Controller>() {
    t.forEachMesh([&]<size_t dim>(Omega_h::Mesh &mesh) {
      OmegahMeshField<dim, Controller> omf(mesh);
      testCurvedGradientFiniteDifference<dim, 1, Controller>(mesh, omf);
      testCurvedGradientFiniteDifference<dim, 2, Controller>(mesh, omf);
    });
  });
  return t.exitCode();
}
