#include "MeshField.hpp"
#include "MeshField_Integrate.hpp"
#include "MeshField_MappedElement.hpp"
#include "MeshField_TestFunctions.hpp"
#include "Omega_h_build.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

// MappedElement tests on triangles and tetrahedra.  The meshes are sheared so
// the Jacobians are full matrices.
//
// 1. P2 field on P1 geometry 
// 2. finite differences on curved P2 geometry
// 3. measure/Integrator with curved geometry 

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
template <size_t dim>
using OmegahMeshField = MeshField::OmegahMeshField<ExecutionSpace, dim>;

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

// set component c of the field element elm to (c+1)*func at its nodes, with
// the node positions from the coordinate element coordElm
template <size_t dim, typename Func, typename FieldElem, typename CoordElem>
void interpolate(FieldElem &elm, CoordElem const &coordElm, Func const &func) {
  constexpr size_t numComp = FieldElem::NumComponents;
  MeshField::setDofByInterpolation(
      elm, KOKKOS_LAMBDA(const int ent, Vec<dim> const &xi) {
        const auto f = func(coordElm.getValue(ent, xi));
        Kokkos::Array<MeshField::Real, numComp> val;
        for (size_t c = 0; c < numComp; ++c)
          val[c] = (c + 1) * f;
        return val;
      });
}

// 1. P2 field on P1 geometry reproduces a quadratic exactly
template <size_t dim>
bool testP2FieldOnP1Geometry(Omega_h::Mesh &mesh, OmegahMeshField<dim> &omf) {
  QuadraticFunction<dim> func;
  auto cf1Wc = omf.template CreateLagrangeCoordinateField<1>(Shear<dim>());
  constexpr size_t numComp = 2;
  auto fieldWc =
      omf.template CreateLagrangeField<MeshField::Real, 2, numComp>();
  auto geomElm = omf.CreateLagrangeElement(cf1Wc.field);
  auto fieldElm = omf.CreateLagrangeElement(fieldWc.field);
  interpolate<dim>(fieldElm, geomElm, func);

  auto elm = omf.CreateLagrangeElement(fieldWc.field, geomElm);
  auto coordElm = omf.CreateLagrangeElement(cf1Wc.field, geomElm);

  MeshField::LO errors = 0;
  Kokkos::parallel_reduce(
      "checkP2FieldOnP1Geometry", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, MeshField::LO &lerr) {
        const auto c = elm.coefficients(ent);
        const auto cx = coordElm.coefficients(ent);
        for (int pt = 0; pt < numPts; ++pt) {
          const auto xi = point<dim>(pt);
          const auto pos = coordElm.valueFromCoefficients(cx, xi);
          Vec<dim> x;
          for (size_t d = 0; d < dim; ++d)
            x[d] = pos[d];
          const auto v = elm.valueFromCoefficients(c, xi);
          const auto g = elm.gradientFromCoefficients(c, xi);
          const auto gf = func.grad(x);
          for (size_t ci = 0; ci < numComp; ++ci) {
            if (!(Kokkos::fabs(v[ci] - (ci + 1) * func(x)) <=
                  MeshField::Epsilon))
              lerr++;
            for (size_t d = 0; d < dim; ++d)
              if (!(Kokkos::fabs(g[d * numComp + ci] - (ci + 1) * gf[d]) <=
                    MeshField::Epsilon))
                lerr++;
          }
        }
      },
      errors);
  std::cout << "testP2FieldOnP1Geometry(" << dim
            << "d): " << (errors ? "FAIL" : "pass") << "\n";
  return errors == 0;
}

// 2. physical gradients on curved geometry agree with central differences
template <size_t dim, size_t FieldOrder>
bool testCurvedGradientFiniteDifference(Omega_h::Mesh &mesh,
                                        OmegahMeshField<dim> &omf) {
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
  auto fieldElm = omf.CreateLagrangeElement(fieldWc.field);
  interpolate<dim>(fieldElm, geomElm, func);

  auto elm = omf.CreateLagrangeElement(fieldWc.field, geomElm);
  auto coordElm = omf.CreateLagrangeElement(cfWc.field, geomElm);

  const MeshField::Real h = 1e-5;
  MeshField::LO errors = 0;
  Kokkos::parallel_reduce(
      "checkFiniteDifference", mesh.nelems(),
      KOKKOS_LAMBDA(const int ent, MeshField::LO &lerr) {
        const auto c = elm.coefficients(ent);
        const auto cx = coordElm.coefficients(ent);
        for (int pt = 0; pt < numPts; ++pt) {
          const auto xi = point<dim>(pt);
          const auto g = elm.gradientFromCoefficients(c, xi);
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
  std::cout << "testCurvedGradientFiniteDifference(" << dim << "d, P"
            << FieldOrder << " field): " << (errors ? "FAIL" : "pass") << "\n";
  return errors == 0;
}

// compute the measure (3D volume, 2D area, 1D length)
class MeasureIntegrator : public MeshField::Integrator {
public:
  MeasureIntegrator() : MeshField::Integrator(2) {}
  void atPoints(Kokkos::View<MeshField::Real **>,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) override {
    MeshField::Real sum = 0;
    Kokkos::parallel_reduce(
        "integrateMeasure", w.extent(0),
        KOKKOS_LAMBDA(const int i, MeshField::Real &lsum) {
          lsum += w(i) * dV(i);
        },
        sum);
    measure += sum;
  }
  MeshField::Real measure = 0;
};

// 3. measure of P2 geometry through an Integrator over a MappedElement
template <size_t dim, typename CoordinateMap>
bool testMeasure(OmegahMeshField<dim> &omf, CoordinateMap const &coordMap,
                 MeshField::Real expected) {
  auto cfWc = omf.template CreateLagrangeCoordinateField<2>(coordMap);
  auto fieldWc = omf.template CreateLagrangeField<MeshField::Real, 1, 1>();
  auto elm = omf.CreateLagrangeElement(fieldWc.field,
                                       omf.CreateLagrangeElement(cfWc.field));
  MeasureIntegrator integrator;
  integrator.process(elm);
  const bool pass =
      std::abs(integrator.measure - expected) <= MeshField::Epsilon;
  std::cout << "testMeasure(" << dim << "d): " << integrator.measure
            << " expected " << expected << (pass ? " pass" : " FAIL") << "\n";
  return pass;
}

template <size_t dim> int runTests(Omega_h::Mesh &mesh) {
  OmegahMeshField<dim> omf(mesh);
  int failed = 0;
  failed += !testP2FieldOnP1Geometry<dim>(mesh, omf);
  failed += !testCurvedGradientFiniteDifference<dim, 1>(mesh, omf);
  failed += !testCurvedGradientFiniteDifference<dim, 2>(mesh, omf);
  if constexpr (dim == 2) {
    // the unit square with its top bent into y = 1 + delta * x(1 - x); the
    // top edges are parabolas, so P2 reproduces the boundary and the area
    const MeshField::Real delta = 0.1;
    failed += !testMeasure<dim>(
        omf,
        KOKKOS_LAMBDA(Vec<dim> x) {
          x[1] *= 1 + delta * x[0] * (1 - x[0]);
          return x;
        },
        1.0 + delta / 6.0);
  } else {
    failed += !testMeasure<dim>(omf, Shear<dim>(), 6.0); // 1x2x3 box
  }
  return failed;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    auto lib = Omega_h::Library(&argc, &argv);
    MeshField::Debug = true;
    auto tris = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0,
                                   3, 3, 0);
    auto tets = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 2.0, 3.0,
                                   2, 2, 2);
    failed += runTests<2>(tris);
    failed += runTests<3>(tets);
  }
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
