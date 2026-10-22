#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Integrate.hpp"
#include "Omega_h_build.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>

// Checks MeshField::TriangleIntegration:
// 1. every rule has positive weights and points inside the reference
//    triangle, and integrates xi0^a xi1^b exactly for a + b up to its
//    accuracy,
// 2. getAccurate(order) picks the expected rule for orders 1 to 4 and none
//    for order 5,
// 3. Integrator::process with orders 3 and 4 integrates monomials of that
//    degree exactly over a triangulated unit square.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;

bool isClose(Real a, Real b, Real tol) {
  return std::fabs(a - b) <= tol * (1 + std::fabs(b));
}

Real factorial(int n) { return n <= 1 ? 1.0 : n * factorial(n - 1); }

// exact integral of xi0^a xi1^b over the reference triangle
Real triangleMonomial(int a, int b) {
  return factorial(a) * factorial(b) / factorial(a + b + 2);
}

// A rule of accuracy p must integrate xi0^a xi1^b exactly for a + b <= p.
// The a = b = 0 case checks that the weights sum to 1/2, the area of the
// reference triangle.
int checkRule(MeshField::Integration<2> const &rule) {
  const auto pts = rule.getPoints();
  const int p = rule.getAccuracy();
  int failed = (static_cast<int>(pts.size()) != rule.countPoints());
  for (auto const &pt : pts) {
    const Real x0 = pt.param[0], x1 = pt.param[1];
    failed += !(pt.weight > 0 && x0 >= 0 && x1 >= 0 && x0 + x1 <= 1);
  }
  for (int a = 0; a <= p; ++a) {
    for (int b = 0; a + b <= p; ++b) {
      Real sum = 0;
      for (auto const &pt : pts)
        sum += pt.weight * std::pow(pt.param[0], a) * std::pow(pt.param[1], b);
      failed += !isClose(sum, triangleMonomial(a, b), 1e-14);
    }
  }
  std::cout << "triangle rule, accuracy " << p << ", " << pts.size()
            << " points: failures " << failed << "\n";
  return failed;
}

// getAccurate(order) returns the first rule with accuracy >= order
int checkRuleSelection() {
  const auto tri = MeshField::getIntegration<MeshField::Triangle>();
  const int numPts[] = {1, 3, 6, 6}; // orders 1 to 4
  int failed = 0;
  for (int order = 1; order <= 4; ++order)
    failed += tri->getAccurate(order)->countPoints() != numPts[order - 1];
  failed += tri->getAccurate(5) != nullptr;
  std::cout << "rule selection: failures " << failed << "\n";
  return failed;
}

// Integrates x^px y^py over the mesh. The coordinate map is affine on each
// triangle, so the integrand has degree px + py in the parametric
// coordinates and a rule of that order is exact.
template <typename FieldElement>
class MonomialIntegrator : public MeshField::Integrator {
public:
  MonomialIntegrator(FieldElement &feIn, int order, int pxIn, int pyIn)
      : MeshField::Integrator(order), fe(feIn), px(pxIn), py(pyIn) {}
  void atPoints(Kokkos::View<Real **> p, Kokkos::View<Real *> w,
                Kokkos::View<Real *> dV) override {
    const int numPtsPerElem = p.extent(0) / fe.numMeshEnts;
    const auto x = MeshField::evaluate(fe, p, numPtsPerElem);
    const Real a = px, b = py;
    Real sum = 0;
    Kokkos::parallel_reduce(
        "integrateMonomial", w.extent(0),
        KOKKOS_LAMBDA(const int i, Real &lsum) {
          lsum +=
              w(i) * dV(i) * Kokkos::pow(x(i, 0), a) * Kokkos::pow(x(i, 1), b);
        },
        sum);
    integral += sum;
  }
  FieldElement &fe;
  int px, py;
  Real integral = 0;
};

// integral of x^px y^py over the unit square
Real squareMonomial(int px, int py) { return 1.0 / ((px + 1) * (py + 1)); }

int checkIntegrator(Omega_h::Mesh &mesh) {
  MeshField::OmegahMeshField<ExecutionSpace, 2> omf(mesh);
  auto coordField = omf.getCoordField();
  const auto [shp, map] = MeshField::Omegah::getTriangleElement<1>(mesh);
  MeshField::FieldElement fe(mesh.nelems(), coordField.field, shp, map);
  // {order, px, py}: each monomial has degree equal to the order
  const int cases[][3] = {{3, 2, 1}, {3, 0, 3}, {4, 2, 2}, {4, 3, 1}};
  int failed = 0;
  std::cout.precision(16);
  for (auto const &c : cases) {
    MonomialIntegrator integrator(fe, c[0], c[1], c[2]);
    integrator.process(fe);
    const Real expected = squareMonomial(c[1], c[2]);
    const bool ok = isClose(integrator.integral, expected, 1e-13);
    std::cout << "order " << c[0] << ": integral of x^" << c[1] << " y^" << c[2]
              << " = " << integrator.integral << ", expected " << expected
              << (ok ? "" : " FAIL") << "\n";
    failed += !ok;
  }
  return failed;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    auto lib = Omega_h::Library(&argc, &argv);
    failed += checkRule(MeshField::TriangleIntegration::N1());
    failed += checkRule(MeshField::TriangleIntegration::N2());
    failed += checkRule(MeshField::TriangleIntegration::N4());
    failed += checkRuleSelection();
    auto tris = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0,
                                   3, 3, 0);
    failed += checkIntegrator(tris);
  }
  Kokkos::finalize();
  return failed ? 1 : 0;
}
