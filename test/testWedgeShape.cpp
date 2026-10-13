#include <Kokkos_Core.hpp> 
#include <MeshField_Defines.hpp>
#include <MeshField_Shape.hpp>
#include <iostream>

// Checks of MeshField::LinearWedgeShape. The checks run inside Kokkos kernels
// so the shape functions are exercised in the default execution space (e.g.,
// on a GPU), and failures are counted explicitly so the test also fails in
// builds where assert() is disabled.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;
using Shape = MeshField::LinearWedgeShape;


KOKKOS_INLINE_FUNCTION bool isClose(Real a, Real b, Real tol) {
  return Kokkos::fabs(a - b) <= tol;
}

// A function in the span of the wedge basis,
// {1, xi0, xi1, zeta, xi0*zeta, xi1*zeta}, and its gradient
KOKKOS_INLINE_FUNCTION Real f(Real x0, Real x1, Real z) {
  return 0.5 + 2 * x0 - 3 * x1 + 0.25 * z + 4 * x0 * z - 1.5 * x1 * z;
}
KOKKOS_INLINE_FUNCTION MeshField::Vector3 gradF(Real x0, Real x1, Real z) {
  return {2 + 4 * z, -3 - 1.5 * z, 0.25 + 4 * x0 - 1.5 * x1};
}

// Each shape function is one at its own node and zero at the others.
int checkKroneckerDelta() {
  int numFailed = 0;
  Kokkos::parallel_reduce(
      "wedgeKronecker", Kokkos::RangePolicy<ExecutionSpace>(0, Shape::numNodes),
      KOKKOS_LAMBDA(const int node, int &failed) {
        const Shape shp;
        const auto coords = shp.getNodeParametricCoords();
        const MeshField::Vector3 xi{coords[3 * node], coords[3 * node + 1],
                                    coords[3 * node + 2]};
        const auto N = shp.getValues(xi);
        for (size_t m = 0; m < Shape::numNodes; ++m) {
          const Real expected = (static_cast<size_t>(node) == m) ? 1.0 : 0.0;
          failed += !isClose(N[m], expected, MeshField::MachinePrecision);
        }
      },
      numFailed);
  return numFailed;
}

// At interior points: the values sum to one, the gradients sum to zero, the
// basis reproduces f and its gradient exactly, and the analytic gradients
// match central finite differences of the values.
int checkInteriorPoints() {
  // clang-format off
  const Real pts[][3] = {{1. / 3, 1. / 3,  0.0},
                         {0.2,    0.3,    -0.4},
                         {0.6,    0.1,     0.7},
                         {0.05,   0.9,     0.95},
                         {0.45,   0.45,   -0.95}};
  // clang-format on
  const int numPts = sizeof(pts) / sizeof(pts[0]);
  Kokkos::View<Real *[3], ExecutionSpace> pts_d("wedgePts", numPts);
  auto pts_h = Kokkos::create_mirror_view(pts_d);
  for (int i = 0; i < numPts; ++i)
    for (int d = 0; d < 3; ++d)
      pts_h(i, d) = pts[i][d];
  Kokkos::deep_copy(pts_d, pts_h);

  int numFailed = 0;
  Kokkos::parallel_reduce(
      "wedgeInterior", Kokkos::RangePolicy<ExecutionSpace>(0, numPts),
      KOKKOS_LAMBDA(const int i, int &failed) {
        const Shape shp;
        const MeshField::Vector3 xi{pts_d(i, 0), pts_d(i, 1), pts_d(i, 2)};
        const auto N = shp.getValues(xi);
        const auto G = shp.getLocalGradients(xi);
        const auto nodes = shp.getNodeParametricCoords();

        Real sumN = 0;
        Real sumG[3] = {0, 0, 0};
        Real fInterp = 0;
        Real gInterp[3] = {0, 0, 0};
        for (size_t n = 0; n < Shape::numNodes; ++n) {
          const Real fn = f(nodes[3 * n], nodes[3 * n + 1], nodes[3 * n + 2]);
          sumN += N[n];
          fInterp += N[n] * fn;
          for (int d = 0; d < 3; ++d) {
            sumG[d] += G[3 * n + d];
            gInterp[d] += G[3 * n + d] * fn;
          }
        }
        const Real tol = 10 * MeshField::MachinePrecision;
        failed += !isClose(sumN, 1.0, tol);
        failed += !isClose(fInterp, f(xi[0], xi[1], xi[2]), tol);
        const auto gExact = gradF(xi[0], xi[1], xi[2]);
        for (int d = 0; d < 3; ++d) {
          failed += !isClose(sumG[d], 0.0, tol);
          failed += !isClose(gInterp[d], gExact[d], tol);
        }

        const Real h = 1e-6;
        for (int d = 0; d < 3; ++d) {
          auto xp = xi;
          auto xm = xi;
          xp[d] += h;
          xm[d] -= h;
          const auto Np = shp.getValues(xp);
          const auto Nm = shp.getValues(xm);
          for (size_t n = 0; n < Shape::numNodes; ++n)
            failed += !isClose((Np[n] - Nm[n]) / (2 * h), G[3 * n + d], 1e-8);
        }
      },
      numFailed);
  return numFailed;
}


int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int numFailed = 0;
  {
    const int kronecker = checkKroneckerDelta();
    std::cout << "LinearWedgeShape Kronecker delta failures: " << kronecker
              << "\n";
    const int interior = checkInteriorPoints();
    std::cout << "LinearWedgeShape interior point failures: " << interior
              << "\n";
    numFailed = kronecker + interior;
  }
  Kokkos::finalize();
  return numFailed ? 1 : 0;
}
