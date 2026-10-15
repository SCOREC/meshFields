#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Shape.hpp>
#include <iostream>

// Checks of MeshField::LinearCubicWedgeShape, linear in (xi0, xi1) and cubic
// in zeta. The checks run inside Kokkos kernels so the shape functions are
// exercised in the default execution space, and failures are counted
// explicitly so the test also fails in builds where assert() is disabled.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;
using Shape = MeshField::LinearCubicWedgeShape;

KOKKOS_INLINE_FUNCTION bool isClose(Real a, Real b, Real tol) {
  return Kokkos::fabs(a - b) <= tol;
}

// A function in the span of the cubic wedge basis,
// {1, xi0, xi1} x {1, zeta, zeta^2, zeta^3}, and its gradient
KOKKOS_INLINE_FUNCTION Real f(Real x0, Real x1, Real z) {
  return 0.5 + 2 * x0 - 3 * x1 + 0.25 * z - 1.5 * z * z + 0.75 * z * z * z +
         4 * x0 * z - 1.5 * x1 * z * z - 2 * x0 * z * z * z + x1 * z * z * z;
}
KOKKOS_INLINE_FUNCTION MeshField::Vector3 gradF(Real x0, Real x1, Real z) {
  return {2 + 4 * z - 2 * z * z * z,
          -3 - 1.5 * z * z + z * z * z,
          0.25 - 3 * z + 2.25 * z * z + 4 * x0 - 3 * x1 * z - 6 * x0 * z * z +
              3 * x1 * z * z};
}

// Each shape function is one at its own node and zero at the others.
int checkKroneckerDelta() {
  int numFailed = 0;
  Kokkos::parallel_reduce(
      "cubicWedgeKronecker",
      Kokkos::RangePolicy<ExecutionSpace>(0, Shape::numNodes),
      KOKKOS_LAMBDA(const int node, int &failed) {
        const Shape shp;
        const auto coords = shp.getNodeParametricCoords();
        const MeshField::Vector3 xi{coords[3 * node], coords[3 * node + 1],
                                    coords[3 * node + 2]};
        const auto N = shp.getValues(xi);
        for (size_t m = 0; m < Shape::numNodes; ++m) {
          const Real expected = (static_cast<size_t>(node) == m) ? 1.0 : 0.0;
          failed += !isClose(N[m], expected, 10 * MeshField::MachinePrecision);
        }
      },
      numFailed);
  return numFailed;
}

// Interior points, including points at the zeta stations and between them.
Kokkos::View<Real *[3], ExecutionSpace> makeInteriorPoints() {
  // clang-format off
  const Real pts[][3] = {{1. / 3,  1. / 3,   0.0},
                         {0.2,     0.3,     -0.4},
                         {0.6,     0.1,      0.7},
                         {0.05,    0.9,      0.95},
                         {0.45,    0.45,    -0.95},
                         {0.2,     0.3,      1. / 3},   // at a station
                         {0.6,     0.1,     -1. / 3},   // at a station
                         {0.3,     0.2,     -0.7},      // between -1 and -1/3
                         {0.1,     0.8,      0.5}};     // between 1/3 and 1
  // clang-format on
  const int numPts = static_cast<int>(sizeof(pts) / sizeof(pts[0]));
  Kokkos::View<Real *[3], ExecutionSpace> pts_d("cubicWedgePts", numPts);
  auto pts_h = Kokkos::create_mirror_view(pts_d);
  for (int i = 0; i < numPts; ++i)
    for (int d = 0; d < 3; ++d)
      pts_h(i, d) = pts[i][d];
  Kokkos::deep_copy(pts_d, pts_h);
  return pts_d;
}

// At interior points: the values sum to one, the gradients sum to zero, the
// basis reproduces f and its gradient, and the analytic gradients match
// central finite differences of the values.
int checkInteriorPoints() {
  const auto pts_d = makeInteriorPoints();
  const int numPts = static_cast<int>(pts_d.extent(0));

  int numFailed = 0;
  Kokkos::parallel_reduce(
      "cubicWedgeInterior", Kokkos::RangePolicy<ExecutionSpace>(0, numPts),
      KOKKOS_LAMBDA(const int i, int &failed) {
        const Real tol = 10 * MeshField::MachinePrecision;
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
        failed += !isClose(sumN, 1.0, tol);
        failed += !isClose(fInterp, f(xi[0], xi[1], xi[2]), tol);
        const auto gExact = gradF(xi[0], xi[1], xi[2]);
        for (int d = 0; d < 3; ++d) {
          failed += !isClose(sumG[d], 0.0, tol);
          failed += !isClose(gInterp[d], gExact[d], tol);
        }

        // gradients against central finite differences of the values
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

// Runs both checks and prints their failure counts. 
int runCubicWedgeShapeChecks() {
  const int kronecker = checkKroneckerDelta();
  std::cout << "LinearCubicWedgeShape Kronecker delta failures: " << kronecker
            << "\n";
  const int interior = checkInteriorPoints();
  std::cout << "LinearCubicWedgeShape interior point failures: " << interior
            << "\n";
  return kronecker + interior;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  const int numFailed = runCubicWedgeShapeChecks();
  Kokkos::finalize();
  return numFailed ? 1 : 0;
}
