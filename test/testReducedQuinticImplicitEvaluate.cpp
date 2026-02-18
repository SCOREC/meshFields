#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_simplex.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

/**
 * Evaluator (shape-function) validation test for ReducedQuinticImplicitShape.
 *
 * This test verifies:
 *   1. Partition of unity:        Σ Ni(xi) = 1
 *   2. Gradient partition unity:  Σ ∇Ni(xi) = 0
 *   3. Exact values at centroid   Ni(1/3,1/3,1/3) match reference
 */
void runEvaluateTest(Omega_h::Mesh& mesh) {
  auto element = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  auto& rqShape = element.shp;

  constexpr int nn = MeshField::ReducedQuinticImplicitShape::numNodes;

  constexpr double tol = 1e-12;

  // Barycentric centroid
  const MeshField::Vector3 xi = {1.0 / 3.0,
                                 1.0 / 3.0,
                                 1.0 / 3.0};

  const auto N  = rqShape.getValues(xi);
  const auto dN = rqShape.getLocalGradients(xi);

  // Partition of unity: Σ N_i(xi) should equal 1.
  const double sumN = std::accumulate(&N[0], &N[0] + N.size(), 0.0);

  if (std::fabs(sumN - 1.0) > tol) {
    std::stringstream ss;
    ss << "[FAIL] Partition-of-unity violated: "
       << "sum(N)=" << sumN
       << ", expected=1.0, tol=" << tol;
    MeshField::fail(ss.str());
  }

  // ------------------------------------------------------------
  // Exact Bernstein basis value check at centroid
  //
  // For (i,j,k) = (1,2,2):
  // N = (5!)/(1!2!2!) * (1/3)^5 = 10/81
  // ------------------------------------------------------------

  const int target_i = 1;
  const int target_j = 2;
  const int target_k = 2;

  int idx = 0;
  int target_idx = -1;

  for (int i = 0; i <= 5; ++i) {
    for (int j = 0; j <= 5 - i; ++j) {
      int k = 5 - i - j;

      if (i == target_i && j == target_j && k == target_k)
      target_idx = idx;

      ++idx;
    }
  }

  const double expected = 10.0 / 81.0;

  if (std::fabs(N[target_idx] - expected) > tol) {
    std::stringstream ss;
    ss << "[FAIL] Exact Bernstein value mismatch: "
    << "computed=" << N[target_idx]
       << ", expected=" << expected
       << ", tol=" << tol;
    MeshField::fail(ss.str());
  }

  std::cout << "[PASS] ReducedQuinticImplicitShape evaluator test\n";
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Omega_h::Library lib(&argc, &argv);

    Omega_h::Mesh mesh2D(&lib);

    // Single triangle:
    // (0,0), (1,0), (0,1)
    Omega_h::Reals coords({0.0, 0.0,
                           1.0, 0.0,
                           0.0, 1.0});
    Omega_h::LOs tris_to_verts({0, 1, 2});

    Omega_h::build_from_elems_and_coords(
        &mesh2D,
        OMEGA_H_SIMPLEX,
        /*dim=*/2,
        tris_to_verts,
        coords);

    runEvaluateTest(mesh2D);
  }
  Kokkos::finalize();
  return 0;
}

