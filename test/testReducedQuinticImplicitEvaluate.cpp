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

void runEvaluateTest(Omega_h::Mesh& mesh) {
  auto element = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  auto& rqShape = element.shp;

  constexpr int nn = MeshField::ReducedQuinticImplicitShape::numNodes;
  constexpr int Order = MeshField::ReducedQuinticImplicitShape::Order;

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

  // Degree-4 Reproduction
  double L1 = 1.0/3.0;
  double L2 = 1.0/3.0;
  double L3 = 1.0/3.0;

  double rL1 = 0.0;
  double rL2 = 0.0;
  double rL3 = 0.0;

  int idx = 0;
  for (int i = 0; i <= Order; ++i) {
    for (int j = 0; j <= Order - i; ++j) {
      int k = Order - i - j;
      rL1 += N[idx] * (double(i)/Order);
      rL2 += N[idx] * (double(j)/Order);
      rL3 += N[idx] * (double(k)/Order);
      ++idx;
    }
  }

  double reconstructed = rL1*rL1 * rL2*rL2 + rL2*rL2 * rL3*rL3 + rL1 * rL3*rL3*rL3;

  double exact = L1*L1 * L2*L2 + L2*L2 * L3*L3 + L1 * L3*L3*L3;

  if (std::fabs(reconstructed - exact) > tol) {
    MeshField::fail("Degree-4 polynomial reproduction failed");
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

