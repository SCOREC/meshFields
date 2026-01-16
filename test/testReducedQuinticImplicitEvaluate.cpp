#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_simplex.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <sstream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

/**
 * Evaluator (shape-function) sanity test for ReducedQuinticImplicitShape.
 *
 * What this checks:
 *   - Partition of unity at a representative interior point:
 *       sum_i N_i(xi) == 1
 *
 * Notes:
 *   - We evaluate at the barycentric centroid xi = (1/3, 1/3, 1/3).
 *   - We also compute local gradients to ensure the gradient path compiles/works
 */
void runEvaluateTest(Omega_h::Mesh& mesh) {
  // Get the reduced-quintic implicit element definition for this mesh.
  // The shape object provides shape values/gradients in barycentric coordinates.
  auto element = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  auto& rqShape = element.shp;

  constexpr int nn = static_cast<int>(MeshField::ReducedQuinticImplicitShape::numNodes);

  // Barycentric coordinates for the triangle centroid.
  // Convention: xi entries are nonnegative and sum to 1.
  const MeshField::Vector3 xi = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

  // Evaluate shape functions and local gradients at the centroid.
  const auto N  = rqShape.getValues(xi);
  (void)rqShape.getLocalGradients(xi); // computed for coverage; not asserted here

  constexpr double tol = 1e-10;

  // Partition of unity: Σ N_i(xi) should equal 1.
  double sumN = 0.0;
  for (int i = 0; i < nn; ++i) sumN += static_cast<double>(N[i]);

  if (std::fabs(sumN - 1.0) > tol) {
    std::stringstream ss;
    ss << "[FAIL] ReducedQuinticImplicit partition-of-unity violated at centroid: "
       << "sum(N)=" << sumN << ", tol=" << tol << "\n";
    MeshField::fail(ss.str());
  }

  std::cout << "[PASS] ReducedQuinticImplicit evaluator test (partition of unity)\n";
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Omega_h::Library lib(&argc, &argv);

    // Build a single 2D simplex (triangle) mesh:
    // vertices: (0,0), (1,0), (0,1)
    Omega_h::Mesh mesh2D(&lib);
    Omega_h::Reals coords({0.0, 0.0,
                           1.0, 0.0,
                           0.0, 1.0});
    Omega_h::LOs tris_to_verts({0, 1, 2});

    Omega_h::build_from_elems_and_coords(
      &mesh2D, OMEGA_H_SIMPLEX, /*dim=*/2, tris_to_verts, coords
    );

    runEvaluateTest(mesh2D);
  }
  Kokkos::finalize();
  return 0;
}

