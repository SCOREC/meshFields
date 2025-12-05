#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

void runEvaluateTest(Omega_h::Mesh &mesh) {
  auto element = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  auto &rqShape = element.shp;

  constexpr int nn = MeshField::ReducedQuinticImplicitShape::numNodes;

  MeshField::Vector3 xi = {1.0/3.0, 1.0/3.0, 1.0/3.0};

  auto N = rqShape.getValues(xi);

  auto dN = rqShape.getLocalGradients(xi);

  const double tol = 1e-10;

  double sumN = 0.0;
  for (int i = 0; i < nn; ++i) {
    sumN += N[i];
  }

  if (std::fabs(sumN - 1.0) > tol) {
    std::stringstream ss;
    ss << "[FAIL] Partition-of-unity violated: sum(N)=" << sumN << "\n";
    MeshField::fail(ss.str());
  }

  double gx = 0.0, gy = 0.0;

  for (int i = 0; i < nn; ++i) {
    gx += dN[i][0];
    gy += dN[i][1];
  }

  if (std::fabs(gx) > tol || std::fabs(gy) > tol) {
    std::stringstream ss;
    ss << "[FAIL] Gradient sum violated: gx=" << gx << ", gy=" << gy << "\n";
    MeshField::fail(ss.str());
  }

  std::cout << "[PASS] ReducedQuinticImplicit Evaluate test\n";
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    Omega_h::Library lib(&argc, &argv);

    Omega_h::Mesh mesh2D(&lib);

    Omega_h::Reals coords({0.0,0.0, 1.0,0.0, 0.0,1.0});
    Omega_h::LOs tris_to_verts({0,1,2});

    Omega_h::build_from_elems_and_coords(
        &mesh2D, OMEGA_H_SIMPLEX, 2, tris_to_verts, coords);

    runEvaluateTest(mesh2D);
  }

  Kokkos::finalize();
  return 0;
}

