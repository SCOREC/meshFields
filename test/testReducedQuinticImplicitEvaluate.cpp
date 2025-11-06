#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <MeshField_Integrate.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

template <size_t dim>
Omega_h::Mesh createMesh(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  const auto numEnts3d = (dim == 3 ? 3 : 0);
  return Omega_h::build_box(world, family, len, len, len, 2, 2, numEnts3d);
}

template <template <typename...> typename Controller, size_t dim>
void runReducedQuinticEvaluate(Omega_h::Mesh &mesh,
                               MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf) {
  const auto ShapeOrder = 5;
  auto field = omf.getCoordField();

  const auto [shp, map] = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  MeshField::FieldElement fes(mesh.nelems(), field, shp, map);

  MeshField::ReducedQuinticImplicitShape shapeFn;
  MeshField::Vector3 xi = {1.0 / 3.0, 1.0 / 3.0, 0.0};
  auto N = shapeFn.getValues(xi);
  auto dN = shapeFn.getLocalGradients(xi);

  double sumN = 0.0;
  for (auto v : N)
    sumN += v;

  double sumDx = 0.0, sumDy = 0.0;
  auto grads = shp.getLocalGradients(xi);
  for (int i = 0; i < MeshField::ReducedQuinticImplicitShape::numNodes; ++i) {
    const auto& g = grads[i];  // reference to Kokkos::Array<double,2>
    sumDx += g[0];
    sumDy += g[1];
  }
  printf("Gradient sums: dx=%f dy=%f\n", sumDx, sumDy);

  const double tol = 1e-10;
  if (std::fabs(sumN - 1.0) > tol) {
    std::cerr << "Shape function partition-of-unity test failed: sumN=" << sumN
              << " (expected 1.0)\n";
    assert(false);
  }
  if (std::fabs(sumDx) > tol || std::fabs(sumDy) > tol) {
    std::cerr << "Gradient sum test failed: d(sumN)/dx=" << sumDx
              << " d(sumN)/dy=" << sumDy << " (expected 0.0)\n";
    assert(false);
  }

  std::cout << "[ReducedQuinticImplicitEvaluate] sumN=" << sumN
            << " dSum=(" << sumDx << "," << sumDy << ")\n";
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  Omega_h::Library lib(&argc, &argv);

  {
    auto mesh2D = createMesh<2>(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::KokkosController> omf2D(mesh2D);
    runReducedQuinticEvaluate<MeshField::KokkosController>(mesh2D, omf2D);
  }

  Kokkos::finalize();
  return 0;
}

