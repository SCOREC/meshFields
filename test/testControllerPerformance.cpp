#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_For.hpp"       
#include "MeshField_SimdFor.hpp"
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif
#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
template <template <typename...> typename Controller>
void doRun(Omega_h::Mesh &mesh, MeshField::OmegahMeshField<ExecutionSpace, Controller> &omf) {
  
}

int main(int argc, char **argv) {
  int n = atoi(argv[1]), runs = atoi(argv[2]);
  std::cout << n << " " << runs << std::endl;
  Kokkos::initialize(argc, argv);
  #ifdef MESHFIELDS_ENABLE_CABANA
  {
  using cab = MeshField::CabanaController<ExecutionSpace, MemorySpace, int[11][12][13]>;
  using kokkos = MeshField::KokkosController<MemorySpace, ExecutionSpace, int****>;
  cab cabCtrlr({n});
  kokkos kokkosCtrlr({n, 11, 12, 13});
  auto cabField = MeshField::makeField<cab, 0>(cabCtrlr);
  auto kokkosField = MeshField::makeField<kokkos, 0>(kokkosCtrlr);
  auto cabDim4 = KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l) {
    cabField(i, j, k, l) = i + j + k + l;
  };
  auto kokkosDim4 = KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l) {
    kokkosField(i, j, k, l) = i + j + k + l;
  };
  for (int i = 0; i < runs; ++i) {
  std::chrono::milliseconds times[2];
  auto start = std::chrono::high_resolution_clock::now();
  MeshField::parallel_for(ExecutionSpace(), {0, 0, 0, 0}, {n, 11, 12, 13}, kokkosDim4, "KokkosTest");
  times[0] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
  start = std::chrono::high_resolution_clock::now();
  MeshField::simd_parallel_for(cabCtrlr, {0, 0, 0, 0}, {n, 11, 12, 13}, cabDim4, "CabTest");
  times[1] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
  std::cout << "Cabana time: " << times[1].count() << " Kokkos time: " << times[0].count() << std::endl;
  }
  }
  Kokkos::finalize();
  #endif
  return 0;
}
