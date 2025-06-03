#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_For.hpp"
#include "MeshField_SimdFor.hpp"
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif
#include <Kokkos_Core.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
#ifndef DIM1
#define DIM1 3
#endif
#ifndef DIM2
#define DIM2 3
#endif
int main(int argc, char **argv) {
  int n = atoi(argv[1]), runs = atoi(argv[2]);
  Kokkos::initialize(argc, argv);
#ifdef MESHFIELDS_ENABLE_CABANA
  if (atoi(argv[3])) {
    double avg = 0;
    using cab = MeshField::CabanaController<ExecutionSpace, MemorySpace,
                                            int[DIM1][DIM2]>;
    cab cabCtrlr({n});
    auto cabField = MeshField::makeField<cab, 0>(cabCtrlr);
    auto cabDim4 = KOKKOS_LAMBDA(const int &i, const int &j, const int &k) {
      cabField(i, j, k) = i + j + k;
    };
    for (int i = 0; i < runs; ++i) {
      auto start = std::chrono::system_clock::now();
      MeshField::simd_parallel_for(cabCtrlr, {0, 0, 0}, {n, DIM1, DIM2},
                                   cabDim4, "CabTest");
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - start);
      avg += duration.count();
    }
    avg = avg / runs;
    std::cout << std::fixed << std::setprecision(1) << avg << std::endl;
  } else {
    double avg = 0;
    using kokkos =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int ***>;
    kokkos kokkosCtrlr({n, DIM1, DIM2});
    auto kokkosField = MeshField::makeField<kokkos, 0>(kokkosCtrlr);
    auto kokkosDim4 = KOKKOS_LAMBDA(const int &i, const int &j, const int &k) {
      kokkosField(i, j, k) = i + j + k;
    };
    for (int i = 0; i < runs; ++i) {
      auto start = std::chrono::system_clock::now();
      MeshField::parallel_for(ExecutionSpace(), {0, 0, 0}, {n, DIM1, DIM2},
                              kokkosDim4, "KokkosTest");
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - start);
      avg += duration.count();
    }
    avg = avg / runs;
    std::cout << std::fixed << std::setprecision(1) << avg << std::endl;
  }

  Kokkos::finalize();
#endif
  return 0;
}
