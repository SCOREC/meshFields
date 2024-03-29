#include "test.hpp"
#include "CabanaController.hpp"
#include "MeshField.hpp"
#include "MeshField_Macros.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdio>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

int main(int argc, char **argv) {
  Kokkos::ScopeGuard scope_gaurd(argc, argv);
  /*
  auto rank1 = KOKKOS_LAMBDA(const int& i) {
    printf("I am rank 1:(%d)\n",i);
  };
  auto rank2 = KOKKOS_LAMBDA(const int& i, const int& j) {
    printf("I am rank 2: (%d,%d)\n",i,j);
  };
  */
  // parallel_for2(rank1);
  // parallel_for2(rank2);

  // parallel_for( rank2, {0,0}, {5,5} );
  // parallel_for( rank1, {0}, {5} );
  // simd_for( rank1, {0}, {10} );
  // simd_for( rank2, {0,0}, {10,10} );
  const int x = 100;
  using ctrl =
      Controller::CabanaController<ExecutionSpace, MemorySpace, int[x]>;
  ctrl c1(x);
  MeshField::MeshField<ctrl> mf(c1);
  auto field0 = mf.makeField<0>();
  auto vKernel = KOKKOS_LAMBDA(const int &i, const int &j) {
    printf("vKernel -> (%d,%d)\n", i, j);
    field0(i, j) = i + j;
    assert(field0(i, j) == i + j);
  };
  simd_for<ctrl::vecLen>(vKernel, {0, 0}, {x, x});
}
