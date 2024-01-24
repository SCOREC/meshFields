#include "MeshField.hpp"
#include "CabanaController.hpp"
#include "KokkosController.hpp"
#include "MeshField_Macros.hpp"
#include "MeshField_Utility.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <vector>
#include <iostream>
//#include <initializer_list>
#include <stdio.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;


int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);
  printf("== START testNset ==\n");

  const int N = 10;

  using kok1 = Controller::KokkosController<MemorySpace,ExecutionSpace, int**, int***,int****, int*****>;
  kok1 c1({N,N,
            N,N,N,
            N,N,N,N,
            N,N,N,N,N});
  assert(c1.size(2,0) == N);
  assert(c1.size(2,1) == N);
  assert(c1.size(2,2) == N);
  assert(c1.size(2,3) == N);
  
  printf("== END testNset ==\n");
  return 0;
}

