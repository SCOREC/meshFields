#include "MeshField.hpp"
#include "CabanaController.hpp"
#include "KokkosController.hpp"
#include "MeshField_Macros.hpp"
#include "MeshField_Utility.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdio.h>

#define TOLERANCE 1e-10;

// helper testing functions

// compare doubles within a tolerance
KOKKOS_INLINE_FUNCTION
bool doubleCompare(double d1, double d2) {
  double diff = fabs(d1 - d2);
  return diff < TOLERANCE;
}

// returns (x-1)*(x/2) = 1 + 2 + 3 + 4 + ... + x
KOKKOS_INLINE_FUNCTION
int seriesSum( int x ) {
    return (int)(((double)x-1.0))*(((double)x/2.0));
}

using ExecutionSpace = Kokkos::Cuda;
using MemorySpace = Kokkos::CudaUVMSpace;

void testParallelScan() {
  printf("== START testParallelScan ==\n");

  const int N = 100;

  printf("-- start testParallelScan on Kokkos --\n");
  {
    using s_kok = Controller::KokkosController<MemorySpace,ExecutionSpace, int*,int*>;
    s_kok c1({N,N});
    MeshField::MeshField<s_kok> mfk(c1);
    auto pre = mfk.makeField<0>();
    auto post = mfk.makeField<1>();
    
    auto scan_kernel = KOKKOS_LAMBDA(int i, int& partial_sum, const bool is_final) {
      if( is_final ) pre(i) = partial_sum;
      partial_sum += i;
      if( is_final ) post(i) = partial_sum;
    };
  
    for( int i = 1; i <= N; i++ ) {
      int result;
      mfk.parallel_scan("default", {0}, {i}, scan_kernel, result );
      assert( result == seriesSum(i) );
    }
  }
  printf("-- end testParallelScan on Kokkos --\n");
  printf("-- start testParallelScan on Cabana --\n");
  {
    using s_cab = Controller::CabanaController<ExecutionSpace,MemorySpace, int,int>;
    s_cab c1(N);
    MeshField::MeshField<s_cab> mfc(c1);
    auto pre = mfc.makeField<0>();
    auto post = mfc.makeField<1>();
    
    auto scan_kernel = KOKKOS_LAMBDA(int i, int& partial_sum, const bool is_final) {
      if( is_final ) pre(i) = partial_sum;
      partial_sum += i;
      if( is_final ) post(i) = partial_sum;
    };
  
    for( int i = 1; i <= N; i++ ) {
      int result;
      mfc.parallel_scan("default", {0}, {i}, scan_kernel, result );
      assert( result == seriesSum(i) );
    }
  }
  printf("-- end testParallelScan on Cabana --\n");
  printf("== END testParallelScan ==\n");
}

int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  
  testParallelScan();

  return 0;
}
