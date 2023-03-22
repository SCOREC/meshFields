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

using ExecutionSpace = Kokkos::Cuda;
using MemorySpace = Kokkos::CudaUVMSpace;

void testMakeSliceCabana( int num_tuples ) {
  printf("== START testMakeSliceCabana ==\n");
  using Ctrl = Controller::CabanaController<ExecutionSpace,MemorySpace,double>;
  Ctrl c( num_tuples );
  MeshField::MeshField<Ctrl> cabanaMeshField(c);

  auto field0 = cabanaMeshField.makeField<0>();

  auto testKernel = KOKKOS_LAMBDA( const int x ) {
    double gamma = (double)x;
    field0(x) = gamma;
    assert(doubleCompare(field0(x),gamma));
  };
  Kokkos::parallel_for("testMakeSliceCabana()", num_tuples, testKernel);

  printf("== END testMakeSliceCabana ==\n");
}

void testParallelReduceCabana() {
  printf("== START testParallelReduceCabana ==\n");
  using Ctrl = Controller::CabanaController<ExecutionSpace,MemorySpace,int>;
  const int N = 9;
  Ctrl c( N );
  MeshField::MeshField<Ctrl> cabanaMeshField(c);
  
  {
    double result, verify;
    auto reduce_kernel = KOKKOS_LAMBDA( const int &i, double& lsum ) {
      lsum += i * 1.0;
    };
    cabanaMeshField.parallel_reduce("CabanaReduceTest1", {0}, {N}, reduce_kernel, result );
    for( int i = 0; i < N; i++ ) verify+=i*1.0;
    assert( verify == result );
  }
  {
    double result, verify;
    auto reduce_kernel = KOKKOS_LAMBDA( const int &i, const int& j, double& lsum ) {
      lsum += i * j;
    };
    cabanaMeshField.parallel_reduce("CabanaReduceTest2", {0,0}, {N,N}, reduce_kernel, result );
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
        verify += i * j;
      }
    }
    assert( verify == result );
  }
  {
    double result, verify;
    auto reduce_kernel = KOKKOS_LAMBDA( const int &i, const int& j, const int& k, double& lsum ) {
      lsum += i * j * k;
    };
    cabanaMeshField.parallel_reduce("CabanaReduceTest3", {0,0,0}, {N,N,N}, reduce_kernel, result );
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
        for( int k = 0; k < N; k++ ) { 
          verify += i * j * k; 
        }
      }
    }
    assert( verify == result );
  }
  {
    double result, verify;
    auto reduce_kernel = KOKKOS_LAMBDA( const int &i, const int& j, const int& k, const int& l, double& lsum ) {
      lsum += i * j * k * l;
    };
    cabanaMeshField.parallel_reduce("CabanaReduceTest4", {0,0,0,0}, {N,N,N,N}, reduce_kernel, result );
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
        for( int k = 0; k < N; k++ ) { 
          for( int l = 0; l < N; l++ ) { 
            verify += i * j * k * l; 
          }
        }
      }
    }
    assert( verify == result );
  }
  printf("== END testParallelReduceCabana ==\n");
}


int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  
  testMakeSliceCabana(num_tuples);
  testParallelReduceCabana();

  return 0;
}
