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

void testMakeSliceKokkos( int num_tuples ) {
  printf("== START testMakeSliceKokkos ==\n");
  int N = 10;
  using Ctrlr = Controller::KokkosController<MemorySpace,ExecutionSpace,double*>;
  Ctrlr c({10});
  MeshField::MeshField<Ctrlr> kokkosMeshField(c);

  auto field0 = kokkosMeshField.makeField<0>();
  
  auto testKernel = KOKKOS_LAMBDA( const int x ) {
    double gamma = (double)x;
    field0(x) = gamma;
    assert(doubleCompare(field0(x),gamma));
  };

  Kokkos::parallel_for("testMakeSliceKokkos()", N, testKernel);
  printf("== END testMakeSliceKokkos ==\n");
}

void testKokkosConstructor( int num_tuples ) {
  printf("== START testKokkosConstructor ==\n");
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,ExecutionSpace,double**[3]>;
    Ctrlr c({num_tuples,num_tuples});
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,ExecutionSpace,double[3]>;
    Ctrlr c;
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,ExecutionSpace,int*****>;
    Ctrlr c({10,10,10,10,10});
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,ExecutionSpace,int>;
    Ctrlr c;
    MeshField::MeshField<Ctrlr> kok(c);
  }

  printf("== END testKokkosConstructor ==\n");
}


void testingStufffs() {

  printf("== START testingStufffs ==\n");
  using Ctrlr = Controller::KokkosController<MemorySpace,ExecutionSpace,int*>;
  Ctrlr c({10});
  MeshField::MeshField<Ctrlr> kok(c);
  
  auto field0 = kok.makeField<0>();

  auto vectorKernel = KOKKOS_LAMBDA(const int &s) {
    field0(s) = 3;
  };
  Kokkos::parallel_for("tag", 10, vectorKernel );

  printf("== END testingStufffs ==\n");
}


void testKokkosParallelFor() {

  printf("== START testKokkosParallelFor ==\n");

  using Ctrlr = Controller::KokkosController<MemorySpace, ExecutionSpace, int**>;
  Ctrlr c({10,10});
  MeshField::MeshField<Ctrlr> kok(c);
  
  auto field0 = kok.makeField<0>();

  auto vectorKernel = KOKKOS_LAMBDA (const int s, const int a) {
    field0(s,a) = s+a;
    assert(field0(s,a) == s+a);
  };

  kok.parallel_for({0,0},{10,10},vectorKernel, "testKokkosParallelFor()");

  printf("== END testKokkosParallelFor ==\n");
}

void kokkosDocumentationLiesTest() { // They dont...
  printf("== START kokkosDocumentationLiesTest ==\n");
  Kokkos::Array<int64_t,3> start = {0,0,0};
  Kokkos::Array<int64_t,3> end = {2,2,2};
  Kokkos::parallel_for("0_0", Kokkos::MDRangePolicy< Kokkos::Rank<3> >(start,end),
    KOKKOS_LAMBDA (const int c, const int f, const int p) {
      printf("Kokkos documentation lies!!!: c:%d f:%d p:%d\n",c,f,p);
    });
  printf("== END kokkosDocumentationLiesTest ==\n");
}

void kokkosParallelReduceTest() {
  /* Examples from Kokkos Documentation:
   * https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html?highlight=parallel_reduce*/

  printf("== START kokkosParallelReduceTest ==\n");
  using Ctrlr = Controller::KokkosController<MemorySpace,ExecutionSpace,int*>;
  Ctrlr c1({10});
  MeshField::MeshField<Ctrlr> kok(c1);

  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA( const int& i, double& lsum ) {
      lsum += 1.0 * i;
    };
    kok.parallel_reduce("ReduceTest",{0},{N},kernel,result);
    double result_verify = 0;
    for( int i = 0; i < N; i++ ) {
      result_verify += 1.0*i;
    }
    assert( result_verify == result );
    printf("Reduce test 1-D Result: %d %.2lf\n",N,result);
  }
  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA( const int& i, const int&j, double& lsum ) {
      lsum += i * j;
    };
    kok.parallel_reduce("ReduceTest2",{0,0},{N,N},kernel,result);
    double result_verify = 0;
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
        result_verify += i*j;
      }
    }
    assert( result_verify == result );

    printf("Reduce test 2-D Result: %d %.2lf\n",N,result);
  }
  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA( const int& i, const int& j, const int& k, double& lsum ) {
      lsum += i * j * k;
    };
    kok.parallel_reduce("ReduceTest3",{0,0,0},{N,N,N},kernel,result);
    double result_verify = 0;
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
        for( int k = 0; k < N; k++ ) {
          result_verify += i*j*k;
        }
      }
    }
    assert( result_verify == result );
    printf("Reduce test 3-D Result: %d %.2lf\n",N,result);
  }
  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA( const int& i, const int& j, const int& k, const int& l, double& lsum ) {
      lsum += i * j * k * l;
    };
    kok.parallel_reduce("ReduceTest4",{0,0,0,0},{N,N,N,N},kernel,result);
    double result_verify = 0;
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
        for( int k = 0; k < N; k++ ) {
          for( int l = 0; l < N; l++ ) {
            result_verify += i*j*k*l;
          }
        }
      }
    }
    assert( result_verify == result );
    printf("Reduce test 4-D Result: %d %.2lf\n",N,result);
  }

  printf("== END kokkosParallelReduceTest ==\n");
}

void kokkosSizeTest() {
  printf("== START kokkosSizeTest ==\n");
  const int a = 5;
  const int b = 4;
  const int c = 3;
  const int d = 2;
  const int e = 1;

  using Ctrlr1 = Controller::KokkosController<MemorySpace,ExecutionSpace,int*, double[a]>;
  using Ctrlr2 = Controller::KokkosController<MemorySpace,ExecutionSpace,int****[e], double[a][b][c][d][e]>;
  Ctrlr1 c1({5});
  Ctrlr2 c2({5,4,3,2});
  MeshField::MeshField<Ctrlr1> kok1(c1); // sizes ->[[5,0,0,0,0],[5,0,0,0,0,0]]
  MeshField::MeshField<Ctrlr2> kok2(c2); // sizes -> [[5,4,3,2,1]]

  for( int i = 0; i < 2; i++ ) {
    for( int j = 0; j < 5; j++ ) {
      printf("kok1.size(%d,%d) = %d\n",i,j,kok1.size(i,j));
      //assert( kok1_sizes[i][j] == psi[j] );
    }
  }


  printf("== END kokkosSizeTest ==\n");
}

int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  
  testKokkosConstructor(num_tuples);
  testKokkosParallelFor();
  kokkosParallelReduceTest();
  testMakeSliceKokkos(num_tuples);
  kokkosSizeTest();

  //kokkosDocumentationLiesTest();
  return 0;
}
