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

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

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
    double result = 0, verify = 0;
    auto reduce_kernel = KOKKOS_LAMBDA( const int &i, double& lsum ) {
      lsum += i * 1.0;
    };
    cabanaMeshField.parallel_reduce("CabanaReduceTest1", {0}, {N}, reduce_kernel, result );
    for( int i = 0; i < N; i++ ) verify+=i*1.0;
    assert(doubleCompare(verify, result));
  }
  {
    double result = 0, verify = 0;
    auto reduce_kernel = KOKKOS_LAMBDA( const int &i, const int& j, double& lsum ) {
      lsum += i * j;
    };
    cabanaMeshField.parallel_reduce("CabanaReduceTest2", {0,0}, {N,N}, reduce_kernel, result );
    for( int i = 0; i < N; i++ ) {
      for( int j = 0; j < N; j++ ) {
        verify += i * j;
      }
    }
    assert(doubleCompare(verify, result));
  }
  {
    double result = 0, verify = 0;
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

    assert(doubleCompare(verify, result));
  }
  {
    double result = 0, verify = 0;
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
    assert(doubleCompare(verify, result));
  }
  printf("== END testParallelReduceCabana ==\n");
}

void testCabanaControllerSize() {

  printf("== START testCabanaControllerSize ==\n");

  const int a=6,b=5,c=4,d=3;
  const int psi[4] = {a,b,c,d};

  using simple = Controller::CabanaController<ExecutionSpace,MemorySpace,int[b]>;
  using multi = Controller::CabanaController<ExecutionSpace,MemorySpace,int[b][c][d],char[b][c][d], bool[b][c][d]>;
  using varied = Controller::CabanaController<ExecutionSpace,MemorySpace,double[b][c], int, float[b][c][d], char[b]>;
  using empty = Controller::CabanaController<ExecutionSpace,MemorySpace, int>;

  simple c1(a);
  multi c2(a);
  varied c3(a);
  empty c4;

  MeshField::MeshField<simple> simple_kok(c1);
  MeshField::MeshField<multi> multi_kok(c2);
  MeshField::MeshField<varied> varied_kok(c3);
  MeshField::MeshField<empty> empty_kok(c4);
  
  // simple_kok
  for( int i = 0; i < 2; i++ ) { assert( simple_kok.size(0,i) == psi[i] ); }
  
  // multi_kok
  for( int i = 0; i < 3; i++ ) {
    for( int j = 0; j < 4; j++ ) {
      assert( multi_kok.size(i,j) == psi[j] );
    }
  } 

  // varied_kok
  for( int i = 0; i < 3; i++ ) assert( varied_kok.size(0,i) == psi[i] );
                               assert( varied_kok.size(1,0) == psi[0] );
  for( int i = 0; i < 4; i++ ) assert( varied_kok.size(2,i) == psi[i] );
  for( int i = 0; i < 2; i++ ) assert( varied_kok.size(3,i) == psi[i] );
  
  // empty_kok
  for( int i = 0; i < 4; i++ ) {
    assert( empty_kok.size(0,i) == 0 );
  }
  printf("== END testCabanaControllerSize ==\n");
}

void testCabanaFieldSize() {
  printf("== START testCabanaFieldSize ==\n");

  const int a=6,b=5,c=4,d=3;
  const int psi[4] = {a,b,c,d};

  using simple = Controller::CabanaController<ExecutionSpace,MemorySpace,int[b]>;
  using multi = Controller::CabanaController<ExecutionSpace,MemorySpace,int[b][c][d],char[b][c][d], bool[b][c][d]>;
  using varied = Controller::CabanaController<ExecutionSpace,MemorySpace,double[b][c], int, float[b][c][d], char[b]>;
  using empty = Controller::CabanaController<ExecutionSpace,MemorySpace, int>;
  simple c1(a);
  multi c2(a);
  varied c3(a);
  empty c4;
  MeshField::MeshField<simple> simple_kok(c1);
  MeshField::MeshField<multi> multi_kok(c2);
  MeshField::MeshField<varied> varied_kok(c3);
  MeshField::MeshField<empty> empty_kok(c4);
  
  const int MAX_RANK = 4;

  { // simple_kok
    auto field0 = simple_kok.makeField<0>();
    for( int i = 0; i < 2; i++ ) assert( field0.size(i) == psi[i] );
  }

  { // multi_kok
    auto field0 = multi_kok.makeField<0>();
    auto field1 = multi_kok.makeField<1>();
    auto field2 = multi_kok.makeField<2>();
    for( int i = 0; i < MAX_RANK; i++ ) {
      assert( field0.size(i) == psi[i] );
      assert( field1.size(i) == psi[i] );
      assert( field2.size(i) == psi[i] );
    }
  }

  { // varied_kok
    auto field0 = varied_kok.makeField<0>();
    auto field1 = varied_kok.makeField<1>();
    auto field2 = varied_kok.makeField<2>();
    auto field3 = varied_kok.makeField<3>();

    for( int i = 0; i < 3; i++ ) { assert( field0.size(i) == psi[i] ); }
    for( int i = 0; i < 1; i++ ) { assert( field1.size(i) == psi[i] ); }
    for( int i = 0; i < 4; i++ ) { assert( field2.size(i) == psi[i] ); }
    for( int i = 0; i < 2; i++ ) { assert( field3.size(i) == psi[i] ); }
  }
  { // empty_kok
    auto field0 = empty_kok.makeField<0>();
    for( int i = 0; i < MAX_RANK; i++ ) { assert( field0.size(i) == 0 ); }
  }
  printf("== END testCabanaFieldSize ==\n");
}

void testCabanaParallelFor() {
  printf("== START testCabanaParallelFor() ==\n");
  const int x=10,y=9,z=8,a=7;
  { 
    using simd_ctrlr = Controller::CabanaController<ExecutionSpace,MemorySpace,int,int[y],
          int[y][z], int[y][z][a]>;
    simd_ctrlr c1(x);
    MeshField::MeshField<simd_ctrlr> mf(c1);
    auto field0 = mf.makeField<0>();
    auto field1 = mf.makeField<1>();
    auto field2 = mf.makeField<2>();
    auto field3 = mf.makeField<3>();

    auto vectorKernel = KOKKOS_LAMBDA( const int& i ) {
      field0(i) = i;
      assert(field0(i) == i);
    };
    mf.parallel_for( {0},{x}, vectorKernel, "simple_loop");
    
    auto vectorKernel2 = KOKKOS_LAMBDA( const int& i, const int& j ) {
      field1(i,j) = i+j;
      assert(field1(i,j) == i+j);
    };
    mf.parallel_for( {0,0},{x,y}, vectorKernel2, "simple_loop");

    auto vectorKernel3 = KOKKOS_LAMBDA( const int& i, const int& j, const int& k ) {
      field2(i,j,k) = i+j+k;
      assert( field2(i,j,k) == i+j+k );
    };
    mf.parallel_for( {0,0,0},{x,y,z}, vectorKernel3, "simple_loop");


    auto vectorKernel4 = KOKKOS_LAMBDA( const int& i, const int& j, const int& k,
                                        const int& l) {
      field3(i,j,k,l) = i+j+k+l;
      assert( field3(i,j,k,l) == i+j+k+l );
    };
    mf.parallel_for( {0,0,0,0},{x,y,z,a}, vectorKernel4, "simple_loop");
    
  }

  printf("== END testCabanaParallelFor() ==\n");
}

int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  testMakeSliceCabana(num_tuples);
  testParallelReduceCabana();
  testCabanaControllerSize();
  testCabanaFieldSize();
  testCabanaParallelFor();
  return 0;
}
