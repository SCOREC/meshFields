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

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

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
      mfk.parallel_scan("default", 0, i, scan_kernel, result );
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
      mfc.parallel_scan("default", 0, i, scan_kernel, result );
      assert( result == seriesSum(i) );
    }
  }
  printf("-- end testParallelScan on Cabana --\n");
  printf("== END testParallelScan ==\n");
}

void testSetField() {
  printf("== START testSetField ==\n");

  { // KOKKOS FIELD
    const int N = 10;
    using kok1 = Controller::KokkosController<MemorySpace,ExecutionSpace,int*,int**,int***,int****,int*****>;
    kok1 c1({N,N,N,N,N,N,N,N,N,N,N,N,N,N,N});
    MeshField::MeshField mf(c1);
    auto f1 = mf.makeField<0>();
    auto f2 = mf.makeField<1>();
    auto f3 = mf.makeField<2>();
    auto f4 = mf.makeField<3>();
    auto f5 = mf.makeField<4>();

    Kokkos::View<int*> v1("1",N);
    Kokkos::View<int**> v2("2",N,N);
    Kokkos::View<int***> v3("3",N,N,N);
    Kokkos::View<int****> v4("4",N,N,N,N);
    Kokkos::View<int*****> v5("5",N,N,N,N,N);

    Kokkos::Array start = MeshFieldUtil::to_kokkos_array<5>({0,0,0,0,0});
    Kokkos::Array end = MeshFieldUtil::to_kokkos_array<5>({N,N,N,N,N});
    Kokkos::MDRangePolicy<Kokkos::Rank<5>> p(start,end);

    Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l, const int& m){
      v1(i) += i;
      v2(i,j) += i+j;
      v3(i,j,k) += i+j+k;
      v4(i,j,k,l) += i+j+k+l;
      v5(i,j,k,l,m) += i+j+k+l+m;
    });

    mf.setField(f1,v1); 
    mf.setField(f2,v2); 
    mf.setField(f3,v3); 
    mf.setField(f4,v4); 
    mf.setField(f5,v5); 
    

    Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l, const int& m){
      assert( f1(i) == v1(i) );
      assert( f2(i,j) == v2(i,j) );
      assert( f3(i,j,k) == v3(i,j,k) );
      assert( f4(i,j,k,l) == v4(i,j,k,l) );
      assert( f5(i,j,k,l,m) == v5(i,j,k,l,m) );

    });
  }
  { // CABANA FIELD
    const int N = 10;
    using cab1 = Controller::CabanaController<ExecutionSpace,MemorySpace,int,int[N],int[N][N],int[N][N][N]>;
    cab1 c1(N);
    MeshField::MeshField<cab1> mf(c1);
    auto f1 = mf.makeField<0>();
    auto f2 = mf.makeField<1>();
    auto f3 = mf.makeField<2>();
    auto f4 = mf.makeField<3>();

    Kokkos::View<int*> v1("1",N);
    Kokkos::View<int**> v2("2",N,N);
    Kokkos::View<int***> v3("3",N,N,N);
    Kokkos::View<int****> v4("4",N,N,N,N);

    Kokkos::Array start = MeshFieldUtil::to_kokkos_array<4>({0,0,0,0});
    Kokkos::Array end = MeshFieldUtil::to_kokkos_array<4>({N,N,N,N});
    Kokkos::MDRangePolicy<Kokkos::Rank<4>> p(start,end);

    Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l){
      v1(i) += i;
      v2(i,j) += i+j;
      v3(i,j,k) += i+j+k;
      v4(i,j,k,l) += i+j+k+l;
    });

    mf.setField(f1,v1); 
    mf.setField(f2,v2); 
    mf.setField(f3,v3); 
    mf.setField(f4,v4); 
    

    Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l){
      assert( f1(i) == v1(i) );
      assert( f2(i,j) == v2(i,j) );
      assert( f3(i,j,k) == v3(i,j,k) );
      assert( f4(i,j,k,l) == v4(i,j,k,l) );

    });
  }


  printf("== END testSetField ==\n");
}

void testSetCorrect() {
  
  printf("== START testSetCorrect ==\n");

  const int N = 10;

  using kok1 = Controller::KokkosController<MemorySpace,ExecutionSpace, int**, int***,int****, int*****>;
  kok1 c1({N,1,
            N,N,N,
            N,N,15,N,
            N,N,N,N,6});

  // Checking that sizes are loaded correctly
  assert(c1.size(0,1) == 1);
  assert(c1.size(2,2) == 15);
  assert(c1.size(3,4) == 6);
  
  printf("== END testSetCorrect ==\n");
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);
  testParallelScan();
  testSetField();
  testSetCorrect();
  return 0;
}
