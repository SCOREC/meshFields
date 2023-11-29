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

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);
  const int N = 10;
  using kok1 = Controller::KokkosController<MemorySpace,ExecutionSpace,int*,int**,int***,int****,int*****>;
  kok1 c1({N,N,N,N,N,N,N,N,N,N,N,N,N,N,N});
  
  MeshField::MeshField mf(c1);
  MeshField::Field field1 = mf.makeField<0>();
  MeshField::Field field2 = mf.makeField<1>();
  MeshField::Field field3 = mf.makeField<2>();
  MeshField::Field field4 = mf.makeField<3>();
  MeshField::Field field5 = mf.makeField<4>();

  Kokkos::View<int*> view1("1",N);
  Kokkos::View<int**> view2("2",N,N);
  Kokkos::View<int***> view3("3",N,N,N);
  Kokkos::View<int****> view4("4",N,N,N,N);
  Kokkos::View<int*****> view5("5",N,N,N,N,N);

  Kokkos::Array start = MeshFieldUtil::to_kokkos_array<5>({0,0,0,0,0});
  Kokkos::Array end = MeshFieldUtil::to_kokkos_array<5>({N,N,N,N,N});
  Kokkos::MDRangePolicy<Kokkos::Rank<5>> p(start,end);

  Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l, const int& m){
    view1(i) = i;
    view2(i,j) = i+j;
    view3(i,j,k) = i+j+k;
    view4(i,j,k,l) = i+j+k+l;
    view5(i,j,k,l,m) = i+j+k+l+m;
  });

  mf.setField(field1,view1);
  mf.setField(field2,view2);
  mf.setField(field3,view3);
  mf.setField(field4,view4);
  mf.setField(field5,view5);
  field1.serialize();
  field2.serialize();
  field3.serialize();
  return 0;
}
