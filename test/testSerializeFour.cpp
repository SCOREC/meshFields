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

  const int N = 30;
  using kok1 = Controller::KokkosController<MemorySpace,ExecutionSpace,int*,int**,int***,int****,int*****>;
  kok1 c1({N,N,N,N,N,N,N,N,N,N,N,N,N,N,N});
  
  MeshField::MeshField mf(c1);
  MeshField::Field field4 = mf.makeField<3>();

  Kokkos::View<int****> view4("4",N,N,N,N);

  Kokkos::Array start = MeshFieldUtil::to_kokkos_array<5>({0,0,0,0,0});
  Kokkos::Array end = MeshFieldUtil::to_kokkos_array<5>({N,N,N,N,N});
  Kokkos::MDRangePolicy<Kokkos::Rank<5>> p(start,end);

  Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l, const int& m){
    view4(i,j,k,l) = i+j+k+l;
  });

  mf.setField(field4,view4);

  auto serialized4 = field4.serialize();
  
  MeshField::Field deserialized4 = mf.makeField<3>();

  deserialized4.deserialize(serialized4);

  Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l, const int& m){
    assert(view4(i, j, k, l) == deserialized4(i, j, k, l));
  });
  return 0;
}
