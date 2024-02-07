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
  MeshField::Field field2 = mf.makeField<1>();

  Kokkos::View<int**> view2("2",N,N);

  Kokkos::Array start = MeshFieldUtil::to_kokkos_array<5>({0,0,0,0,0});
  Kokkos::Array end = MeshFieldUtil::to_kokkos_array<5>({N,N,N,N,N});
  Kokkos::MDRangePolicy<Kokkos::Rank<5>> p(start,end);

  Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l, const int& m){
    view2(i,j) = i+j;
  });

  mf.setField(field2,view2);

  auto serialized2 = field2.serialize();
  
  MeshField::Field deserialized2 = mf.makeField<1>();

  deserialized2.deserialize(serialized2);

  Kokkos::parallel_for( "",p,KOKKOS_LAMBDA(const int& i,const int& j, const int& k, const int& l, const int& m){
    assert(view2(i, j) == deserialized2(i, j));
  });
  return 0;
}
