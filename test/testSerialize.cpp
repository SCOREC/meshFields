#include "KokkosController.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_Macros.hpp"
#include "MeshField_Utility.hpp"

#include <Kokkos_Core.hpp>

#include <initializer_list>
#include <iostream>
#include <vector>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

void test_1_1_16() {
  using kCon =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int ***>;
  kCon c1({1, 1, 16});
  MeshField::Field field = MeshField::makeField<kCon, 0>(c1);
  auto field_serialized = field.serialize();
}

void test_multi() {
  const int N = 10;
  using kok1 =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int *, int **,
                                  int ***, int ****, int *****>;
  kok1 c1({N, N, N, N, N, N, N, N, N, N, N, N, N, N, N});

  MeshField::Field field1 = MeshField::makeField<kok1, 0>(c1);
  MeshField::Field field2 = MeshField::makeField<kok1, 1>(c1);
  MeshField::Field field3 = MeshField::makeField<kok1, 2>(c1);
  MeshField::Field field4 = MeshField::makeField<kok1, 3>(c1);
  MeshField::Field field5 = MeshField::makeField<kok1, 4>(c1);

  Kokkos::View<int *> view1("1", N);
  Kokkos::View<int **> view2("2", N, N);
  Kokkos::View<int ***> view3("3", N, N, N);
  Kokkos::View<int ****> view4("4", N, N, N, N);
  Kokkos::View<int *****> view5("5", N, N, N, N, N);

  Kokkos::Array start = MeshFieldUtil::to_kokkos_array<5>({0, 0, 0, 0, 0});
  Kokkos::Array end = MeshFieldUtil::to_kokkos_array<5>({N, N, N, N, N});
  Kokkos::MDRangePolicy<Kokkos::Rank<5>> p(start, end);

  Kokkos::parallel_for(
      "", p,
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l,
                    const int &m) {
        view1(i) = i;
        view2(i, j) = i + j;
        view3(i, j, k) = i + j + k;
        view4(i, j, k, l) = i + j + k + l;
        view5(i, j, k, l, m) = i + j + k + l + m;
      });

  field1.set(view1);
  field2.set(view2);
  field3.set(view3);
  field4.set(view4);
  field5.set(view5);

  auto serialized1 = field1.serialize();
  auto serialized2 = field2.serialize();
  auto serialized3 = field3.serialize();
  auto serialized4 = field4.serialize();
  auto serialized5 = field5.serialize();

  field1.deserialize(serialized1);
  field2.deserialize(serialized2);
  field3.deserialize(serialized3);
  field4.deserialize(serialized4);
  field5.deserialize(serialized5);

  Kokkos::parallel_for(
      "", p,
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l,
                    const int &m) {
        assert(view1(i) == field1(i));
        assert(view2(i, j) == field2(i, j));
        assert(view3(i, j, k) == field3(i, j, k));
        assert(view4(i, j, k, l) == field4(i, j, k, l));
        assert(view5(i, j, k, l, m) == field5(i, j, k, l, m));
      });
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);
  test_1_1_16();
  test_multi();
  return 0;
}
