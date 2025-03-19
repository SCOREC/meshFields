#include "CabanaController.hpp"
#include "KokkosController.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_Macros.hpp"
#include "MeshField_Reduce.hpp"
#include "MeshField_Scan.hpp"
#include "MeshField_SimdFor.hpp"
#include "MeshField_Utility.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <initializer_list>
#include <iostream>
#include <stdio.h>
#include <vector>

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
int seriesSum(int x) { return (int)(((double)x - 1.0)) * (((double)x / 2.0)); }

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

void testMakeSliceCabana(int num_tuples) {
  printf("== START testMakeSliceCabana ==\n");
  using Ctrl = MeshField::CabanaController<ExecutionSpace, MemorySpace, double>;
  Ctrl c({num_tuples});

  auto field0 = MeshField::makeField<Ctrl, 0>(c);

  auto testKernel = KOKKOS_LAMBDA(const int x) {
    double gamma = (double)x;
    field0(x) = gamma;
    assert(doubleCompare(field0(x), gamma));
  };
  Kokkos::parallel_for("testMakeSliceCabana()", num_tuples, testKernel);

  printf("== END testMakeSliceCabana ==\n");
}

void testParallelReduceCabana() {
  printf("== START testParallelReduceCabana ==\n");
  using Ctrl = MeshField::CabanaController<ExecutionSpace, MemorySpace, int>;
  const int N = 9;
  Ctrl c({N});

  {
    double result = 0, verify = 0;
    auto reduce_kernel = KOKKOS_LAMBDA(const int &i, double &lsum) {
      lsum += i * 1.0;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "CabanaReduceTest1", {0}, {N},
                               reduce_kernel, result);
    for (int i = 0; i < N; i++)
      verify += i * 1.0;
    assert(doubleCompare(verify, result));
  }
  {
    double result = 0, verify = 0;
    auto reduce_kernel =
        KOKKOS_LAMBDA(const int &i, const int &j, double &lsum) {
      lsum += i * j;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "CabanaReduceTest2", {0, 0},
                               {N, N}, reduce_kernel, result);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        verify += i * j;
      }
    }
    assert(doubleCompare(verify, result));
  }
  {
    double result = 0, verify = 0;
    auto reduce_kernel =
        KOKKOS_LAMBDA(const int &i, const int &j, const int &k, double &lsum) {
      lsum += i * j * k;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "CabanaReduceTest3", {0, 0, 0},
                               {N, N, N}, reduce_kernel, result);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          verify += i * j * k;
        }
      }
    }

    assert(doubleCompare(verify, result));
  }
  {
    double result = 0, verify = 0;
    auto reduce_kernel = KOKKOS_LAMBDA(const int &i, const int &j, const int &k,
                                       const int &l, double &lsum) {
      lsum += i * j * k * l;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "CabanaReduceTest4",
                               {0, 0, 0, 0}, {N, N, N, N}, reduce_kernel,
                               result);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          for (int l = 0; l < N; l++) {
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

  const int a = 6, b = 5, c = 4, d = 3;
  const int psi[4] = {a, b, c, d};

  using simple =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, int[b]>;
  simple c1({a});
  for (int i = 0; i < 2; i++) {
    assert(c1.size(0, i) == psi[i]);
  }

  using multi =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, int[b][c][d],
                                  char[b][c][d], bool[b][c][d]>;
  multi c2({a, a, a});
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      assert(c2.size(i, j) == psi[j]);
    }
  }

  using varied =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, double[b][c],
                                  int, float[b][c][d], char[b]>;
  varied c3({a, a, a, a});
  for (int i = 0; i < 3; i++)
    assert(c3.size(0, i) == psi[i]);
  assert(c3.size(1, 0) == psi[0]);
  for (int i = 0; i < 4; i++)
    assert(c3.size(2, i) == psi[i]);
  for (int i = 0; i < 2; i++)
    assert(c3.size(3, i) == psi[i]);

  using empty = MeshField::CabanaController<ExecutionSpace, MemorySpace, int>;
  empty c4;
  for (int i = 0; i < 4; i++) {
    assert(c4.size(0, i) == 0);
  }
  printf("== END testCabanaControllerSize ==\n");
}

void testCabanaFieldSize() {
  printf("== START testCabanaFieldSize ==\n");

  const int a = 6, b = 5, c = 4, d = 3;
  const int psi[4] = {a, b, c, d};

  using simple =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, int[b]>;
  using multi =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, int[b][c][d],
                                  char[b][c][d], bool[b][c][d]>;
  using varied =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, double[b][c],
                                  int, float[b][c][d], char[b]>;
  using empty = MeshField::CabanaController<ExecutionSpace, MemorySpace, int>;
  simple c1({a});
  multi c2({a, a, a});
  varied c3({a, a, a, a});
  empty c4;

  const int MAX_RANK = 4;

  {
    auto field0 = MeshField::makeField<simple, 0>(c1);
    for (int i = 0; i < 2; i++)
      assert(field0.size(i) == psi[i]);
  }

  {
    auto field0 = MeshField::makeField<multi, 0>(c2);
    auto field1 = MeshField::makeField<multi, 1>(c2);
    auto field2 = MeshField::makeField<multi, 2>(c2);
    for (int i = 0; i < MAX_RANK; i++) {
      assert(field0.size(i) == psi[i]);
      assert(field1.size(i) == psi[i]);
      assert(field2.size(i) == psi[i]);
    }
  }

  {
    auto field0 = MeshField::makeField<varied, 0>(c3);
    auto field1 = MeshField::makeField<varied, 1>(c3);
    auto field2 = MeshField::makeField<varied, 2>(c3);
    auto field3 = MeshField::makeField<varied, 3>(c3);

    for (int i = 0; i < 3; i++) {
      assert(field0.size(i) == psi[i]);
    }
    for (int i = 0; i < 1; i++) {
      assert(field1.size(i) == psi[i]);
    }
    for (int i = 0; i < 4; i++) {
      assert(field2.size(i) == psi[i]);
    }
    for (int i = 0; i < 2; i++) {
      assert(field3.size(i) == psi[i]);
    }
  }
  {
    auto field0 = MeshField::makeField<empty, 0>(c4);
    for (int i = 0; i < MAX_RANK; i++) {
      assert(field0.size(i) == 0);
    }
  }
  multi diffc2({a, a + 1, a + 2});
  varied diffc3({a, a + 3, a + 4, a + 5});
  {
    auto field0 = MeshField::makeField<multi, 0>(diffc2);
    auto field1 = MeshField::makeField<multi, 1>(diffc2);
    auto field2 = MeshField::makeField<multi, 2>(diffc2);
    assert(field0.size(0) == a);
    assert(field1.size(0) == a + 1);
    assert(field2.size(0) == a + 2);
    for (int i = 1; i < MAX_RANK; ++i) {
      assert(field0.size(i) == psi[i]);
      assert(field1.size(i) == psi[i]);
      assert(field2.size(i) == psi[i]);
    }
  }
  {
    auto field0 = MeshField::makeField<varied, 0>(diffc3);
    auto field1 = MeshField::makeField<varied, 1>(diffc3);
    auto field2 = MeshField::makeField<varied, 2>(diffc3);
    auto field3 = MeshField::makeField<varied, 3>(diffc3);
    assert(field0.size(0) == a);
    assert(field1.size(0) == a + 3);
    assert(field2.size(0) == a + 4);
    assert(field3.size(0) == a + 5);
    for (int i = 1; i < 3; i++) {
      assert(field0.size(i) == psi[i]);
    }
    for (int i = 1; i < 1; i++) {
      assert(field1.size(i) == psi[i]);
    }
    for (int i = 1; i < 4; i++) {
      assert(field2.size(i) == psi[i]);
    }
    for (int i = 1; i < 2; i++) {
      assert(field3.size(i) == psi[i]);
    }
  }
  printf("== END testCabanaFieldSize ==\n");
}

void testCabanaParallelFor() {
  printf("== START testCabanaParallelFor() ==\n");
  const int x = 10, y = 9, z = 8, a = 7;
  {
    using simd_ctrlr =
        MeshField::CabanaController<ExecutionSpace, MemorySpace, int, int[y],
                                    int[y][z], int[y][z][a]>;
    simd_ctrlr c1({x, x, x, x});
    auto field0 = MeshField::makeField<simd_ctrlr, 0>(c1);
    auto field1 = MeshField::makeField<simd_ctrlr, 1>(c1);
    auto field2 = MeshField::makeField<simd_ctrlr, 2>(c1);
    auto field3 = MeshField::makeField<simd_ctrlr, 3>(c1);

    auto vectorKernel = KOKKOS_LAMBDA(const int &i) {
      field0(i) = i;
      assert(field0(i) == i);
    };
    MeshField::simd_parallel_for(c1, {0}, {x}, vectorKernel, "simple_loop");

    auto vectorKernel2 = KOKKOS_LAMBDA(const int &i, const int &j) {
      field1(i, j) = i + j;
      assert(field1(i, j) == i + j);
    };
    MeshField::simd_parallel_for(c1, {0, 0}, {x, y}, vectorKernel2,
                                 "simple_loop");

    auto vectorKernel3 =
        KOKKOS_LAMBDA(const int &i, const int &j, const int &k) {
      field2(i, j, k) = i + j + k;
      assert(field2(i, j, k) == i + j + k);
    };
    MeshField::simd_parallel_for(c1, {0, 0, 0}, {x, y, z}, vectorKernel3,
                                 "simple_loop");

    auto vectorKernel4 =
        KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l) {
      field3(i, j, k, l) = i + j + k + l;
      assert(field3(i, j, k, l) == i + j + k + l);
    };
    MeshField::simd_parallel_for(c1, {0, 0, 0, 0}, {x, y, z, a}, vectorKernel4,
                                 "simple_loop");
  }

  printf("== END testCabanaParallelFor() ==\n");
}

void testParallelScan() {
  const int N = 100;
  printf("== START testParallelScan ==\n");
  using s_cab =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, int, int>;
  s_cab c1({N, N});
  auto pre = MeshField::makeField<s_cab, 0>(c1);
  auto post = MeshField::makeField<s_cab, 1>(c1);

  auto scan_kernel =
      KOKKOS_LAMBDA(int i, int &partial_sum, const bool is_final) {
    if (is_final)
      pre(i) = partial_sum;
    partial_sum += i;
    if (is_final)
      post(i) = partial_sum;
  };

  for (int i = 1; i <= N; i++) {
    int result;
    MeshField::parallel_scan(ExecutionSpace(), "default", 0, i, scan_kernel,
                             result);
    assert(result == seriesSum(i));
  }
  printf("== END testParallelScan ==\n");
}

void testSetField() {
  printf("== START testSetField ==\n");
  const int N = 10;
  using cab1 = MeshField::CabanaController<ExecutionSpace, MemorySpace, int,
                                           int[N], int[N][N], int[N][N][N]>;
  cab1 c1({N, N, N, N});
  auto f1 = MeshField::makeField<cab1, 0>(c1);
  auto f2 = MeshField::makeField<cab1, 1>(c1);
  auto f3 = MeshField::makeField<cab1, 2>(c1);
  auto f4 = MeshField::makeField<cab1, 3>(c1);

  Kokkos::View<int *> v1("1", N);
  Kokkos::View<int **> v2("2", N, N);
  Kokkos::View<int ***> v3("3", N, N, N);
  Kokkos::View<int ****> v4("4", N, N, N, N);

  Kokkos::Array start = MeshFieldUtil::to_kokkos_array<4>({0, 0, 0, 0});
  Kokkos::Array end = MeshFieldUtil::to_kokkos_array<4>({N, N, N, N});
  Kokkos::MDRangePolicy<Kokkos::Rank<4>> p(start, end);

  Kokkos::parallel_for(
      "", p,
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l) {
        v1(i) += i;
        v2(i, j) += i + j;
        v3(i, j, k) += i + j + k;
        v4(i, j, k, l) += i + j + k + l;
      });

  f1.set(v1);
  f2.set(v2);
  f3.set(v3);
  f4.set(v4);

  Kokkos::parallel_for(
      "", p,
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l) {
        assert(f1(i) == v1(i));
        assert(f2(i, j) == v2(i, j));
        assert(f3(i, j, k) == v3(i, j, k));
        assert(f4(i, j, k, l) == v4(i, j, k, l));
      });
  printf("== END testSetField ==\n");
}

int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  testMakeSliceCabana(num_tuples);
  testParallelReduceCabana();
  testCabanaControllerSize();
  testCabanaFieldSize();
  testCabanaParallelFor();
  testSetField();
  testParallelScan();
  return 0;
}
