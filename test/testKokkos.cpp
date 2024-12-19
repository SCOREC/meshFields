#include "KokkosController.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_For.hpp"
#include "MeshField_Macros.hpp"
#include "MeshField_Reduce.hpp"
#include "MeshField_Scan.hpp"
#include "MeshField_Utility.hpp"

#include <cassert>
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

void testMakeSliceKokkos() {
  printf("== START testMakeSliceKokkos ==\n");
  int N = 10;
  using Ctrlr =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, double *>;
  Ctrlr c({10});

  auto field0 = MeshField::makeField<Ctrlr, 0>(c);

  auto testKernel = KOKKOS_LAMBDA(const int x) {
    double gamma = (double)x;
    field0(x) = gamma;
    assert(doubleCompare(field0(x), gamma));
  };

  Kokkos::parallel_for("testMakeSliceKokkos()", N, testKernel);
  printf("== END testMakeSliceKokkos ==\n");
}

void testKokkosConstructor(int num_tuples) {
  printf("== START testKokkosConstructor ==\n");
  {
    using Ctrlr =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, double **[3]>;
    Ctrlr c({num_tuples, num_tuples});
  }
  {
    using Ctrlr =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, double[3]>;
    Ctrlr c;
  }
  {
    using Ctrlr =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int *****>;
    Ctrlr c({10, 10, 10, 10, 10});
  }
  {
    using Ctrlr = MeshField::KokkosController<MemorySpace, ExecutionSpace, int>;
    Ctrlr c;
  }

  printf("== END testKokkosConstructor ==\n");
}

void testingStufffs() {

  printf("== START testingStufffs ==\n");
  using Ctrlr = MeshField::KokkosController<MemorySpace, ExecutionSpace, int *>;
  Ctrlr c({10});

  auto field0 = MeshField::makeField<Ctrlr, 0>(c);

  auto vectorKernel = KOKKOS_LAMBDA(const int &s) { field0(s) = 3; };
  Kokkos::parallel_for("tag", 10, vectorKernel);

  printf("== END testingStufffs ==\n");
}

void testKokkosParallelFor() {
  printf("== START testKokkosParallelFor ==\n");
  const int a = 10;
  const int b = 9;
  const int c = 8;
  const int d = 7;
  const int e = 6;
  {
    using Ctrlr =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int[a],
                                    int[a][b], int[a][b][c], int[a][b][c][d],
                                    int[a][b][c][d][e]>;
    Ctrlr ctrl;
    auto rk1 = MeshField::makeField<Ctrlr, 0>(ctrl);
    auto rk2 = MeshField::makeField<Ctrlr, 1>(ctrl);
    auto rk3 = MeshField::makeField<Ctrlr, 2>(ctrl);
    auto rk4 = MeshField::makeField<Ctrlr, 3>(ctrl);
    auto rk5 = MeshField::makeField<Ctrlr, 4>(ctrl);

    auto k1 = KOKKOS_LAMBDA(const int &i) {
      rk1(i) = i;
      assert(rk1(i) == i);
    };
    auto k2 = KOKKOS_LAMBDA(const int i, const int j) {
      rk2(i, j) = i + j;
      assert(rk2(i, j) == i + j);
    };
    auto k3 = KOKKOS_LAMBDA(int i, int j, int k) {
      rk3(i, j, k) = i + j + k;
      assert(rk3(i, j, k) == i + j + k);
    };
    auto k4 = KOKKOS_LAMBDA(int i, int j, int k, int l) {
      rk4(i, j, k, l) = i + j + k + l;
      assert(rk4(i, j, k, l) == i + j + k + l);
    };
    auto k5 = KOKKOS_LAMBDA(int i, int j, int k, int l, int m) {
      rk5(i, j, k, l, m) = i + j + k + l + m;
      assert(rk5(i, j, k, l, m) == i + j + k + l + m);
    };
    MeshField::parallel_for(ExecutionSpace(), {0}, {a}, k1,
                            "testKokkosParallelFor(rank1)");
    MeshField::parallel_for(ExecutionSpace(), {0, 0}, {a, b}, k2,
                            "testKokkosParallelFor(rank2)");
    MeshField::parallel_for(ExecutionSpace(), {0, 0, 0}, {a, b, c}, k3,
                            "testKokkosParallelFor(rank3)");
    MeshField::parallel_for(ExecutionSpace(), {0, 0, 0, 0}, {a, b, c, d}, k4,
                            "testKokkosParallelFor(rank4)");
    MeshField::parallel_for(ExecutionSpace(), {0, 0, 0, 0, 0}, {a, b, c, d, e},
                            k5, "testKokkosParallelFor(rank5)");
  }

  printf("== END testKokkosParallelFor ==\n");
}

void kokkosParallelReduceTest() {
  /* Examples from Kokkos Documentation:
   * https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html?highlight=parallel_reduce*/

  printf("== START kokkosParallelReduceTest ==\n");
  using Ctrlr = MeshField::KokkosController<MemorySpace, ExecutionSpace, int *>;
  Ctrlr c1({10});
  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA(const int &i, double &lsum) {
      lsum += 1.0 * i;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "ReduceTest", {0}, {N}, kernel,
                               result);
    double result_verify = 0;
    for (int i = 0; i < N; i++) {
      result_verify += 1.0 * i;
    }
    assert(doubleCompare(result_verify, result));
    printf("Reduce test 1-D Result: %d %.2lf\n", N, result);
  }
  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA(const int &i, const int &j, double &lsum) {
      lsum += i * j;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "ReduceTest2", {0, 0}, {N, N},
                               kernel, result);
    double result_verify = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        result_verify += i * j;
      }
    }
    assert(doubleCompare(result_verify, result));

    printf("Reduce test 2-D Result: %d %.2lf\n", N, result);
  }
  {
    double result;
    int N = 10;
    auto kernel =
        KOKKOS_LAMBDA(const int &i, const int &j, const int &k, double &lsum) {
      lsum += i * j * k;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "ReduceTest3", {0, 0, 0},
                               {N, N, N}, kernel, result);
    double result_verify = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          result_verify += i * j * k;
        }
      }
    }
    assert(doubleCompare(result_verify, result));
    printf("Reduce test 3-D Result: %d %.2lf\n", N, result);
  }
  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA(const int &i, const int &j, const int &k,
                                const int &l, double &lsum) {
      lsum += i * j * k * l;
    };
    MeshField::parallel_reduce(ExecutionSpace(), "ReduceTest4", {0, 0, 0, 0},
                               {N, N, N, N}, kernel, result);
    double result_verify = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          for (int l = 0; l < N; l++) {
            result_verify += i * j * k * l;
          }
        }
      }
    }
    assert(doubleCompare(result_verify, result));
    printf("Reduce test 4-D Result: %d %.2lf\n", N, result);
  }

  printf("== END kokkosParallelReduceTest ==\n");
}

void kokkosControllerSizeTest() {
  printf("== START kokkosControllerSizeTest ==\n");

  const int a = 5;
  const int b = 4;
  const int c = 3;
  const int d = 2;
  const int e = 1;

  const int psi[5] = {a, b, c, d, e};
  /* BEGIN STATIC DIMENSIONS TESTS */
  {
    using simple_static =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int[a]>;
    simple_static c1;
    assert(c1.size(0, 0) == a);
  }
  {
    using large_simple_static =
        MeshField::KokkosController<MemorySpace, ExecutionSpace,
                                    int[a][b][c][d][e]>;
    large_simple_static c2;
    for (int i = 0; i < 5; i++) {
      assert(c2.size(0, i) == psi[i]);
    }
  }
  {
    using multi_static =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int[a][b][c],
                                    double[a][b][c]>;
    multi_static c3;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        int foo = c3.size(i, j);
        assert((foo == psi[j]));
      }
    }
  }
  /* END STATIC DIMENSIONS TESTS */
  /* BEGIN DYNAMIC DIMENSION TESTS */
  {
    using simple_dynamic =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int *>;
    simple_dynamic c1({5});
    assert(c1.size(0, 0) == a);
  }
  {
    using large_simple_dynamic =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int *****>;
    large_simple_dynamic c2({5, 4, 3, 2, 1});
    for (int i = 0; i < 5; i++)
      assert(c2.size(0, i) == psi[i]);
  }
  {
    using multi_dynamic =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int ***,
                                    double ***>;
    multi_dynamic c3({5, 4, 3, 5, 4, 3});
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        assert(c3.size(i, j) == psi[j]);
  }
  /* END DYNAMIC DIMENSION TESTS */
  /* BEGIN MIXED DIMENSION TESTS */
  {
    using simple_mixed =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int *[b]>;
    simple_mixed c1({5});
    assert(c1.size(0, 0) == a);
    assert(c1.size(0, 1) == b);
  }
  {
    using large_simple_mixed =
        MeshField::KokkosController<MemorySpace, ExecutionSpace,
                                    int **[c][d][e]>;
    large_simple_mixed c2({5, 4});
    for (int i = 0; i < 5; i++)
      assert(c2.size(0, i) == psi[i]);
  }
  {
    using multi_mixed =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int ***[d][e],
                                    double **[c][d][e]>;
    multi_mixed c3({5, 4, 3, 5, 4});
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        assert(c3.size(i, j) == psi[j]);
      }
    }
  }
  {
    using complex_multi_mixed =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int ***[d][e],
                                    double **[c][d][e], char ****[e],
                                    bool *[b][c][d][e]>;
    complex_multi_mixed c4({5, 4, 3, 5, 4, 5, 4, 3, 2, 5});
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 5; j++) {
        assert(c4.size(i, j) == psi[j]);
      }
    }
  }
  /* END MIXED DIMENSION TESTS */

  printf("== END kokkosControllerSizeTest ==\n");
}

void kokkosFieldSizeTest() {
  printf("== START kokkosFieldSizeTest ==\n");
  const int a = 5, b = 4, c = 3, d = 2, e = 1;
  const int psi[5] = {a, b, c, d, e};
  {
    using simple_static =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int[a],
                                    char[a][b], double[a][b][c],
                                    bool[a][b][c][d], long[a][b][c][d][e]>;
    using simple_dynamic =
        MeshField::KokkosController<MemorySpace, ExecutionSpace, int *, char **,
                                    double ***, bool ****, long *****>;
    using mixed =
        MeshField::KokkosController<MemorySpace, ExecutionSpace,
                                    int *[b][c][d][e], char **[c][d][e],
                                    double ***[d][e], bool ****[e], long *****>;
    simple_static c1;
    simple_dynamic c2({5, 5, 4, 5, 4, 3, 5, 4, 3, 2, 5, 4, 3, 2, 1});
    mixed c3({5, 5, 4, 5, 4, 3, 5, 4, 3, 2, 5, 4, 3, 2, 1});

    {
      auto field0 = MeshField::makeField<simple_static, 0>(c1);
      auto field1 = MeshField::makeField<simple_static, 1>(c1);
      auto field2 = MeshField::makeField<simple_static, 2>(c1);
      auto field3 = MeshField::makeField<simple_static, 3>(c1);
      auto field4 = MeshField::makeField<simple_static, 4>(c1);

      assert(field0.size(0) == a);
      for (int i = 0; i < 2; i++) {
        assert(field1.size(i) == psi[i]);
      }
      for (int i = 0; i < 3; i++) {
        assert(field2.size(i) == psi[i]);
      }
      for (int i = 0; i < 4; i++) {
        assert(field3.size(i) == psi[i]);
      }
      for (int i = 0; i < 5; i++) {
        assert(field4.size(i) == psi[i]);
      }
    }
    {
      auto field0 = MeshField::makeField<simple_dynamic, 0>(c2);
      auto field1 = MeshField::makeField<simple_dynamic, 1>(c2);
      auto field2 = MeshField::makeField<simple_dynamic, 2>(c2);
      auto field3 = MeshField::makeField<simple_dynamic, 3>(c2);
      auto field4 = MeshField::makeField<simple_dynamic, 4>(c2);

      assert(field0.size(0) == a);
      for (int i = 0; i < 2; i++) {
        assert(field1.size(i) == psi[i]);
      }
      for (int i = 0; i < 3; i++) {
        assert(field2.size(i) == psi[i]);
      }
      for (int i = 0; i < 4; i++) {
        assert(field3.size(i) == psi[i]);
      }
      for (int i = 0; i < 5; i++) {
        assert(field4.size(i) == psi[i]);
      }
    }
    {
      auto field0 = MeshField::makeField<mixed, 0>(c3);
      auto field1 = MeshField::makeField<mixed, 1>(c3);
      auto field2 = MeshField::makeField<mixed, 2>(c3);
      auto field3 = MeshField::makeField<mixed, 3>(c3);
      auto field4 = MeshField::makeField<mixed, 4>(c3);

      for (int i = 0; i < 5; i++) {
        assert(field0.size(i) == psi[i]);
        assert(field1.size(i) == psi[i]);
        assert(field2.size(i) == psi[i]);
        assert(field3.size(i) == psi[i]);
        assert(field4.size(i) == psi[i]);
      }
    }
  }

  printf("== END kokkosFieldSizeTest ==\n");
}

void testParallelScan() {
  const int N = 100;
  printf("== START testParallelScan ==\n");
  using s_kok =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int *, int *>;
  s_kok c1({N, N});
  auto pre = MeshField::makeField<s_kok, 0>(c1);
  auto post = MeshField::makeField<s_kok, 1>(c1);

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
  using kok1 =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int *, int **,
                                  int ***, int ****, int *****>;
  kok1 c1({N, N, N, N, N, N, N, N, N, N, N, N, N, N, N});
  auto f1 = MeshField::makeField<kok1, 0>(c1);
  auto f2 = MeshField::makeField<kok1, 1>(c1);
  auto f3 = MeshField::makeField<kok1, 2>(c1);
  auto f4 = MeshField::makeField<kok1, 3>(c1);
  auto f5 = MeshField::makeField<kok1, 4>(c1);

  Kokkos::View<int *> v1("1", N);
  Kokkos::View<int **> v2("2", N, N);
  Kokkos::View<int ***> v3("3", N, N, N);
  Kokkos::View<int ****> v4("4", N, N, N, N);
  Kokkos::View<int *****> v5("5", N, N, N, N, N);

  Kokkos::Array start = MeshFieldUtil::to_kokkos_array<5>({0, 0, 0, 0, 0});
  Kokkos::Array end = MeshFieldUtil::to_kokkos_array<5>({N, N, N, N, N});
  Kokkos::MDRangePolicy<Kokkos::Rank<5>> p(start, end);

  Kokkos::parallel_for(
      "", p,
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l,
                    const int &m) {
        v1(i) += i;
        v2(i, j) += i + j;
        v3(i, j, k) += i + j + k;
        v4(i, j, k, l) += i + j + k + l;
        v5(i, j, k, l, m) += i + j + k + l + m;
      });

  f1.set(v1);
  f2.set(v2);
  f3.set(v3);
  f4.set(v4);
  f5.set(v5);

  Kokkos::parallel_for(
      "", p,
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l,
                    const int &m) {
        assert(f1(i) == v1(i));
        assert(f2(i, j) == v2(i, j));
        assert(f3(i, j, k) == v3(i, j, k));
        assert(f4(i, j, k, l) == v4(i, j, k, l));
        assert(f5(i, j, k, l, m) == v5(i, j, k, l, m));
      });
  printf("== END testSetField ==\n");
}

void testSetCorrect() {
  printf("== START testSetCorrect ==\n");
  const int N = 10;
  using kok1 = MeshField::KokkosController<MemorySpace, ExecutionSpace, int **,
                                           int ***, int ****, int *****>;
  kok1 c1({N, 1, N, N, N, N, N, 15, N, N, N, N, N, 6});

  // Checking that sizes are loaded correctly
  assert(c1.size(0, 1) == 1);
  assert(c1.size(2, 2) == 15);
  assert(c1.size(3, 4) == 6);

  printf("== END testSetCorrect ==\n");
}

int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);

  testKokkosConstructor(num_tuples);
  testKokkosParallelFor();
  kokkosParallelReduceTest();
  testMakeSliceKokkos();
  kokkosControllerSizeTest();
  kokkosFieldSizeTest();
  testParallelScan();
  testSetField();
  testSetCorrect();

  return 0;
}
