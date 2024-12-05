#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Macros.hpp"
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
      Controller::KokkosController<MemorySpace, ExecutionSpace, double *>;
  Ctrlr c({10});
  MeshField::MeshField<Ctrlr> kokkosMeshField(c);

  auto field0 = kokkosMeshField.makeField<0>();

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
        Controller::KokkosController<MemorySpace, ExecutionSpace, double **[3]>;
    Ctrlr c({num_tuples, num_tuples});
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr =
        Controller::KokkosController<MemorySpace, ExecutionSpace, double[3]>;
    Ctrlr c;
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int *****>;
    Ctrlr c({10, 10, 10, 10, 10});
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int>;
    Ctrlr c;
    MeshField::MeshField<Ctrlr> kok(c);
  }

  printf("== END testKokkosConstructor ==\n");
}

void testingStufffs() {

  printf("== START testingStufffs ==\n");
  using Ctrlr =
      Controller::KokkosController<MemorySpace, ExecutionSpace, int *>;
  Ctrlr c({10});
  MeshField::MeshField<Ctrlr> kok(c);

  auto field0 = kok.makeField<0>();

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
        Controller::KokkosController<MemorySpace, ExecutionSpace, int[a],
                                     int[a][b], int[a][b][c], int[a][b][c][d],
                                     int[a][b][c][d][e]>;
    Ctrlr ctrl;
    MeshField::MeshField<Ctrlr> kok(ctrl);

    auto rk1 = kok.makeField<0>();
    auto rk2 = kok.makeField<1>();
    auto rk3 = kok.makeField<2>();
    auto rk4 = kok.makeField<3>();
    auto rk5 = kok.makeField<4>();

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
    kok.parallel_for({0}, {a}, k1, "testKokkosParallelFor(rank1)");
    kok.parallel_for({0, 0}, {a, b}, k2, "testKokkosParallelFor(rank2)");
    kok.parallel_for({0, 0, 0}, {a, b, c}, k3, "testKokkosParallelFor(rank3)");
    kok.parallel_for({0, 0, 0, 0}, {a, b, c, d}, k4,
                     "testKokkosParallelFor(rank4)");
    kok.parallel_for({0, 0, 0, 0, 0}, {a, b, c, d, e}, k5,
                     "testKokkosParallelFor(rank5)");
  }

  printf("== END testKokkosParallelFor ==\n");
}

void kokkosParallelReduceTest() {
  /* Examples from Kokkos Documentation:
   * https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html?highlight=parallel_reduce*/

  printf("== START kokkosParallelReduceTest ==\n");
  using Ctrlr =
      Controller::KokkosController<MemorySpace, ExecutionSpace, int *>;
  Ctrlr c1({10});
  MeshField::MeshField<Ctrlr> kok(c1);

  {
    double result;
    int N = 10;
    auto kernel = KOKKOS_LAMBDA(const int &i, double &lsum) {
      lsum += 1.0 * i;
    };
    kok.parallel_reduce("ReduceTest", {0}, {N}, kernel, result);
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
    kok.parallel_reduce("ReduceTest2", {0, 0}, {N, N}, kernel, result);
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
    kok.parallel_reduce("ReduceTest3", {0, 0, 0}, {N, N, N}, kernel, result);
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
    kok.parallel_reduce("ReduceTest4", {0, 0, 0, 0}, {N, N, N, N}, kernel,
                        result);
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
  {
    /* BEGIN STATIC DIMENSIONS TESTS */
    using simple_static =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int[a]>;
    using large_simple_static =
        Controller::KokkosController<MemorySpace, ExecutionSpace,
                                     int[a][b][c][d][e]>;
    using multi_static =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int[a][b][c],
                                     double[a][b][c]>;

    simple_static c1;
    large_simple_static c2;
    multi_static c3;

    MeshField::MeshField<simple_static> simple_kok(c1);
    MeshField::MeshField<large_simple_static> large_kok(c2);
    MeshField::MeshField<multi_static> multi_kok(c3);

    assert(simple_kok.size(0, 0) == a);

    for (int i = 0; i < 5; i++) {
      assert(large_kok.size(0, i) == psi[i]);
    }

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        int foo = multi_kok.size(i, j);
        assert((foo == psi[j]));
      }
    }

    /* END STATIC DIMENSIONS TESTS */
  }
  {
    /* BEGIN DYNAMIC DIMENSION TESTS */
    using simple_dynamic =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int *>;
    using large_simple_dynamic =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int *****>;
    using multi_dynamic =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int ***,
                                     double ***>;

    simple_dynamic c1({5});
    large_simple_dynamic c2({5, 4, 3, 2, 1});
    multi_dynamic c3({5, 4, 3, 5, 4, 3});

    MeshField::MeshField<simple_dynamic> simple_kok(c1);
    MeshField::MeshField<large_simple_dynamic> large_kok(c2);
    MeshField::MeshField<multi_dynamic> multi_kok(c3);

    assert(simple_kok.size(0, 0) == a);

    for (int i = 0; i < 5; i++)
      assert(large_kok.size(0, i) == psi[i]);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        assert(multi_kok.size(i, j) == psi[j]);

    /* END DYNAMIC DIMENSION TESTS */
  }

  {
    /* BEGIN MIXED DIMENSION TESTS */
    using simple_mixed =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int *[b]>;
    using large_simple_mixed =
        Controller::KokkosController<MemorySpace, ExecutionSpace,
                                     int **[c][d][e]>;
    using multi_mixed =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int ***[d][e],
                                     double **[c][d][e]>;
    using complex_multi_mixed =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int ***[d][e],
                                     double **[c][d][e], char ****[e],
                                     bool *[b][c][d][e]>;

    simple_mixed c1({5});
    large_simple_mixed c2({5, 4});
    multi_mixed c3({5, 4, 3, 5, 4});
    complex_multi_mixed c4({5, 4, 3, 5, 4, 5, 4, 3, 2, 5});

    MeshField::MeshField<simple_mixed> simple_kok(c1);
    MeshField::MeshField<large_simple_mixed> large_kok(c2);
    MeshField::MeshField<multi_mixed> multi_kok(c3);
    MeshField::MeshField<complex_multi_mixed> complex_kok(c4);

    assert(simple_kok.size(0, 0) == a);
    assert(simple_kok.size(0, 1) == b);

    for (int i = 0; i < 5; i++)
      assert(large_kok.size(0, i) == psi[i]);

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        assert(multi_kok.size(i, j) == psi[j]);
      }
    }

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 5; j++) {
        assert(complex_kok.size(i, j) == psi[j]);
      }
    }
    /* END MIXED DIMENSION TESTS */
  }

  printf("== END kokkosControllerSizeTest ==\n");
}

void kokkosFieldSizeTest() {
  printf("== START kokkosFieldSizeTest ==\n");
  const int a = 5, b = 4, c = 3, d = 2, e = 1;
  const int psi[5] = {a, b, c, d, e};
  {
    using simple_static =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int[a],
                                     char[a][b], double[a][b][c],
                                     bool[a][b][c][d], long[a][b][c][d][e]>;
    using simple_dynamic =
        Controller::KokkosController<MemorySpace, ExecutionSpace, int *,
                                     char **, double ***, bool ****,
                                     long *****>;
    using mixed = Controller::KokkosController<
        MemorySpace, ExecutionSpace, int *[b][c][d][e], char **[c][d][e],
        double ***[d][e], bool ****[e], long *****>;
    simple_static c1;
    simple_dynamic c2({5, 5, 4, 5, 4, 3, 5, 4, 3, 2, 5, 4, 3, 2, 1});
    mixed c3({5, 5, 4, 5, 4, 3, 5, 4, 3, 2, 5, 4, 3, 2, 1});

    MeshField::MeshField<simple_static> simple_kok(c1);
    MeshField::MeshField<simple_dynamic> dynamic_kok(c2);
    MeshField::MeshField<mixed> mixed_kok(c3);

    {
      auto field0 = simple_kok.makeField<0>();
      auto field1 = simple_kok.makeField<1>();
      auto field2 = simple_kok.makeField<2>();
      auto field3 = simple_kok.makeField<3>();
      auto field4 = simple_kok.makeField<4>();

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
      auto field0 = dynamic_kok.makeField<0>();
      auto field1 = dynamic_kok.makeField<1>();
      auto field2 = dynamic_kok.makeField<2>();
      auto field3 = dynamic_kok.makeField<3>();
      auto field4 = dynamic_kok.makeField<4>();

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
      auto field0 = mixed_kok.makeField<0>();
      auto field1 = mixed_kok.makeField<1>();
      auto field2 = mixed_kok.makeField<2>();
      auto field3 = mixed_kok.makeField<3>();
      auto field4 = mixed_kok.makeField<4>();

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
      Controller::KokkosController<MemorySpace, ExecutionSpace, int *, int *>;
  s_kok c1({N, N});
  MeshField::MeshField<s_kok> mfk(c1);
  auto pre = mfk.makeField<0>();
  auto post = mfk.makeField<1>();

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
    mfk.parallel_scan("default", 0, i, scan_kernel, result);
    assert(result == seriesSum(i));
  }
  printf("== END testParallelScan ==\n");
}

void testSetField() {
  printf("== START testSetField ==\n");
  const int N = 10;
  using kok1 =
      Controller::KokkosController<MemorySpace, ExecutionSpace, int *, int **,
                                   int ***, int ****, int *****>;
  kok1 c1({N, N, N, N, N, N, N, N, N, N, N, N, N, N, N});
  MeshField::MeshField mf(c1);
  auto f1 = mf.makeField<0>();
  auto f2 = mf.makeField<1>();
  auto f3 = mf.makeField<2>();
  auto f4 = mf.makeField<3>();
  auto f5 = mf.makeField<4>();

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

  mf.setField(f1, v1);
  mf.setField(f2, v2);
  mf.setField(f3, v3);
  mf.setField(f4, v4);
  mf.setField(f5, v5);

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
  using kok1 = Controller::KokkosController<MemorySpace, ExecutionSpace, int **,
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
