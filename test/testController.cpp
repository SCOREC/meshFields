#include "MeshField_TestFields.hpp"
#include "MeshField_TestHarness.hpp"
#include <MeshField_Scan.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#ifdef MESHFIELDS_ENABLE_CABANA
#include <MeshField_SimdFor.hpp>
#endif

using MeshField::LO;

/*
 * The sizes, parallel_for access, and Field::set of a field of rank R
 */
template <typename T, typename Ctrlr, int R>
void testRank(Ctrlr &c, Extents n) {
  INFO("rank " << R);
  auto field = MeshField::makeField<Ctrlr, R - 1>(c);
  for (int d = 0; d < R; ++d) {
    CAPTURE(d);
    REQUIRE(field.size(d) == n[d]);
    REQUIRE(c.size(R - 1, d) == n[d]);
  }

  fill<R, Ctrlr>(field, n);
  CHECK(mismatches<R>(field, n) == 0);

  const auto view = createView<R, T>(n);
  fill<R>(view, n);
  fill<R, Ctrlr>(field, n, 1); // so that a set that does nothing fails
  field.set(view);
  CHECK(mismatches<R>(field, n) == 0);
}

TEMPLATE_TEST_CASE_SIG("Kokkos field rank", "[kokkos]",
                       ((typename T, int R), T, R), (int, 1), (int, 2),
                       (int, 3), (int, 4), (int, 5), (double, 1), (double, 2),
                       (double, 3), (double, 4), (double, 5)) {
  auto c = createRanks<KokkosRanks<T>>();
  constexpr Extents extents[] = {rank1, rank2, rank3, rank4, rank5};
  testRank<T, KokkosRanks<T>, R>(c, extents[R - 1]);
}

/*
 * Kokkos only: construct controllers of fixed and runtime extents, and of
 * rank 0
 */
TEST_CASE("Kokkos construct", "[kokkos]") {
  using RuntimeThenFixed =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, double **[3]>;
  using Fixed =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, double[3]>;
  using Runtime =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int *****>;
  RuntimeThenFixed runtimeThenFixed({4, 5});
  Fixed fixed;
  Runtime runtime({2, 3, 4, 5, 6});
  // a rank 0 field holds one value; it has no extents to check
  MeshField::KokkosController<MemorySpace, ExecutionSpace, int> rank0;
  const int runtimeThenFixedExtents[3] = {4, 5, 3};
  for (int d = 0; d < 3; ++d)
    CHECK(runtimeThenFixed.size(0, d) == runtimeThenFixedExtents[d]);
  CHECK(fixed.size(0, 0) == 3);
  for (int d = 0; d < 5; ++d)
    CHECK(runtime.size(0, d) == d + 2);
}

/*
 * Kokkos only: static first extents, runtime extents after the first, given
 * in one list for all fields, and rank 5
 */
TEST_CASE("Kokkos extents", "[kokkos]") {
  using Ctrlr =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int[5],
                                  int **[3], int ***[2][1], int *****>;
  Ctrlr c({4, 6, /**/ 7, 8, 9, /**/ 2, 3, 4, 5, 6});
  const int ranks[4] = {1, 3, 5, 5};
  const int extents[4][5] = {{5}, {4, 6, 3}, {7, 8, 9, 2, 1}, {2, 3, 4, 5, 6}};
  for (int f = 0; f < 4; ++f)
    for (int d = 0; d < ranks[f]; ++d) {
      CAPTURE(f, d);
      CHECK(c.size(f, d) == extents[f][d]);
    }

  // runtime extents after the first, as in CreateLagrangeField
  auto field = MeshField::makeField<Ctrlr, 1>(c);
  fill<3>(field, {4, 6, 3});
  CHECK(mismatches<3>(field, {4, 6, 3}) == 0);
}

#ifdef MESHFIELDS_ENABLE_CABANA
TEMPLATE_TEST_CASE_SIG("Cabana field rank", "[cabana]",
                       ((typename T, int R), T, R), (int, 1), (int, 2),
                       (int, 3), (int, 4), (double, 1), (double, 2),
                       (double, 3), (double, 4)) {
  auto c = createRanks<CabanaRanks<T>>();
  constexpr Extents extents[] = {rank1, rank2, rank3, rank4};
  testRank<T, CabanaRanks<T>, R>(c, extents[R - 1]);
}

/*
 * Cabana only: simd_parallel_for of rank R
 */
template <typename T, int R>
void testSimdParallelFor(CabanaRanks<T> &c, Extents n) {
  auto f = MeshField::makeField<CabanaRanks<T>, R - 1>(c);
  if constexpr (R == 1) {
    auto kernel = KOKKOS_LAMBDA(int i) { f(i) = indexValue(i); };
    MeshField::simd_parallel_for<CabanaRanks<T>>({0}, {n[0]}, kernel, "simd1");
  } else if constexpr (R == 2) {
    auto kernel = KOKKOS_LAMBDA(int i, int j) { f(i, j) = indexValue(i, j); };
    MeshField::simd_parallel_for<CabanaRanks<T>>({0, 0}, {n[0], n[1]}, kernel,
                                                 "simd2");
  } else if constexpr (R == 3) {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k) {
      f(i, j, k) = indexValue(i, j, k);
    };
    MeshField::simd_parallel_for<CabanaRanks<T>>({0, 0, 0}, {n[0], n[1], n[2]},
                                                 kernel, "simd3");
  } else {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k, int l) {
      f(i, j, k, l) = indexValue(i, j, k, l);
    };
    MeshField::simd_parallel_for<CabanaRanks<T>>(
        {0, 0, 0, 0}, {n[0], n[1], n[2], n[3]}, kernel, "simd4");
  }
  INFO("rank " << R);
  CHECK(mismatches<R>(f, n) == 0);
}

TEMPLATE_TEST_CASE_SIG("Cabana simd_parallel_for", "[cabana]",
                       ((typename T, int R), T, R), (int, 1), (int, 2),
                       (int, 3), (int, 4), (double, 1), (double, 2),
                       (double, 3), (double, 4)) {
  auto c = createRanks<CabanaRanks<T>>();
  constexpr Extents extents[] = {rank1, rank2, rank3, rank4};
  testSimdParallelFor<T, R>(c, extents[R - 1]);
}

/*
 * Cabana only: a controller with no entries
 */
TEST_CASE("Cabana empty", "[cabana]") {
  MeshField::CabanaController<ExecutionSpace, MemorySpace, int> c;
  for (int d = 0; d < 4; ++d)
    CHECK(c.size(0, d) == 0);
}
#endif

/*
 * parallel_reduce of ranks 1 to 4; no controller is involved.  The sum of
 * i*j*... over [0,N)^R is S^R with S = N(N-1)/2.
 */
constexpr int N = 10;
Kokkos::Array<double, 4> reduceRanks() {
  Kokkos::Array<double, 4> r = {0, 0, 0, 0};
  auto k1 = KOKKOS_LAMBDA(int i, double &l) { l += i; };
  auto k2 = KOKKOS_LAMBDA(int i, int j, double &l) { l += i * j; };
  auto k3 = KOKKOS_LAMBDA(int i, int j, int k, double &l) { l += i * j * k; };
  auto k4 = KOKKOS_LAMBDA(int i, int j, int k, int m, double &l) {
    l += i * j * k * m;
  };
  MeshField::parallel_reduce(ExecutionSpace(), "reduce1", {0}, {N}, k1, r[0]);
  MeshField::parallel_reduce(ExecutionSpace(), "reduce2", {0, 0}, {N, N}, k2,
                             r[1]);
  MeshField::parallel_reduce(ExecutionSpace(), "reduce3", {0, 0, 0}, {N, N, N},
                             k3, r[2]);
  MeshField::parallel_reduce(ExecutionSpace(), "reduce4", {0, 0, 0, 0},
                             {N, N, N, N}, k4, r[3]);
  return r;
}

TEST_CASE("parallel_reduce", "[for]") {
  const double S = N * (N - 1) / 2.0;
  const auto r = reduceRanks();
  CHECK(r[0] == S);
  CHECK(r[1] == S * S);
  CHECK(r[2] == S * S * S);
  CHECK(r[3] == S * S * S * S);
}

/*
 * parallel_scan; no controller is involved.  The sum of [0,n) is n(n-1)/2.
 */
int scanSum(int n) {
  auto kernel = KOKKOS_LAMBDA(int i, int &partialSum, const bool) {
    partialSum += i;
  };
  int result = 0;
  MeshField::parallel_scan(ExecutionSpace(), "scan", 0, n, kernel, result);
  return result;
}

TEST_CASE("parallel_scan", "[for]") {
  const int n = GENERATE(Catch::Generators::range(1, 101));
  CAPTURE(n);
  CHECK(scanSum(n) == n * (n - 1) / 2);
}
