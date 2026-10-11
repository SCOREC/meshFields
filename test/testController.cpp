#include "MeshField_TestFields.hpp"
#include <MeshField_Scan.hpp>
#ifdef MESHFIELDS_ENABLE_CABANA
#include <MeshField_SimdFor.hpp>
#endif
#include <string>

using MeshField::LO;

/*
 * The sizes, parallel_for access, and Field::set of a field of rank R
 */
template <typename T, typename Ctrlr, int R>
void testRank(Ctrlr &c, Extents n) {
  const auto rank = "(rank " + std::to_string(R) + ")";
  auto field = MeshField::makeField<Ctrlr, R - 1>(c);
  LO sizeErrors = extentMismatches<R>(field, n);
  for (int d = 0; d < R; ++d)
    sizeErrors += c.size(R - 1, d) != n[d];
  report("testSize" + rank, sizeErrors);

  fill<R>(field, n);
  report("testParallelFor" + rank, mismatches<R>(field, n));

  const auto view = createView<R, T>(n);
  fill<R>(view, n);
  fill<R>(field, n, 1); // so that a set that does nothing fails
  field.set(view);
  report("testSet" + rank, mismatches<R>(field, n));
}

/*
 * Tests written once for every controller and data type T
 */
template <template <typename> typename Ranks, typename T>
void testShared(std::string const &typeName) {
  detail::ContextScope scope(typeName);
  using Ctrlr = Ranks<T>;
  auto c = createRanks<Ctrlr>();
  testRank<T, Ctrlr, 1>(c, rank1);
  testRank<T, Ctrlr, 2>(c, rank2);
  testRank<T, Ctrlr, 3>(c, rank3);
  testRank<T, Ctrlr, 4>(c, rank4);
  if constexpr (Ctrlr::MAX_RANK >= 5)
    testRank<T, Ctrlr, 5>(c, rank5);
}

/*
 * Kokkos only: construct controllers of fixed and runtime extents, and of
 * rank 0
 */
void testKokkosConstruct() {
  using RuntimeThenFixed =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, double **[3]>;
  using Fixed = MeshField::KokkosController<MemorySpace, ExecutionSpace,
                                            double[3]>;
  using Runtime =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int *****>;
  RuntimeThenFixed runtimeThenFixed({4, 5});
  Fixed fixed;
  Runtime runtime({2, 3, 4, 5, 6});
  // a rank 0 field holds one value; it has no extents to check
  MeshField::KokkosController<MemorySpace, ExecutionSpace, int> rank0;
  LO errors = 0;
  const int runtimeThenFixedExtents[3] = {4, 5, 3};
  for (int d = 0; d < 3; ++d)
    errors += runtimeThenFixed.size(0, d) != runtimeThenFixedExtents[d];
  errors += fixed.size(0, 0) != 3;
  for (int d = 0; d < 5; ++d)
    errors += runtime.size(0, d) != d + 2;
  report("testKokkosConstruct", errors);
}

/*
 * Kokkos only: static first extents, runtime extents after the first, given
 * in one list for all fields, and rank 5
 */
void testKokkosExtents() {
  using Ctrlr =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int[5],
                                  int **[3], int ***[2][1], int *****>;
  Ctrlr c({4, 6, /**/ 7, 8, 9, /**/ 2, 3, 4, 5, 6});
  const int ranks[4] = {1, 3, 5, 5};
  const int extents[4][5] = {
      {5}, {4, 6, 3}, {7, 8, 9, 2, 1}, {2, 3, 4, 5, 6}};
  LO errors = 0;
  for (int f = 0; f < 4; ++f)
    for (int d = 0; d < ranks[f]; ++d)
      errors += c.size(f, d) != extents[f][d];
  report("testKokkosExtents", errors);

  // runtime extents after the first, as in CreateLagrangeField
  auto field = MeshField::makeField<Ctrlr, 1>(c);
  fill<3>(field, {4, 6, 3});
  report("testKokkosRuntimeExtents", mismatches<3>(field, {4, 6, 3}));
}

#ifdef MESHFIELDS_ENABLE_CABANA
/*
 * Cabana only: simd_parallel_for of rank R
 */
template <typename T, int R>
void testSimdParallelFor(CabanaRanks<T> &c, Extents n) {
  auto f = MeshField::makeField<CabanaRanks<T>, R - 1>(c);
  if constexpr (R == 1) {
    auto kernel = KOKKOS_LAMBDA(int i) { f(i) = indexValue(i); };
    MeshField::simd_parallel_for(c, {0}, {n[0]}, kernel, "simd1");
  } else if constexpr (R == 2) {
    auto kernel = KOKKOS_LAMBDA(int i, int j) { f(i, j) = indexValue(i, j); };
    MeshField::simd_parallel_for(c, {0, 0}, {n[0], n[1]}, kernel, "simd2");
  } else if constexpr (R == 3) {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k) {
      f(i, j, k) = indexValue(i, j, k);
    };
    MeshField::simd_parallel_for(c, {0, 0, 0}, {n[0], n[1], n[2]}, kernel,
                                 "simd3");
  } else {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k, int l) {
      f(i, j, k, l) = indexValue(i, j, k, l);
    };
    MeshField::simd_parallel_for(c, {0, 0, 0, 0}, {n[0], n[1], n[2], n[3]},
                                 kernel, "simd4");
  }
  report("testSimdParallelFor(rank " + std::to_string(R) + ")",
         mismatches<R>(f, n));
}

template <typename T> void testSimd(std::string const &typeName) {
  detail::ContextScope scope(typeName);
  auto c = createRanks<CabanaRanks<T>>();
  testSimdParallelFor<T, 1>(c, rank1);
  testSimdParallelFor<T, 2>(c, rank2);
  testSimdParallelFor<T, 3>(c, rank3);
  testSimdParallelFor<T, 4>(c, rank4);
}

/*
 * Cabana only: a controller with no entries
 */
void testCabanaEmpty() {
  MeshField::CabanaController<ExecutionSpace, MemorySpace, int> c;
  LO errors = 0;
  for (int d = 0; d < 4; ++d)
    errors += c.size(0, d) != 0;
  report("testCabanaEmpty", errors);
}
#endif

/*
 * parallel_reduce of ranks 1 to 4; no controller is involved.  The sum of
 * i*j*... over [0,N)^R is S^R with S = N(N-1)/2.
 */
void testParallelReduce() {
  const int N = 10;
  const double S = N * (N - 1) / 2.0;
  double r1 = 0, r2 = 0, r3 = 0, r4 = 0;
  auto k1 = KOKKOS_LAMBDA(int i, double &l) { l += i; };
  auto k2 = KOKKOS_LAMBDA(int i, int j, double &l) { l += i * j; };
  auto k3 = KOKKOS_LAMBDA(int i, int j, int k, double &l) { l += i * j * k; };
  auto k4 = KOKKOS_LAMBDA(int i, int j, int k, int m, double &l) {
    l += i * j * k * m;
  };
  MeshField::parallel_reduce(ExecutionSpace(), "reduce1", {0}, {N}, k1, r1);
  MeshField::parallel_reduce(ExecutionSpace(), "reduce2", {0, 0}, {N, N}, k2,
                             r2);
  MeshField::parallel_reduce(ExecutionSpace(), "reduce3", {0, 0, 0}, {N, N, N},
                             k3, r3);
  MeshField::parallel_reduce(ExecutionSpace(), "reduce4", {0, 0, 0, 0},
                             {N, N, N, N}, k4, r4);
  report("testParallelReduce", (r1 != S) + (r2 != S * S) + (r3 != S * S * S) +
                                   (r4 != S * S * S * S));
}

/*
 * parallel_scan; no controller is involved.  The sum of [0,n) is n(n-1)/2.
 */
void testParallelScan() {
  auto kernel = KOKKOS_LAMBDA(int i, int &partialSum, const bool) {
    partialSum += i;
  };
  LO errors = 0;
  for (int n = 1; n <= 100; ++n) {
    int result = 0;
    MeshField::parallel_scan(ExecutionSpace(), "scan", 0, n, kernel, result);
    errors += result != n * (n - 1) / 2;
  }
  report("testParallelScan", errors);
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  {
    detail::ContextScope scope("Kokkos");
    testShared<KokkosRanks, int>("int");
    testShared<KokkosRanks, double>("double");
    testKokkosConstruct();
    testKokkosExtents();
  }
#ifdef MESHFIELDS_ENABLE_CABANA
  {
    detail::ContextScope scope("Cabana");
    testShared<CabanaRanks, int>("int");
    testShared<CabanaRanks, double>("double");
    testSimd<int>("int");
    testSimd<double>("double");
    testCabanaEmpty();
  }
#endif
  testParallelReduce();
  testParallelScan();
  return t.exitCode();
}
