#ifndef MESHFIELD_TESTFIELDS_HPP
#define MESHFIELD_TESTFIELDS_HPP

/**
 * @file MeshField_TestFields.hpp
 * @brief controllers with one field of each rank, and helpers that fill and
 * check a Field or a Kokkos::View of any rank, shared by the controller tests
 */

#include "MeshField_TestHarness.hpp"
#include <MeshField_Field.hpp>
#include <MeshField_For.hpp>
#include <MeshField_Reduce.hpp>

using MemorySpace = ExecutionSpace::memory_space;

/** @brief the extents of a field or View of rank up to 5 */
using Extents = Kokkos::Array<int, 5>;

// Fields of type T, one of each rank up to the controller's MAX_RANK: 5 for
// Kokkos, 4 for Cabana.  Each field has a different number of entries (its
// first extent), so mixed up fields show.  The Kokkos data types include the
// first extent, Cabana's do not.
constexpr int Y = 3, Z = 4, A = 5, B = 2;
constexpr Extents rank1 = {10}, rank2 = {11, Y}, rank3 = {12, Y, Z},
                  rank4 = {13, Y, Z, A}, rank5 = {14, Y, Z, A, B};
template <typename T>
using KokkosRanks =
    MeshField::KokkosController<MemorySpace, ExecutionSpace, T *, T *[Y],
                                T *[Y][Z], T *[Y][Z][A], T *[Y][Z][A][B]>;
#ifdef MESHFIELDS_ENABLE_CABANA
template <typename T>
using CabanaRanks = MeshField::CabanaController<ExecutionSpace, MemorySpace, T,
                                                T[Y], T[Y][Z], T[Y][Z][A]>;
#endif

/** @brief create a controller with the fields of ranks 1 to MAX_RANK */
template <typename Ctrlr> Ctrlr createRanks() {
  // both controllers take the runtime extents, here only the first ones
  if constexpr (Ctrlr::MAX_RANK >= 5)
    return Ctrlr({rank1[0], rank2[0], rank3[0], rank4[0], rank5[0]});
  else
    return Ctrlr({rank1[0], rank2[0], rank3[0], rank4[0]});
}

/** @brief a value that differs at every index, so transposed indices show;
 *  every value used is exact in float and double */
KOKKOS_INLINE_FUNCTION int indexValue(int i, int j = 0, int k = 0, int l = 0,
                                      int m = 0) {
  return i + 10 * j + 100 * k + 1000 * l + 10000 * m;
}

/** @brief set every entry of a Field or View of rank R to indexValue +
 *  offset */
template <int R, typename Container>
void fill(Container const &c, Extents n, int offset = 0) {
  static_assert(R >= 1 && R <= 5);
  if constexpr (R == 1) {
    auto kernel = KOKKOS_LAMBDA(int i) { c(i) = indexValue(i) + offset; };
    MeshField::parallel_for(ExecutionSpace(), {0}, {n[0]}, kernel, "fill1");
  } else if constexpr (R == 2) {
    auto kernel = KOKKOS_LAMBDA(int i, int j) {
      c(i, j) = indexValue(i, j) + offset;
    };
    MeshField::parallel_for(ExecutionSpace(), {0, 0}, {n[0], n[1]}, kernel,
                            "fill2");
  } else if constexpr (R == 3) {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k) {
      c(i, j, k) = indexValue(i, j, k) + offset;
    };
    MeshField::parallel_for(ExecutionSpace(), {0, 0, 0}, {n[0], n[1], n[2]},
                            kernel, "fill3");
  } else if constexpr (R == 4) {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k, int l) {
      c(i, j, k, l) = indexValue(i, j, k, l) + offset;
    };
    MeshField::parallel_for(ExecutionSpace(), {0, 0, 0, 0},
                            {n[0], n[1], n[2], n[3]}, kernel, "fill4");
  } else {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k, int l, int m) {
      c(i, j, k, l, m) = indexValue(i, j, k, l, m) + offset;
    };
    MeshField::parallel_for(ExecutionSpace(), {0, 0, 0, 0, 0},
                            {n[0], n[1], n[2], n[3], n[4]}, kernel, "fill5");
  }
}

/** @brief the number of entries of a Field or View of rank R that differ from
 *  indexValue */
template <int R, typename Container>
MeshField::LO mismatches(Container const &c, Extents n) {
  static_assert(R >= 1 && R <= 5);
  MeshField::LO errors = 0;
  if constexpr (R == 1) {
    auto kernel = KOKKOS_LAMBDA(int i, MeshField::LO &lerr) {
      lerr += c(i) != indexValue(i);
    };
    MeshField::parallel_reduce(ExecutionSpace(), "mismatches1", {0}, {n[0]},
                               kernel, errors);
  } else if constexpr (R == 2) {
    auto kernel = KOKKOS_LAMBDA(int i, int j, MeshField::LO &lerr) {
      lerr += c(i, j) != indexValue(i, j);
    };
    MeshField::parallel_reduce(ExecutionSpace(), "mismatches2", {0, 0},
                               {n[0], n[1]}, kernel, errors);
  } else if constexpr (R == 3) {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k, MeshField::LO &lerr) {
      lerr += c(i, j, k) != indexValue(i, j, k);
    };
    MeshField::parallel_reduce(ExecutionSpace(), "mismatches3", {0, 0, 0},
                               {n[0], n[1], n[2]}, kernel, errors);
  } else if constexpr (R == 4) {
    auto kernel =
        KOKKOS_LAMBDA(int i, int j, int k, int l, MeshField::LO &lerr) {
      lerr += c(i, j, k, l) != indexValue(i, j, k, l);
    };
    MeshField::parallel_reduce(ExecutionSpace(), "mismatches4", {0, 0, 0, 0},
                               {n[0], n[1], n[2], n[3]}, kernel, errors);
  } else {
    auto kernel = KOKKOS_LAMBDA(int i, int j, int k, int l, int m,
                                MeshField::LO &lerr) {
      lerr += c(i, j, k, l, m) != indexValue(i, j, k, l, m);
    };
    MeshField::parallel_reduce(ExecutionSpace(), "mismatches5",
                               {0, 0, 0, 0, 0}, {n[0], n[1], n[2], n[3], n[4]},
                               kernel, errors);
  }
  return errors;
}

/** @brief a Kokkos::View<T*...> of rank R */
template <int R, typename T> auto createView(Extents n) {
  static_assert(R >= 1 && R <= 5);
  if constexpr (R == 1)
    return Kokkos::View<T *>("view1", n[0]);
  else if constexpr (R == 2)
    return Kokkos::View<T **>("view2", n[0], n[1]);
  else if constexpr (R == 3)
    return Kokkos::View<T ***>("view3", n[0], n[1], n[2]);
  else if constexpr (R == 4)
    return Kokkos::View<T ****>("view4", n[0], n[1], n[2], n[3]);
  else
    return Kokkos::View<T *****>("view5", n[0], n[1], n[2], n[3], n[4]);
}

/** @brief the number of extents of a field of rank R that differ from n */
template <int R, typename Field>
MeshField::LO extentMismatches(Field const &field, Extents n) {
  MeshField::LO errors = 0;
  for (int d = 0; d < R; ++d)
    errors += static_cast<int>(field.size(d)) != n[d];
  return errors;
}

#endif
