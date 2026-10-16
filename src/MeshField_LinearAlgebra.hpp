#ifndef MESHFIELD_LINEARALGEBRA_HPP
#define MESHFIELD_LINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <cassert>

/**
 * @file MeshField_LinearAlgebra.hpp
 * @brief small dense matrix operations callable on host and device
 *
 * Matrices are Dim x Dim and stored row-major in a
 * Kokkos::Array<Real, Dim * Dim>.
 */

namespace MeshField {

/**
 * @brief the determinant of a 1x1, 2x2, or 3x3 matrix stored row-major
 */
template <size_t Dim>
KOKKOS_INLINE_FUNCTION Real
determinant(Kokkos::Array<Real, Dim * Dim> const &A) {
  static_assert(Dim >= 1 && Dim <= 3, "only 1x1, 2x2, and 3x3 are supported");
  if constexpr (Dim == 1) {
    return A[0];
  } else if constexpr (Dim == 2) {
    return A[0] * A[3] - A[1] * A[2];
  } else {
    return A[0] * (A[4] * A[8] - A[5] * A[7]) -
           A[1] * (A[3] * A[8] - A[5] * A[6]) +
           A[2] * (A[3] * A[7] - A[4] * A[6]);
  }
}

/**
 * @brief invert a 1x1, 2x2, or 3x3 matrix stored row-major
 */
template <size_t Dim>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, Dim * Dim>
invert(Kokkos::Array<Real, Dim * Dim> const &A) {
  static_assert(Dim >= 1 && Dim <= 3, "only 1x1, 2x2, and 3x3 are supported");
  const auto det = determinant<Dim>(A);
  assert(Kokkos::fabs(det) > Epsilon);
  Kokkos::Array<Real, Dim * Dim> inv;
  if constexpr (Dim == 1) {
    inv[0] = 1;
  } else if constexpr (Dim == 2) {
    inv[0] = A[3];
    inv[1] = -A[1];
    inv[2] = -A[2];
    inv[3] = A[0];
  } else {
    // adjugate
    inv[0] = A[4] * A[8] - A[5] * A[7];
    inv[1] = A[2] * A[7] - A[1] * A[8];
    inv[2] = A[1] * A[5] - A[2] * A[4];
    inv[3] = A[5] * A[6] - A[3] * A[8];
    inv[4] = A[0] * A[8] - A[2] * A[6];
    inv[5] = A[2] * A[3] - A[0] * A[5];
    inv[6] = A[3] * A[7] - A[4] * A[6];
    inv[7] = A[1] * A[6] - A[0] * A[7];
    inv[8] = A[0] * A[4] - A[1] * A[3];
  }
  for (size_t i = 0; i < Dim * Dim; ++i)
    inv[i] /= det;
  return inv;
}

} // namespace MeshField

#endif
