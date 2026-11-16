#include "MeshField_LinearAlgebra.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;
template <size_t Dim> using Matrix = Kokkos::Array<Real, Dim * Dim>;

template <size_t Dim>
KOKKOS_INLINE_FUNCTION Matrix<Dim> multiply(Matrix<Dim> const &A,
                                            Matrix<Dim> const &B) {
  Matrix<Dim> C;
  for (size_t i = 0; i < Dim; ++i)
    for (size_t j = 0; j < Dim; ++j) {
      C[i * Dim + j] = 0;
      for (size_t k = 0; k < Dim; ++k)
        C[i * Dim + j] += A[i * Dim + k] * B[k * Dim + j];
    }
  return C;
}

KOKKOS_INLINE_FUNCTION bool isClose(Real a, Real b,
                                    Real tol = MeshField::Epsilon) {
  return Kokkos::fabs(a - b) <= tol;
}

/** @brief the number of entries of A and B that differ by more than tol */
template <size_t Dim>
KOKKOS_INLINE_FUNCTION int mismatches(Matrix<Dim> const &A,
                                      Matrix<Dim> const &B,
                                      Real tol = MeshField::Epsilon) {
  int errors = 0;
  for (size_t i = 0; i < Dim * Dim; ++i)
    errors += !isClose(A[i], B[i], tol);
  return errors;
}

template <size_t Dim> KOKKOS_INLINE_FUNCTION Matrix<Dim> identity() {
  Matrix<Dim> I;
  for (size_t i = 0; i < Dim; ++i)
    for (size_t j = 0; j < Dim; ++j)
      I[i * Dim + j] = i == j;
  return I;
}

/** @brief A * invert(A) and invert(A) * A are the identity */
template <size_t Dim>
KOKKOS_INLINE_FUNCTION int inverseMismatches(Matrix<Dim> const &A) {
  const auto inv = MeshField::invert<Dim>(A);
  return mismatches<Dim>(multiply<Dim>(A, inv), identity<Dim>()) +
         mismatches<Dim>(multiply<Dim>(inv, A), identity<Dim>());
}

/** @brief run check, which returns its number of errors, in a kernel */
template <typename Check> bool run(const char *name, Check check) {
  int errors = 0;
  Kokkos::parallel_reduce(
      name, Kokkos::RangePolicy<ExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(const int, int &lerrors) { lerrors += check(); }, errors);
  std::cout << name << ": " << (errors ? "FAIL" : "pass") << "\n";
  return errors == 0;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  failed += !run(
      "determinant of known matrices", KOKKOS_LAMBDA() {
        const Matrix<1> a1{3};
        const Matrix<2> a2{2, -1, //
                           1, 2};
        const Matrix<3> a3{2,  -1, 0, //
                           1,  2,  -1, //
                           0,  -1, 3};
        return !isClose(MeshField::determinant<1>(a1), 3) +
               !isClose(MeshField::determinant<2>(a2), 5) +
               !isClose(MeshField::determinant<3>(a3), 13);
      });
  failed += !run(
      "determinant of singular matrices", KOKKOS_LAMBDA() {
        const Matrix<1> a1{0};
        const Matrix<2> a2{1, 2, //
                           2, 4};
        const Matrix<3> a3{1, 2, 3, //
                           4, 5, 6, //
                           7, 8, 9};
        return !isClose(MeshField::determinant<1>(a1), 0) +
               !isClose(MeshField::determinant<2>(a2), 0) +
               !isClose(MeshField::determinant<3>(a3), 0);
      });
  failed += !run(
      "determinant sign under a row swap", KOKKOS_LAMBDA() {
        const Matrix<2> a2{1, 2, //
                           2, 2};
        const Matrix<2> s2{2, 2, //
                           1, 2};
        const Matrix<3> a3{2,  -1, 0, //
                           1,  2,  -1, //
                           0,  -1, 3};
        const Matrix<3> s3{1,  2,  -1, //
                           2,  -1, 0, //
                           0,  -1, 3};
        return !isClose(MeshField::determinant<2>(a2), -2) +
               !isClose(MeshField::determinant<2>(s2), 2) +
               !isClose(MeshField::determinant<3>(s3),
                        -MeshField::determinant<3>(a3));
      });
  // a transposed (column-major) inverse of a triangular matrix is lower
  // triangular, so these catch a mixed up storage order.  These inverses are
  // exact in floating point, so they are compared with a zero tolerance.
  failed += !run(
      "invert triangular matrices exactly", KOKKOS_LAMBDA() {
        const Matrix<1> a1{4};
        const Matrix<1> inv1{0.25};
        const Matrix<2> a2{1, 2, //
                           0, 1};
        const Matrix<2> inv2{1, -2, //
                             0, 1};
        const Matrix<3> a3{1, 2, 3, //
                           0, 1, 4, //
                           0, 0, 1};
        const Matrix<3> inv3{1, -2, 5, //
                             0, 1,  -4, //
                             0, 0,  1};
        return mismatches<1>(MeshField::invert<1>(a1), inv1, 0) +
               mismatches<2>(MeshField::invert<2>(a2), inv2, 0) +
               mismatches<3>(MeshField::invert<3>(a3), inv3, 0);
      });
  failed += !run(
      "invert gives the identity", KOKKOS_LAMBDA() {
        const Matrix<1> a1{-0.7};
        const Matrix<2> a2{2, -1, //
                           1, 2};
        const Matrix<2> b2{0.3, -1.7, //
                           2.9, 0.4};
        const Matrix<3> a3{2,  -1, 0, //
                           1,  2,  -1, //
                           0,  -1, 3};
        const Matrix<3> b3{0.9,  -0.2, 1.3, //
                           0.4,  1.7,  -0.6, //
                           -1.1, 0.5,  2.2};
        return inverseMismatches<1>(a1) + inverseMismatches<2>(a2) +
               inverseMismatches<2>(b2) + inverseMismatches<3>(a3) +
               inverseMismatches<3>(b3);
      });
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
