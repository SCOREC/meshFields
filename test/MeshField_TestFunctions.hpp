#ifndef MESHFIELD_TESTFUNCTIONS_HPP
#define MESHFIELD_TESTFUNCTIONS_HPP

/**
 * @file MeshField_TestFunctions.hpp
 * @brief analytic functions and coordinate maps shared by the tests
 */

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>

template <size_t dim> using Vec = Kokkos::Array<MeshField::Real, dim>;

/** @brief f(x) = 1 + sum_d (d+1) x_d */
template <size_t dim> struct LinearFunction {
  KOKKOS_INLINE_FUNCTION MeshField::Real operator()(Vec<dim> const &x) const {
    MeshField::Real f = 1;
    for (size_t d = 0; d < dim; ++d)
      f += (d + 1) * x[d];
    return f;
  }
  KOKKOS_INLINE_FUNCTION Vec<dim> grad(Vec<dim> const &) const {
    Vec<dim> g;
    for (size_t d = 0; d < dim; ++d)
      g[d] = d + 1;
    return g;
  }
};

/**
 * @brief f(x) = 1 + sum_d ((d+1) x_d + x_d^2) + x_0 x_1
 *
 * The cross term gives the Hessian an off-diagonal entry.
 */
template <size_t dim> struct QuadraticFunction {
  static_assert(dim >= 2, "the cross term needs at least two dimensions");
  KOKKOS_INLINE_FUNCTION MeshField::Real operator()(Vec<dim> const &x) const {
    MeshField::Real f = 1 + x[0] * x[1];
    for (size_t d = 0; d < dim; ++d)
      f += (d + 1) * x[d] + x[d] * x[d];
    return f;
  }
  KOKKOS_INLINE_FUNCTION Vec<dim> grad(Vec<dim> const &x) const {
    Vec<dim> g;
    for (size_t d = 0; d < dim; ++d)
      g[d] = (d + 1) + 2 * x[d];
    g[0] += x[1];
    g[1] += x[0];
    return g;
  }
};

/**
 * @brief the value of a numComp component function whose component c is
 * (c+1)*f, which checks the component ordering
 */
template <size_t numComp, size_t dim, typename Function>
KOKKOS_INLINE_FUNCTION Kokkos::Array<MeshField::Real, numComp>
valueScaledByComponent(Function const &f, Vec<dim> const &x) {
  Kokkos::Array<MeshField::Real, numComp> val;
  for (size_t c = 0; c < numComp; ++c)
    val[c] = (c + 1) * f(x);
  return val;
}

/**
 * @brief the gradient of the function in valueScaledByComponent, indexed
 * [direction * numComp + component]; f must provide grad
 */
template <size_t numComp, size_t dim, typename Function>
KOKKOS_INLINE_FUNCTION Kokkos::Array<MeshField::Real, dim * numComp>
gradScaledByComponent(Function const &f, Vec<dim> const &x) {
  const auto gf = f.grad(x);
  Kokkos::Array<MeshField::Real, dim * numComp> g;
  for (size_t d = 0; d < dim; ++d)
    for (size_t c = 0; c < numComp; ++c)
      g[d * numComp + c] = (c + 1) * gf[d];
  return g;
}

/**
 * @brief CoordinateMap for an affine shear that makes every Jacobian a full
 * matrix: x += 0.3y + 0.1z, y += 0.2z
 *
 * Its determinant is one, so it preserves measure.  Coordinate fields of any
 * order reproduce it exactly.
 */
template <size_t dim> struct Shear {
  KOKKOS_INLINE_FUNCTION Vec<dim> operator()(Vec<dim> x) const {
    x[0] += 0.3 * x[1];
    if constexpr (dim == 3) {
      x[0] += 0.1 * x[2];
      x[1] += 0.2 * x[2];
    }
    return x;
  }
};

/**
 * @brief CoordinateMap that bends the mesh: y += 0.2x^2
 *
 * Quadratic, so quadratic coordinate fields reproduce it exactly.
 */
template <size_t dim> struct Bend {
  KOKKOS_INLINE_FUNCTION Vec<dim> operator()(Vec<dim> x) const {
    x[1] += 0.2 * x[0] * x[0];
    return x;
  }
};

#endif
