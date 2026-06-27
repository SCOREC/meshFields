#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

// getValues(...) implementation copied from
// SCOREC/core apf/apfShape.cc @ 7cd76473

namespace {
template <typename Array>
KOKKOS_INLINE_FUNCTION bool
sumsToOne(Array &xi, double tol = 10 * MeshField::MachinePrecision) {
  // IIFE, capture by reference is preferred
  const bool sums_to_one = [&]() {
    auto sum = 0.0;
    for (size_t i = 0; i < xi.size(); i++) {
      sum += xi[i];
    }
    return (Kokkos::fabs(sum - 1) <= tol);
  }();
  return sums_to_one;
}

template <typename Array>
KOKKOS_INLINE_FUNCTION bool
greaterThanOrEqualZero(Array &xi, double tol = MeshField::Epsilon) {
  for (size_t i = 0; i < xi.size(); i++) {
    if (xi[i] < -tol) {
      return false;
    }
  }
  return true;
}
} // namespace

namespace MeshField {

using Vector2 = Kokkos::Array<Real, 2>;
using Vector3 = Kokkos::Array<Real, 3>;
using Vector4 = Kokkos::Array<Real, 4>;

struct LinearEdgeShape {
  static const size_t numNodes = 2;
  static const size_t meshEntDim = 1;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    // clang-format off
    return {(1.0 - xi[0]) / 2.0,
            (1.0 + xi[0]) / 2.0};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getLocalGradients() const {
    // clang-format off
    return {-0.5, 0.5};
    // clang-format on
  }
};

struct LinearTriangleShape {
  static const size_t order = 1;
  static const size_t numNodes = 3;
  static const size_t numComponentsPerDof = 1;
  static const size_t meshEntDim = 2;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1],
            xi[0],
            xi[1]};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, meshEntDim * numNodes> getLocalGradients() const {
    // clang-format off
    return { -1,-1,  //first vector
              1, 0,
              0, 1};
    // clang-format on
  }
};

struct LinearTriangleCoordinateShape {
  static const size_t numNodes = 3;
  static const size_t meshEntDim = 2;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1],
            xi[0],
            xi[1]};
    // clang-format on
  }
};

struct LinearTetrahedronShape {
  static const size_t numNodes = 4;
  static const size_t meshEntDim = 3;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector4 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1] - xi[2], 
            xi[0], xi[1], 
            xi[2]};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, meshEntDim * numNodes> getLocalGradients() const {
    // clang-format off
    return {-1, -1, -1, 
             1,  0,  0, 
             0,  1,  0, 
             0,  0,  1};
    // clang-format on
  }
};

struct QuadraticTriangleShape {
  static const size_t numNodes = 6;
  static const size_t meshEntDim = 2;
  constexpr static Mesh_Topology DofHolders[2] = {Vertex, Edge};
  constexpr static size_t NumDofHolders[2] = {3, 3};
  constexpr static size_t DofsPerHolder[2] = {1, 1};
  constexpr static size_t Order = 2;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    const Real xi2 = 1 - xi[0] - xi[1];
    // clang-format off
    return {xi2 * (2 * xi2 - 1),
            xi[0] * (2 * xi[0] - 1),
            xi[1] * (2 * xi[1] - 1),
            4 * xi[0] * xi2,
            4 * xi[0] * xi[1],
            4 * xi[1] * xi2};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector2, numNodes> getLocalGradients(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    const Real xi2 = 1 - xi[0] - xi[1];
    // clang-format off
    return {-4*xi2+1,-4*xi2+1,
             4*xi[0]-1,0,
             0,4*xi[1]-1,
             4*(xi2-xi[0]),-4*xi[0],
             4*xi[1],4*xi[0],
             -4*xi[1],4*(xi2-xi[1]) };
    // clang-format on
  }
};

struct QuadraticTetrahedronShape {
  static const size_t numNodes = 10;
  static const size_t meshEntDim = 3;
  constexpr static Mesh_Topology DofHolders[2] = {Vertex, Edge};
  constexpr static size_t NumDofHolders[2] = {4, 6};
  constexpr static size_t DofsPerHolder[2] = {1, 1};
  constexpr static size_t Order = 2;
  // ordering taken from mfem
  // see mfem/mfem fem/fe/fe_fixed_order.cpp @597cba8
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector4 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    const Real xi3 = 1 - xi[0] - xi[1] - xi[2];
    // clang-format off
    return {xi3*(2*xi3-1),
            xi[0]*(2*xi[0]-1),
            xi[1]*(2*xi[1]-1),
            xi[2]*(2*xi[2]-1),
            4*xi[0]*xi3,
            4*xi[1]*xi3,
            4*xi[2]*xi3,
            4*xi[0]*xi[1],
            4*xi[2]*xi[0],
            4*xi[1]*xi[2]};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector3, numNodes> getLocalGradients(Vector4 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    const Real xi3 = 1 - xi[0] - xi[1] - xi[2];
    const Real d3 = 1 - 4 * xi3;
    // clang-format off
    return {d3,d3,d3,
            4*xi[0]-1,0,0,
            0,4*xi[1]-1,0,
            0,0,4*xi[2]-1,
            4*xi3-4*xi[0],-4*xi[0],-4*xi[0],
            -4*xi[1],4*xi3-4*xi[1],-4*xi[1],
            -4*xi[2],-4*xi[2],4*xi3-4*xi[2],
            4*xi[1],4*xi[0],0,
            4*xi[2],0,4*xi[0],
            0,4*xi[2],4*xi[1]};
    // clang-format on
  }
};

/**
 * @brief Helper functions for reduced quintic coordinate transformations
 */
namespace ReducedQuinticHelpers {
  /**
   * @brief Transform barycentric coordinates to reduced quintic local coordinates
   * 
   * local coordinate system (origin at foot of perpendicular):
   *   - v0 is at (-b, 0)
   *   - v1 is at (a, 0)
   *   - v2 is at (0, c)
   * 
   * @param xi Barycentric coordinates [λ0, λ1, λ2]
   * @param a Distance from origin to v1
   * @param b Distance from origin to v0
   * @param c Perpendicular distance from origin to v2
   * @return Vector2 containing [xi_local, eta_local]
   */
  KOKKOS_INLINE_FUNCTION
  Vector2 barycentricToLocal(
      Vector3 const& xi,
      int const order[3],
      Real a,
      Real b,
      Real c)
  {
    // Reorder barycentric coordinates into vertex ordering
    const Real lambda0 = xi[order[0]];
    const Real lambda1 = xi[order[1]];
    const Real lambda2 = xi[order[2]];

    const Real xi_local  = a * lambda1 - b * lambda0;
    const Real eta_local = c * lambda2;

    return {xi_local, eta_local};
  }

  /**
  * @brief Get polynomial index for reduced quintic basis
  * 
  * Returns [i,j] for xi^i * eta^j terms where i+j <= 5
  * Total of 21 possible terms, but we use 20 (the 21st is constrained)
  * 
  * @param idx Index from 0 to 19
  * @return Pair of integers [xi_power, eta_power]
  */
  KOKKOS_INLINE_FUNCTION
  constexpr Kokkos::Array<int, 2> getReducedQuinticPolyIdx(int idx) {
    // Order: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
    // Pattern: for each total degree d from 0 to 5, enumerate (i,j) where i+j=d
    constexpr Kokkos::Array<int, 2> indices[20] = {
      {0,0}, {1,0}, {0,1}, {2,0}, {1,1}, {0,2}, {3,0}, {2,1}, {1,2}, {0,3},
      {4,0}, {3,1}, {2,2}, {1,3}, {0,4}, {5,0}, {3,2}, {2,3}, {1,4}, {0,5}
    };
    return indices[idx];
  }
} // namespace ReducedQuinticHelpers

/**
 * @brief Reduced quintic triangle element
 * 
 * This element uses 18 nodes (3 vertices × 6 DOFs per vertex):
 * - DOFs per vertex: [value, ∂/∂x, ∂/∂y, ∂²/∂x², ∂²/∂x∂y, ∂²/∂y²]
 * - Polynomial order: 5 (20-term basis: xi^i * eta^j for i+j ≤ 5)
 * 
 * COORDINATE TRANSFORMATION:
 * This element uses element-specific local coordinates based on triangle geometry.
 * 
 * The origin is placed at the foot of perpendicular from v2 onto the v0-v1 edge.
 *   - v0 is at (-b, 0) in local coords, where b = distance from origin to v0
 *   - v1 is at (a, 0) in local coords, where a = distance from origin to v1
 *   - v2 is at (0, c) in local coords, where c = perpendicular distance to v2
 * 
 * meshFields API uses barycentric coordinates: (λ0, λ1, λ2) where λ0+λ1+λ2=1
 * Passed as xi[0]=λ0, xi[1]=λ1, xi[2]=λ2.
 * 
 * Transformation:
 *   xi_local = a*λ1 - b*λ0
 *   eta_local = c*λ2
 * 
 * Applied automatically via helper functions:
 *   barycentricToLocal() and localToBarycentricGradient()
 * 
 * The geometric parameters (a, b, c) are stored with the coefficients and
 * retrieved during evaluation. Shape function coefficients are computed by
 * solving a 20×20 linear system based on boundary conditions.
 * 
 */
struct ReducedQuinticTriangleShape {
  static const size_t numNodes = 18;  // 3 vertices × 6 DOFs per vertex
  static const size_t meshEntDim = 2;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t NumDofHolders[1] = {3};      // 3 vertices
  constexpr static size_t DofsPerHolder[1] = {6};      // 6 DOFs per vertex
  constexpr static size_t Order = 5;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi,
                                          Kokkos::View<const Real*, Kokkos::LayoutStride> elemCoeffs) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));

    // Extract geometric parameters from coefficient array
    // elemCoeffs layout: [order[0], order[1], order[2], a, b, c, sin_theta, cos_theta, coeff_0_0, ..., coeff_17_19]
    const int order[3] = {static_cast<int>(elemCoeffs(0)), 
                          static_cast<int>(elemCoeffs(1)), 
                          static_cast<int>(elemCoeffs(2))};
    const Real a = elemCoeffs(3);
    const Real b = elemCoeffs(4);
    const Real c = elemCoeffs(5);
    
    // Reorder barycentric from meshFields pipeline convention to coords-index convention
    const Real xi0_ordered = 1.0 - xi[0] - xi[1];  // coords[0]
    const Real xi1_ordered = xi[0];                 // coords[1]
    const Real xi2_ordered = xi[1];                 // coords[2]
    Vector3 xi_ordered = {xi0_ordered, xi1_ordered, xi2_ordered};

    // Transform barycentric to local coordinates
    const auto local = ReducedQuinticHelpers::barycentricToLocal(xi_ordered, order, a, b, c);
    const Real xi_local = local[0];
    const Real eta_local = local[1];
    
    // Compute polynomial basis: xi_local^i * eta_local^j
    Real xi_pow[6], eta_pow[6];
    xi_pow[0] = 1.0;  eta_pow[0] = 1.0;
    for (int i = 1; i < 6; i++) {
      xi_pow[i] = xi_pow[i-1] * xi_local;
      eta_pow[i] = eta_pow[i-1] * eta_local;
    }
    
    // Evaluate shape functions using precomputed coefficients
    Kokkos::Array<Real, numNodes> N_reordered;

    for (size_t k = 0; k < numNodes; k++) {
      N_reordered[k] = 0.0;

      for (int i = 0; i < 20; i++) {
        const auto poly = ReducedQuinticHelpers::getReducedQuinticPolyIdx(i);
        const int xi_idx   = poly[0];
        const int eta_idx  = poly[1];

        N_reordered[k] +=
            elemCoeffs(8 + k * 20 + i) *
            xi_pow[xi_idx] *
            eta_pow[eta_idx];
      }
    }

    // Convert back to meshFields vertex ordering
    Kokkos::Array<Real, numNodes> N;

    for (int v = 0; v < 3; ++v) {
      const int orig_v = order[v];

      for (int d = 0; d < 6; ++d) {
        N[orig_v * 6 + d] =
            N_reordered[v * 6 + d];
      }
    }

    return N;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector2, numNodes> getLocalGradients(Vector3 const &xi,
                                                      Kokkos::View<const Real*, Kokkos::LayoutStride> elemCoeffs) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    
    // Extract geometric parameters
    const int order[3] = {static_cast<int>(elemCoeffs(0)), 
                          static_cast<int>(elemCoeffs(1)), 
                          static_cast<int>(elemCoeffs(2))};
    const Real a = elemCoeffs(3);
    const Real b = elemCoeffs(4);
    const Real c = elemCoeffs(5);
    
    // Reorder barycentric from meshFields pipeline convention to coords-index convention
    const Real xi0_ordered = 1.0 - xi[0] - xi[1];  // coords[0]
    const Real xi1_ordered = xi[0];                 // coords[1]
    const Real xi2_ordered = xi[1];                 // coords[2]
    Vector3 xi_ordered = {xi0_ordered, xi1_ordered, xi2_ordered};

    // Transform barycentric to local coordinates
    const auto local = ReducedQuinticHelpers::barycentricToLocal(xi_ordered, order, a, b, c);
    const Real xi_local = local[0];
    const Real eta_local = local[1];
    
    // Compute polynomial basis and derivatives in local coordinates
    Real xi_pow[6], eta_pow[6];
    Real dxi_pow[6], deta_pow[6];
    
    xi_pow[0] = 1.0;  eta_pow[0] = 1.0;
    dxi_pow[0] = 0.0; deta_pow[0] = 0.0;
    
    for (int i = 1; i < 6; i++) {
      xi_pow[i] = xi_pow[i-1] * xi_local;
      eta_pow[i] = eta_pow[i-1] * eta_local;
      dxi_pow[i] = i * xi_pow[i-1];
      deta_pow[i] = i * eta_pow[i-1];
    }

    // Compute matrix for local to barycentric gradient chain rule
    Real J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int col = 0; col < 2; col++) {
        // d(lambda_k)/d(xi[col]) for k = 0, 1, 2
        Real dlambda[3];
        for (int k = 0; k < 3; k++) {
            if      (order[k] == col) dlambda[k] =  1.0;
            else if (order[k] == 2)   dlambda[k] = -1.0;
            else                      dlambda[k] =  0.0;
        }

        // d(xi_local)/d(xi[col])  = a*dlambda[1] - b*dlambda[0]
        // d(eta_local)/d(xi[col]) = c*dlambda[2]
        J[0][col] = a * dlambda[1] - b * dlambda[0];
        J[1][col] = c * dlambda[2];
    }
    
    // Evaluate gradients
    Kokkos::Array<Vector2, numNodes> grad_reordered;

    for (size_t k = 0; k < numNodes; k++) {
      Real dN_dxi_local  = 0.0;
      Real dN_deta_local = 0.0;

      for (int i = 0; i < 20; i++) {
        const auto poly = ReducedQuinticHelpers::getReducedQuinticPolyIdx(i);
        const int xi_idx  = poly[0];
        const int eta_idx = poly[1];
        const Real coeff  = elemCoeffs(8 + k * 20 + i);

        if (xi_idx > 0)
          dN_dxi_local +=
              coeff *
              dxi_pow[xi_idx] *
              eta_pow[eta_idx];

        if (eta_idx > 0)
          dN_deta_local +=
              coeff *
              xi_pow[xi_idx] *
              deta_pow[eta_idx];
      }

      // Apply chain rule to transform local gradients to barycentric gradients
      grad_reordered[k][0] =
          dN_dxi_local * J[0][0] +
          dN_deta_local * J[1][0];

      grad_reordered[k][1] =
          dN_dxi_local * J[0][1] +
          dN_deta_local * J[1][1];
    }

    // Convert back to meshFields vertex ordering
    Kokkos::Array<Vector2, numNodes> gradN_bary;

    for (int v = 0; v < 3; ++v) {
      const int orig_v = order[v];

      for (int d = 0; d < 6; ++d) {
        gradN_bary[orig_v * 6 + d] =
            grad_reordered[v * 6 + d];
      }
    }

    return gradN_bary;
  }
};

} // namespace MeshField
#endif
