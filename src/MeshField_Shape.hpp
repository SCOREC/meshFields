#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

/**
 * @file MeshField_Shape.hpp
 * @brief Shape function definitions for finite element analysis
 *
 * This file defines shape functions for various mesh element types.
 * Each shape function struct provides methods to evaluate basis functions,
 * their gradients, and parametric coordinates of nodes.
 *
 * getValues(...) implementation based on
 * SCOREC/core apf/apfShape.cc @ 7cd76473
 */

namespace {
/** @brief Helper functions for parametric coordinate validation */

/**
 * @brief Check if value is greater than or equal to reference within tolerance
 * @param xi Value to check
 * @param val Reference value
 * @return true if xi >= val within machine precision
 */
KOKKOS_INLINE_FUNCTION bool greaterThanOrEqual(MeshField::Real xi, const MeshField::Real val, const MeshField::Real tol) {
  if ( xi > val ) return true;
  return (Kokkos::fabs(xi - val) <= tol);
}

/**
 * @brief Check if all array elements are greater than or equal to value
 * @tparam Array Array type
 * @param xi Array of values to check
 * @param val Reference value
 * @return true if all xi[i] >= val within machine precision
 */
template <typename Array>
KOKKOS_INLINE_FUNCTION bool eachGreaterThanOrEqual(Array &xi, const MeshField::Real val, const MeshField::Real tol) {
  auto gt = true;
  for (size_t i = 0; i < xi.size(); i++) {
    gt = gt && greaterThanOrEqual(xi[i],val,tol);
  }
  return gt;
}

/**
 * @brief Check if value is less than or equal to reference within tolerance
 * @param xi Value to check
 * @param val Reference value
 * @return true if xi <= val within machine precision
 */
KOKKOS_INLINE_FUNCTION bool lessThanOrEqual(MeshField::Real xi, const MeshField::Real val, const MeshField::Real tol) {
  if ( xi < val ) return true;
  return (Kokkos::fabs(xi - val) <= tol);
}

/**
 * @brief Check if all array elements are less than or equal to value
 * @tparam Array Array type
 * @param xi Array of values to check
 * @param val Reference value
 * @return true if all xi[i] <= val within machine precision
 */
template <typename Array>
KOKKOS_INLINE_FUNCTION bool eachLessThanOrEqual(Array &xi, const MeshField::Real val, const MeshField::Real tol) {
  auto lt = true;
  for (size_t i = 0; i < xi.size(); i++) {
    lt = lt && lessThanOrEqual(xi[i],val,tol);
  }
  return lt;
}

} // namespace

namespace MeshField {

/** @brief 1D parametric coordinate vector */
using Vector1 = Kokkos::Array<Real, 1>;
/** @brief 2D parametric coordinate vector */
using Vector2 = Kokkos::Array<Real, 2>;
/** @brief 3D parametric coordinate vector */
using Vector3 = Kokkos::Array<Real, 3>;

/**
 * @brief Linear (P1) shape functions for 1D edge elements
 *
 * Defines linear basis functions for a 1D edge element with 2 nodes.
 * Parametric coordinate range: xi \f$\in\f$ [-1, 1]
 *
 * Node ordering:
 * - Node 0 at xi = -1
 * - Node 1 at xi =  1
 *
 * Shape functions and ordering from:
 * Zienkiewicz, Taylor, and Zhu
 * 'The Finite Element Method: Its Basis and Fundamentals', 2013
 */
struct LinearEdgeShape {
  static const size_t numNodes = 2;         ///< Number of nodes (2 for linear edge)
  static const size_t meshEntDim = 1;       ///< Mesh entity dimension (1D)
  constexpr static Mesh_Topology DofHolders[1] = {Vertex}; ///< DOFs located at vertices
  constexpr static size_t Order = 1;        ///< Polynomial order (linear)

  /**
   * @brief Get parametric coordinates of element nodes
   * @return Array of node coordinates: [node0_xi, node1_xi]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {-1,  //node 0
            1};   //node 1
    // clang-format on
  }

  /**
   * @brief Evaluate shape functions at parametric coordinate
   * @param xi Parametric coordinate (xi \f$\in\f$ [-1, 1])
   * @return Array of shape function values [N0(xi), N1(xi)]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector1 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,-1.0,ParametricCoordTol));
    // clang-format off
    return {(1.0 - xi[0]) / 2.0,
            (1.0 + xi[0]) / 2.0};
    // clang-format on
  }

  /**
   * @brief Get gradients of shape functions in parametric coordinates
   * @detail gradients are constant - ignoring parametric coord input
   * @return Array of constant gradients [dN0/dxi, dN1/dxi]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getLocalGradients(Vector1 const &) const {
    // clang-format off
    return {-0.5,
             0.5};
    // clang-format on
  }
};

/**
 * @brief Linear (P1) shape functions for 2D triangular elements
 *
 * Defines linear basis functions for a triangular element with 3 nodes.
 * Parametric coordinate range: xi0, xi1 \f$\in\f$ [0, 1]
 *
 * Node ordering:
 * - Node 0 at (0, 0)
 * - Node 1 at (1, 0)
 * - Node 2 at (0, 1)
 *
 * Shape functions use barycentric coordinates where L0 = 1-xi0-xi1, L1 = xi0, L2 = xi1
 *
 * Shape functions and ordering from:
 * Zienkiewicz, Taylor, and Zhu
 * 'The Finite Element Method: Its Basis and Fundamentals', 2013
 */
struct LinearTriangleShape {
  static const size_t numNodes = 3;                         ///< Number of nodes (3 for linear triangle)
  static const size_t numComponentsPerDof = 1;              ///< Components per DOF (scalar field)
  static const size_t meshEntDim = 2;                       ///< Mesh entity dimension (2D)
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};  ///< DOFs located at vertices
  constexpr static size_t Order = 1;                        ///< Polynomial order (linear)

  /**
   * @brief Get parametric coordinates of element nodes
   * @return Array of node coordinates: [node0_xi0, node0_xi1, node1_xi0, node1_xi1, ...]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {0,0,   //node 0
            1,0,   //node 1
            0,1};  //node 2
    // clang-format on
  }

  /**
   * @brief Evaluate shape functions at parametric coordinate
   * @param xi Parametric coordinates (xi0, xi1) where xi0, xi1 \f$\in\f$ [0,1]
   * @return Array of shape function values [N0, N1, N2] = [L0, L1, L2]
   *         where L0 = 1-xi0-xi1, L1 = xi0, L2 = xi1
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    // clang-format off
    return {L0,
            xi[0],  //L1
            xi[1]}; //L2
    // clang-format on
  }

  /**
   * @brief Get gradients of shape functions
   * @detail gradients are constant - ignoring parametric coord input
   * @return Array of gradients [dN0/dxi0, dN0/dxi1, dN1/dxi0, dN1/dxi1, dN2/dxi0, dN2/dxi1]
   *         Constant gradients: [[-1,-1], [1,0], [0,1]]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, meshEntDim * numNodes> getLocalGradients(Vector2 const &) const {
    // clang-format off
    return { -1,-1,  //first vector
              1, 0,
              0, 1};
    // clang-format on
  }
};

/**
 * @brief Linear triangle shape functions for coordinate fields
 *
 * Similar to LinearTriangleShape but specialized for coordinate field usage.
 * Defines linear basis functions for a triangular element with 3 nodes.
 * Parametric coordinate range: xi0, xi1 \f$\in\f$ [0, 1]
 *
 * Shape functions and ordering from:
 * Zienkiewicz, Taylor, and Zhu
 * 'The Finite Element Method: Its Basis and Fundamentals', 2013
 */
struct LinearTriangleCoordinateShape {
  static const size_t numNodes = 3;                         ///< Number of nodes
  static const size_t meshEntDim = 2;                       ///< Mesh entity dimension (2D)
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};  ///< DOFs located at vertices
  constexpr static size_t Order = 1;                        ///< Polynomial order (linear)

  /**
   * @brief Get parametric coordinates of element nodes
   * @return Array of node coordinates: [node0_xi0, node0_xi1, node1_xi0, node1_xi1, node2_xi0, node2_xi1]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {0,0,   //node 0
            1,0,   //node 1
            0,1};  //node 2
    // clang-format on
  }

  /**
   * @brief Evaluate shape functions at parametric coordinate
   * @param xi Parametric coordinates (xi0, xi1) where xi0, xi1 \f$\in\f$ [0,1]
   * @return Array of shape function values using barycentric coordinates [L0, L1, L2]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    // clang-format off
    return {L0,
            xi[0],
            xi[1]};
    // clang-format on
  }
};

/**
 * @brief Quadratic (P2) shape functions for 2D triangular elements
 *
 * Defines quadratic basis functions for a triangular element with 6 nodes.
 * Nodes are located at vertices and edge midpoints.
 * Parametric coordinate range: xi0, xi1 \f$\in\f$ [0, 1]
 *
 * Node ordering:
 * - Nodes 0-2: Triangle vertices
 * - Nodes 3-5: Edge midpoints
 *
 * Shape functions and ordering from:
 * Zienkiewicz, Taylor, and Zhu
 * 'The Finite Element Method: Its Basis and Fundamentals', 2013
 */
//! [QuadraticTriangleShape]
struct QuadraticTriangleShape {
  static const size_t numNodes = 6;                              ///< Number of nodes (6 for quadratic triangle)
  static const size_t meshEntDim = 2;                            ///< Mesh entity dimension (2D)
  constexpr static Mesh_Topology DofHolders[2] = {Vertex, Edge}; ///< DOFs at vertices and edges
  constexpr static size_t NumDofHolders[2] = {3, 3};             ///< 3 vertices, 3 edges
  constexpr static size_t DofsPerHolder[2] = {1, 1};             ///< 1 DOF per vertex/edge
  constexpr static size_t Order = 2;                             ///< Polynomial order (quadratic)

  /**
   * @brief Get parametric coordinates of element nodes
   * @return Array of node coordinates for 6 nodes (3 at vertices, 3 at edge midpoints)
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {
      //nodes at vertices
      0   , 0   , //node 0
      1   , 0   , //node 1
      0   , 1   , //...
      //nodes at middle of edges
      0.5 , 0   ,
      0.5 , 0.5 ,
      0   , 0.5   //node 5
    };
    // clang-format on
  }

  /**
   * @brief Evaluate quadratic shape functions at parametric coordinate
   * @param xi Parametric coordinates (xi0, xi1) where xi0, xi1 \f$\in\f$ [0,1]
   * @return Array of 6 shape function values for quadratic triangle
   *         N_i = L_i(2L_i - 1) for vertices (i=0,1,2)
   *         N_i = 4*L_j*L_k for edge midpoints (i=3,4,5)
   *         where L0=1-xi0-xi1, L1=xi0, L2=xi1 are barycentric coordinates
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    // clang-format off
    return {L0 * (2 * L0 - 1),
            L1 * (2 * L1 - 1),
            L2 * (2 * L2 - 1),
            4 * L1 * L0,
            4 * L1 * L2,
            4 * L2 * L0};
    // clang-format on
  }

  /**
   * @brief Get gradients of quadratic shape functions at parametric coordinate
   * @param xi Parametric coordinates (xi0, xi1) where xi0, xi1 \f$\in\f$ [0,1]
   * @return Array of gradient vectors [dN0/dxi0, dN0/dxi1, ..., dN5/dxi0, dN5/dxi1]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, meshEntDim * numNodes>
  getLocalGradients(Vector2 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    // clang-format off
    return {
      -4*L0+1   , -4*L0+1   ,
      4*L1-1    , 0         ,
      0         , 4*L2-1    ,
      4*(L0-L1) , -4*L1     ,
      4*L2      , 4*L1      ,
      -4*L2     , 4*(L0-L2)
    };
    // clang-format on
  }
};
//! [QuadraticTriangleShape]

/**
 * @brief Linear (P1) shape functions for 3D tetrahedral elements
 *
 * Defines linear basis functions for a tetrahedral element with 4 nodes.
 * Parametric coordinate range: xi0, xi1, xi2 \f$\in\f$ [0, 1]
 *
 * Node ordering:
 * - Node 0 at (0, 0, 0)
 * - Node 1 at (1, 0, 0)
 * - Node 2 at (0, 1, 0)
 * - Node 3 at (0, 0, 1)
 *
 * Shape functions use barycentric coordinates L0, L1, L2, L3
 * where L0 = 1-xi0-xi1-xi2, L1 = xi0, L2 = xi1, L3 = xi2
 *
 * Shape functions and ordering from:
 * Zienkiewicz, Taylor, and Zhu
 * 'The Finite Element Method: Its Basis and Fundamentals', 2013
 */
struct LinearTetrahedronShape {
  static const size_t numNodes = 4;                         ///< Number of nodes (4 for linear tetrahedron)
  static const size_t meshEntDim = 3;                       ///< Mesh entity dimension (3D)
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};  ///< DOFs located at vertices
  constexpr static size_t Order = 1;                        ///< Polynomial order (linear)

  /**
   * @brief Get parametric coordinates of element nodes
   * @return Array of node coordinates for 4 vertices
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {0,0,0,   //node 0
            1,0,0,   //node 1
            0,1,0,   //node 2
            0,0,1};  //node 3
    // clang-format on
  }

  /**
   * @brief Evaluate shape functions at parametric coordinate
   * @param xi Parametric coordinates (xi0, xi1, xi2) where xi_i \f$\in\f$ [0,1]
   * @return Array of shape function values [N0, N1, N2, N3] = [L0, L1, L2, L3]
   *         where L0 = 1-xi0-xi1-xi2, L1 = xi0, L2 = xi1, L3 = xi2
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    // clang-format off
    return {L0,
            xi[0],  //L1
            xi[1],  //L2
            xi[2]}; //L3
    // clang-format on
  }

  /**
   * @brief Get gradients of shape functions in parametric coordinates
   * @detail gradients are constant - ignoring parametric coord input
   * @return Array of constant gradients [dN0/dxi0, dN0/dxi1, dN0/dxi2, dN1/dxi0, ...]
   *         Constant gradients: [[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, meshEntDim * numNodes> getLocalGradients(Vector3 const &) const {
    // clang-format off
    return {-1, -1, -1,
             1,  0,  0,
             0,  1,  0,
             0,  0,  1};
    // clang-format on
  }
};

/**
 * @brief Quadratic (P2) shape functions for 3D tetrahedral elements
 *
 * Defines quadratic basis functions for a tetrahedral element with 10 nodes.
 * Nodes are located at vertices and edge midpoints.
 * Parametric coordinate range: xi0, xi1, xi2 \f$\in\f$ [0, 1]
 *
 * Node ordering:
 * - Nodes 0-3: Tetrahedron vertices
 * - Nodes 4-9: Edge midpoints
 *
 * Shape functions and ordering from:
 * Zienkiewicz, Taylor, and Zhu
 * 'The Finite Element Method: Its Basis and Fundamentals', 2013
 */
struct QuadraticTetrahedronShape {
  static const size_t numNodes = 10;                             ///< Number of nodes (10 for quadratic tetrahedron)
  static const size_t meshEntDim = 3;                            ///< Mesh entity dimension (3D)
  constexpr static Mesh_Topology DofHolders[2] = {Vertex, Edge}; ///< DOFs at vertices and edges
  constexpr static size_t NumDofHolders[2] = {4, 6};             ///< 4 vertices, 6 edges
  constexpr static size_t DofsPerHolder[2] = {1, 1};             ///< 1 DOF per vertex/edge
  constexpr static size_t Order = 2;                             ///< Polynomial order (quadratic)

  /**
   * @brief Get parametric coordinates of element nodes
   * @return Array of node coordinates for 10 nodes (4 at vertices, 6 at edge midpoints)
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {
      //nodes at vertices
      0,0,0,   //node 0
      1,0,0,   //node 1
      0,1,0,   //...
      0,0,1,   //node 3
      //nodes at middle of edges
      0.5 , 0   , 0   ,
      0   , 0.5 , 0   ,
      0   , 0   , 0.5 ,
      0.5 , 0.5 , 0   ,
      0   , 0.5 , 0.5 ,
      0.5 , 0   , 0.5
    };
    // clang-format on
  }

  /**
   * @brief Evaluate quadratic shape functions at parametric coordinate
   * @param xi Parametric coordinates (xi0, xi1, xi2) where xi_i \f$\in\f$ [0,1]
   * @return Array of 10 shape function values for quadratic tetrahedron
   *         N_i = L_i(2L_i - 1) for vertices (i=0,1,2,3)
   *         N_i = 4*L_j*L_k for edge midpoints (i=4-9)
   *         where L0=1-xi0-xi1-xi2, L1=xi0, L2=xi1, L3=xi2 are barycentric coordinates
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    const Real L3 = xi[2];
    // clang-format off
    return {L0*(2*L0-1),
            L1*(2*L1-1),
            L2*(2*L2-1),
            L3*(2*L3-1),
            4*L1*L0,
            4*L2*L0,
            4*L3*L0,
            4*L1*L2,
            4*L2*L3,
            4*L1*L3};
    // clang-format on
  }

  /**
   * @brief Get gradients of quadratic shape functions in parametric coordinates
   * @param xi Parametric coordinates (xi0, xi1, xi2) where xi_i \f$\in\f$ [0,1]
   * @return Array of gradient vectors [dN0/dxi0, dN0/dxi1, dN0/dxi2, ..., dN9/dxi0, dN9/dxi1, dN9/dxi2]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, meshEntDim * numNodes>
  getLocalGradients(Vector3 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    const Real L3 = xi[2];

    const Real d3 = 1 - 4 * L0;
    // clang-format off
    return {
      d3        , d3        , d3        ,
      4*L1-1    , 0         , 0         ,
      0         , 4*L2-1    , 0         ,
      0         , 0         , 4*L3-1    ,
      4*L0-4*L1 , -4*L1     , -4*L1     ,
      -4*L2     , 4*L0-4*L2 , -4*L2     ,
      -4*L3     , -4*L3     , 4*L0-4*L3 ,
      4*L2      , 4*L1      , 0         ,
      0         , 4*L3      , 4*L2      ,
      4*L3      , 0         , 4*L1
    };
    // clang-format on
  }
};

/**
 * @brief Helper functions for reduced quintic coordinate transformations
 */
namespace ReducedQuinticHelpers {
  /**
   * @brief Transform parametric coordinates to reduced quintic local coordinates
   * 
   * local coordinate system (origin at foot of perpendicular):
   *   - v0 is at (-b, 0)
   *   - v1 is at (a, 0)
   *   - v2 is at (0, c)
   * 
   * @param xi Parametric coordinates (xi0, xi1) where L0=1-xi0-xi1, L1=xi0, L2=xi1
   * @param order Vertex reordering [3] - order[i] gives original index
   * @param a Distance from origin to v1
   * @param b Distance from origin to v0
   * @param c Perpendicular distance from origin to v2
   * @return Vector2 containing [xi_local, eta_local]
   */
  KOKKOS_INLINE_FUNCTION
  Vector2 parametricToLocal(
      Vector2 const& xi,
      int const order[3],
      Real a,
      Real b,
      Real c)
  {
    assert(eachLessThanOrEqual(xi,1.0,ParametricCoordTol));
    assert(eachGreaterThanOrEqual(xi,0.0,ParametricCoordTol));
    // First compute barycentric coordinates: L0 = 1-xi[0]-xi[1], L1 = xi[0], L2 = xi[1]
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0,ParametricCoordTol));
    assert(lessThanOrEqual(L0,1.0,ParametricCoordTol));
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    
    // Create array of barycentric coordinates
    Real bary[3] = {L0, L1, L2};
    
    // Reorder barycentric coordinates according to vertex ordering
    const Real lambda0 = bary[order[0]];
    const Real lambda1 = bary[order[1]];
    const Real lambda2 = bary[order[2]];

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
 * Implements the method described in:
 *   S.C. Jardin, "A triangular finite element with first-derivative continuity
 *   applied to fusion MHD applications," Journal of Computational Physics,
 *   Volume 200, Issue 1, 2004, Pages 133-152,
 *   https://doi.org/10.1016/j.jcp.2004.04.004.
 * 
 * This implementation follows the M3DC1 codebase and uses a per-element local
 * coordinate system. Unlike standard reference-element shapes (LinearTriangle,
 * QuadraticTriangle, etc.), each ReducedQuintic element has geometry-dependent
 * shape functions.
 * 
 * This element uses 18 nodes (3 vertices × 6 DOFs per vertex):
 * - DOFs per vertex: [value, d/dx, d/dy, d^2/dx^2, d^2/dxdy, d^2/dy^2]
 * - Polynomial order: 5 (20-term basis: xi^i * eta^j for i+j ≤ 5)
 * 
 * COORDINATE TRANSFORMATION:
 * The longest triangle edge is reordered to lie along the local xi-axis.
 * The origin is placed at the foot of perpendicular from v2 onto the v0-v1 edge.
 *   - v0 is at (-b, 0) in local coords, where b = distance from origin to v0
 *   - v1 is at (a, 0) in local coords, where a = distance from origin to v1
 *   - v2 is at (0, c) in local coords, where c = perpendicular distance to v2
 * 
 * meshFields API uses parametric coordinates: (xi0, xi1) where barycentric coordinates
 * are computed as L0 = 1-xi0-xi1, L1 = xi0, L2 = xi1.
 * 
 * Transformation to local coordinates:
 *   xi_local = a*λ1 - b*λ0
 *   eta_local = c*λ2
 * 
 * Applied automatically via helper function parametricToLocal()
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
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi,
                                          Kokkos::View<const Real*, Kokkos::LayoutStride> elemCoeffs) const {

    // Extract geometric parameters from coefficient array
    // elemCoeffs layout: [order[0], order[1], order[2], a, b, c, sin_theta, cos_theta, coeff_0_0, ..., coeff_17_19]
    const int order[3] = {static_cast<int>(elemCoeffs(0)), 
                          static_cast<int>(elemCoeffs(1)), 
                          static_cast<int>(elemCoeffs(2))};
    const Real a = elemCoeffs(3);
    const Real b = elemCoeffs(4);
    const Real c = elemCoeffs(5);

    // Transform parametric to local coordinates
    const auto local = ReducedQuinticHelpers::parametricToLocal(xi, order, a, b, c);
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

  // Computes the local gradients of the Reduced Quintic shape functions.
  //
  // These gradients are not currently used in the MeshField evaluation pipeline
  // because the Reduced Quintic element uses a linear geometric mapping. The
  // Jacobian is therefore computed from the linear geometry rather than from the
  // Reduced Quintic shape functions.
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector2, numNodes> getLocalGradients(Vector2 const &xi,
                                                      Kokkos::View<const Real*, Kokkos::LayoutStride> elemCoeffs) const {
    
    // Extract geometric parameters
    const int order[3] = {static_cast<int>(elemCoeffs(0)), 
                          static_cast<int>(elemCoeffs(1)), 
                          static_cast<int>(elemCoeffs(2))};
    const Real a = elemCoeffs(3);
    const Real b = elemCoeffs(4);
    const Real c = elemCoeffs(5);

    // Transform parametric to local coordinates
    const auto local = ReducedQuinticHelpers::parametricToLocal(xi, order, a, b, c);
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
