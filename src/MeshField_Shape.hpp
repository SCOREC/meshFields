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
KOKKOS_INLINE_FUNCTION bool greaterThanOrEqual(Real xi, const Real val) {
  if ( xi > val ) return true;
  return (Kokkos::fabs(xi - val) <= MeshField::MachinePrecision);
}

/**
 * @brief Check if all array elements are greater than or equal to value
 * @tparam Array Array type
 * @param xi Array of values to check
 * @param val Reference value
 * @return true if all xi[i] >= val within machine precision
 */
template <typename Array>
KOKKOS_INLINE_FUNCTION bool eachGreaterThanOrEqual(Array &xi, const Real val) {
  auto gt = true;
  for (size_t i = 0; i < xi.size(); i++) {
    gt = gt && greaterThanOrEqual(xi,val);
  }
  return gt;
}

/**
 * @brief Check if value is less than or equal to reference within tolerance
 * @param xi Value to check
 * @param val Reference value
 * @return true if xi <= val within machine precision
 */
KOKKOS_INLINE_FUNCTION bool lessThanOrEqual(Real xi, const Real val) {
  if ( xi < val ) return true;
  return (Kokkos::fabs(xi - val) <= MeshField::MachinePrecision);
}

/**
 * @brief Check if all array elements are less than or equal to value
 * @tparam Array Array type
 * @param xi Array of values to check
 * @param val Reference value
 * @return true if all xi[i] <= val within machine precision
 */
template <typename Array>
KOKKOS_INLINE_FUNCTION bool eachLessThanOrEqual(Array &xi, const Real val) {
  auto lt = true;
  for (size_t i = 0; i < xi.size(); i++) {
    lt = lt && lessThanOrEqual(xi,val);
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
            1}   //node 1
    // clang-format on
  }

  /**
   * @brief Evaluate shape functions at parametric coordinate
   * @param xi Parametric coordinate (xi \f$\in\f$ [-1, 1])
   * @return Array of shape function values [N0(xi), N1(xi)]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector1 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,-1.0));
    // clang-format off
    return {(1.0 - xi[0]) / 2.0,
            (1.0 + xi[0]) / 2.0};
    // clang-format on
  }

  /**
   * @brief Get gradients of shape functions in parametric coordinates
   * @return Array of constant gradients [dN0/dxi, dN1/dxi]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getLocalGradients() const {
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
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,0.0));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0));
    assert(lessThanOrEqual(L0,1.0));
    // clang-format off
    return {L0,
            xi[0],  //L1
            xi[1]}; //L2
    // clang-format on
  }

  /**
   * @brief Get gradients of shape functions
   * @return Array of gradients [dN0/dxi0, dN0/dxi1, dN1/dxi0, dN1/dxi1, dN2/dxi0, dN2/dxi1]
   *         Constant gradients: [[-1,-1], [1,0], [0,1]]
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, meshEntDim * numNodes> getLocalGradients() const {
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
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,0.0));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0));
    assert(lessThanOrEqual(L0,1.0));
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
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,0.0));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0));
    assert(lessThanOrEqual(L0,1.0));
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
  Kokkos::Array<Vector2, numNodes> getLocalGradients(Vector3 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,0.0));
    const Real L0 = 1 - xi[0] - xi[1];
    assert(greaterThanOrEqual(L0,0.0));
    assert(lessThanOrEqual(L0,1.0));
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
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,0.0));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
    assert(greaterThanOrEqual(L0,0.0));
    assert(lessThanOrEqual(L0,1.0));
    // clang-format off
    return {L0,
            xi[0],  //L1
            xi[1],  //L2
            xi[2]}; //L3
    // clang-format on
  }

  /**
   * @brief Get gradients of shape functions in parametric coordinates
   * @return Array of constant gradients [dN0/dxi0, dN0/dxi1, dN0/dxi2, dN1/dxi0, ...]
   *         Constant gradients: [[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]]
   */
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
    }
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
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,0.0));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
    assert(greaterThanOrEqual(L0,0.0));
    assert(lessThanOrEqual(L0,1.0));
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    const Real L3 = xi[1];
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
  Kokkos::Array<Vector3, numNodes> getLocalGradients(Vector3 const &xi) const {
    assert(eachLessThanOrEqual(xi,1.0));
    assert(eachGreaterThanOrEqual(xi,0.0));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
    assert(greaterThanOrEqual(L0,0.0));
    assert(lessThanOrEqual(L0,1.0));
    const Real L1 = xi[0];
    const Real L2 = xi[1];
    const Real L3 = xi[1];

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

} // namespace MeshField
#endif
