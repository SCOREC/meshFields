#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

// getValues(...) implementation copied from
// SCOREC/core apf/apfShape.cc @ 7cd76473

namespace {
template <typename Array> KOKKOS_INLINE_FUNCTION bool lessThanOrEqualOne(Array &xi) {
  auto sum = 0.0;
  for (size_t i = 0; i < xi.size(); i++) {
    sum += xi[i];
  }
  return (Kokkos::fabs(sum - 1) <= MeshField::MachinePrecision) || (sum < 1);
}

template <typename Array>
KOKKOS_INLINE_FUNCTION bool greaterThanOrEqualZero(Array &xi) {
  auto gt = true;
  for (size_t i = 0; i < xi.size(); i++) {
    gt = gt && (xi[i] >= 0);
  }
  return gt;
}
} // namespace

namespace MeshField {

using Vector1 = Kokkos::Array<Real, 1>;
using Vector2 = Kokkos::Array<Real, 2>;
using Vector3 = Kokkos::Array<Real, 3>;
using Vector4 = Kokkos::Array<Real, 4>;

struct LinearEdgeShape {
  static const size_t numNodes = 2;
  static const size_t meshEntDim = 1;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes*meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {-1,  //node 0
            1}   //node 1
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector1 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    // clang-format off
    return {(1.0 - xi[0]) / 2.0,
            (1.0 + xi[0]) / 2.0};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getLocalGradients() const {
    // clang-format off
    return {-0.5,
             0.5};
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
  Kokkos::Array<Real, numNodes*meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {0,0,   //node 0
            1,0,   //node 1
            0,1};  //node 2
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(lessThanOrEqualOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1], //L0
            xi[0],  //L1
            xi[1]}; //L2
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
  Kokkos::Array<Real, numNodes*meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {0,0,   //node 0
            1,0,   //node 1
            0,1};  //node 2
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1],
            xi[0],
            xi[1]};
    // clang-format on
  }
};

struct QuadraticTriangleShape {
  // shape functions and ordering from 
  // Zienkiewicz, Taylor, and Zhu
  // 'The Finite Element Method: Its Basis and Fundamentals', 2013
  static const size_t numNodes = 6;
  static const size_t meshEntDim = 2;
  constexpr static Mesh_Topology DofHolders[2] = {Vertex, Edge};
  constexpr static size_t NumDofHolders[2] = {3, 3};
  constexpr static size_t DofsPerHolder[2] = {1, 1};
  constexpr static size_t Order = 2;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes*meshEntDim> getNodeParametricCoords() const {
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

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector2 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(lessThanOrEqualOne(xi));
    const Real L0 = 1 - xi[0] - xi[1];
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

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector2, numNodes> getLocalGradients(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    const Real L0 = 1 - xi[0] - xi[1];
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

struct LinearTetrahedronShape {
  static const size_t numNodes = 4;
  static const size_t meshEntDim = 3;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes*meshEntDim> getNodeParametricCoords() const {
    // clang-format off
    return {0,0,0,   //node 0
            1,0,0,   //node 1
            0,1,0,   //node 2
            0,0,1};  //node 3
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(lessThanOrEqualOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1] - xi[2],  //L0
            xi[0],  //L1
            xi[1],  //L2
            xi[2]}; //L3
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

struct QuadraticTetrahedronShape {
  // shape functions and ordering from 
  // Zienkiewicz, Taylor, and Zhu
  // 'The Finite Element Method: Its Basis and Fundamentals', 2013
  static const size_t numNodes = 10;
  static const size_t meshEntDim = 3;
  constexpr static Mesh_Topology DofHolders[2] = {Vertex, Edge};
  constexpr static size_t NumDofHolders[2] = {4, 6};
  constexpr static size_t DofsPerHolder[2] = {1, 1};
  constexpr static size_t Order = 2;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes*meshEntDim> getNodeParametricCoords() const {
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

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
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

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector3, numNodes> getLocalGradients(Vector4 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));
    const Real L0 = 1 - xi[0] - xi[1] - xi[2];
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
