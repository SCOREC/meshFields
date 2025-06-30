#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

// getValues(...) implementation copied from
// SCOREC/core apf/apfShape.cc @ 7cd76473
//
// TODO: define MeshField's canonical ordering using PUMI's, these shape
// function implementations require it. A lookup table should be defined for
// each (possibly following what omega_h does in Omega_h_element.hpp).

namespace {
template <typename Array> KOKKOS_INLINE_FUNCTION bool sumsToOne(Array &xi) {
  auto sum = 0.0;
  for (int i = 0; i < xi.size(); i++) {
    sum += xi[i];
  }
  return (Kokkos::fabs(sum - 1) <= MeshField::MachinePrecision);
}

template <typename Array>
KOKKOS_INLINE_FUNCTION bool greaterThanOrEqualZero(Array &xi) {
  auto gt = true;
  for (int i = 0; i < xi.size(); i++) {
    gt = gt && (xi[i] >= 0);
  }
  return gt;
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
            4*xi[0]*xi[1],
            4*xi[1]*xi3,
            4*xi[2]*xi3,
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
            4*xi[1],4*xi[0],0,
            -4*xi[1],4*xi3-4*xi[1],-4*xi[1],
            -4*xi[2],-4*xi[2],4*xi3-4*xi[2],
            4*xi[2],0,4*xi[0],
            0,4*xi[2],4*xi[1]};
    // clang-format on
  }
};

} // namespace MeshField
#endif
