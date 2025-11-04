#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

// getValues(...) implementation copied from
// SCOREC/core apf/apfShape.cc @ 7cd76473

namespace {
template <typename Array> KOKKOS_INLINE_FUNCTION bool sumsToOne(Array &xi) {
  auto sum = 0.0;
  for (size_t i = 0; i < xi.size(); i++) {
    sum += xi[i];
  }
  return (Kokkos::fabs(sum - 1) <= MeshField::MachinePrecision);
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

struct ReducedQuinticImplicitShape {
  static const size_t numNodes = 18;
  static const size_t meshEntDim = 2;
  constexpr static Mesh_Topology DofHolders[1] = {Vertex};
  constexpr static size_t Order = 5;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes> getValues(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));

    const Real L1 = 1.0 - xi[0] - xi[1];
    const Real L2 = xi[0];
    const Real L3 = xi[1];

    const Real L1_2 = L1 * L1;
    const Real L2_2 = L2 * L2;
    const Real L3_2 = L3 * L3;
    const Real L1_3 = L1_2 * L1;
    const Real L2_3 = L2_2 * L2;
    const Real L3_3 = L3_2 * L3;
    const Real L1_4 = L1_3 * L1;
    const Real L2_4 = L2_3 * L2;
    const Real L3_4 = L3_3 * L3;
    const Real L1_5 = L1_4 * L1;
    const Real L2_5 = L2_4 * L2;
    const Real L3_5 = L3_4 * L3;

    Kokkos::Array<Real, numNodes> N;

    N[0] = L1 * (2 * L1 - 1) * (3 * L1 - 1) * (4 * L1 - 1) * (5 * L1 - 1);
    N[1] = L2 * (2 * L2 - 1) * (3 * L2 - 1) * (4 * L2 - 1) * (5 * L2 - 1);
    N[2] = L3 * (2 * L3 - 1) * (3 * L3 - 1) * (4 * L3 - 1) * (5 * L3 - 1);
    
    N[3] = 5.0 * L1_4 * L2;
    N[4] = 5.0 * L2_4 * L1;
    N[5] = 5.0 * L3_4 * L1;

    N[6] = 5.0 * L1_4 * L3;
    N[7] = 5.0 * L2_4 * L3;
    N[8] = 5.0 * L3_4 * L2;

    N[9]  = 10.0 * L1_3 * L2_2;
    N[10] = 10.0 * L2_3 * L1_2;
    N[11] = 10.0 * L3_3 * L1_2;

    N[12] = 10.0 * L1_3 * L3_2;
    N[13] = 10.0 * L2_3 * L3_2;
    N[14] = 10.0 * L3_3 * L2_2;

    N[15] = 20.0 * L1_2 * L2 * L3;
    N[16] = 20.0 * L2_2 * L1 * L3;
    N[17] = 20.0 * L3_2 * L1 * L2;

    return N;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector2, numNodes> getLocalGradients(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));

    const Real L1 = 1.0 - xi[0] - xi[1];
    const Real L2 = xi[0];
    const Real L3 = xi[1];

    const Real eps = 1e-6;

    Vector3 xp = {xi[0] + eps, xi[1], 0.0};
    Vector3 yp = {xi[0], xi[1] + eps, 0.0};

    auto N0 = getValues(xi);
    auto Nx = getValues(xp);
    auto Ny = getValues(yp);

    Kokkos::Array<Vector2, numNodes> dN;
    for (int i = 0; i < numNodes; i++) {
      dN[i][0] = (Nx[i] - N0[i]) / eps;
      dN[i][1] = (Ny[i] - N0[i]) / eps;
    }
    return dN;
  }
};

} // namespace MeshField
#endif
