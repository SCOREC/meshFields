#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

// getValues(...) implementation copied from
// SCOREC/core apf/apfShape.cc @ 7cd76473

namespace { template <typename Array> KOKKOS_INLINE_FUNCTION bool
	sumsToOne(Array &xi) { auto sum = 0.0; for (int i = 0; i < xi.size();
			i++) { sum += xi[i]; } return (Kokkos::fabs(sum - 1) <=
				MeshField::MachinePrecision); }

template <typename Array> KOKKOS_INLINE_FUNCTION bool
	greaterThanOrEqualZero(Array &xi) { auto gt = true; for (int i = 0; i <
			xi.size(); i++) { gt = gt && (xi[i] >= 0); } return gt;
	} } // namespace

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

  static constexpr int Order      = 5;
  static constexpr int meshEntDim = 2;
  static constexpr int numNodes   = 21;

  constexpr static Mesh_Topology DofHolders[1] = {Vertex};

  inline static constexpr double coeff[numNodes] = {
    // i = 0
     1,  5, 10, 10,  5,  1,
    // i = 1
     5, 20, 30, 20,  5,
    // i = 2
    10, 30, 30, 10,
    // i = 3
    10, 20, 10,
    // i = 4
     5,  5,
    // i = 5
     1
  };

  KOKKOS_INLINE_FUNCTION
  void computePowers(
      const Vector3& xi,
      Real& L1, Real& L2, Real& L3,
      Real (&p1)[Order+1],
      Real (&p2)[Order+1],
      Real (&p3)[Order+1]) const
  {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));

    L1 = 1.0 - xi[0] - xi[1];
    L2 = xi[0];
    L3 = xi[1];

    p1[0] = p2[0] = p3[0] = 1.0;
    for (int d = 1; d <= Order; ++d) {
      p1[d] = p1[d-1] * L1;
      p2[d] = p2[d-1] * L2;
      p3[d] = p3[d-1] * L3;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes>
  getValues(Vector3 const &xi) const
  {
    Real L1, L2, L3;
    Real p1[Order+1], p2[Order+1], p3[Order+1];

    computePowers(xi, L1, L2, L3, p1, p2, p3);

    Kokkos::Array<Real, numNodes> N;

    int idx = 0;
    for (int i = 0; i <= Order; ++i) {
      for (int j = 0; j <= Order - i; ++j) {
        int k = Order - i - j;

        N[idx] =
          coeff[idx] *
          p1[i] *
          p2[j] *
          p3[k];

        ++idx;
      }
    }

    return N;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector2, numNodes>
  getLocalGradients(Vector3 const &xi) const
  {
    Real L1, L2, L3;
    Real p1[Order+1], p2[Order+1], p3[Order+1];

    computePowers(xi, L1, L2, L3, p1, p2, p3);

    Kokkos::Array<Vector2, numNodes> dN;

    int idx = 0;
    for (int i = 0; i <= Order; ++i) {
      for (int j = 0; j <= Order - i; ++j) {
        int k = Order - i - j;

        const double c = coeff[idx];

        double dL1 = 0.0;
        double dL2 = 0.0;
        double dL3 = 0.0;

        if (i > 0)
          dL1 = c * i * p1[i-1] * p2[j]   * p3[k];

        if (j > 0)
          dL2 = c * j * p1[i]   * p2[j-1] * p3[k];

        if (k > 0)
          dL3 = c * k * p1[i]   * p2[j]   * p3[k-1];

        const double dNdX = -dL1 + dL2;
        const double dNdY = -dL1 + dL3;

        dN[idx][0] = dNdX;
        dN[idx][1] = dNdY;

        ++idx;
      }
    }

    return dN;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim>
  getLocalGradients() const
  {
    Vector3 xi = {1.0/3.0, 1.0/3.0, 1.0/3.0};

    auto g = getLocalGradients(xi);

    Kokkos::Array<Real, numNodes * meshEntDim> flat{};

    for (int n = 0; n < numNodes; ++n) {
      flat[2*n + 0] = g[n][0];
      flat[2*n + 1] = g[n][1];
    }

    return flat;
  }
};

} // namespace MeshField
#endif
