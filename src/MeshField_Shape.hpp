#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

// getValues(...) implementation copied from SCOREC/core apf/apfShape.cc @
// 7cd76473

namespace { template <typename Array> KOKKOS_INLINE_FUNCTION bool
	sumsToOne(Array &xi) { auto sum = 0.0; for (int i = 0; i < xi.size();
			i++) { sum += xi[i]; } return (Kokkos::fabs(sum - 1) <=
				MeshField::MachinePrecision); }

template <typename Array> KOKKOS_INLINE_FUNCTION bool
	greaterThanOrEqualZero(Array &xi) { auto gt = true; for (int i = 0; i <
			xi.size(); i++) { gt = gt && (xi[i] >= 0); } return gt;
	} } // namespace

namespace MeshField {

using Vector2 = Kokkos::Array<Real, 2>; using Vector3 = Kokkos::Array<Real, 3>;
using Vector4 = Kokkos::Array<Real, 4>;

struct LinearEdgeShape { static const size_t numNodes = 2; static const size_t
	meshEntDim = 1; constexpr static Mesh_Topology DofHolders[1] =
	{Vertex}; constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, numNodes> getValues(Vector2 const
		  &xi) const { assert(greaterThanOrEqualZero(xi));
	  assert(sumsToOne(xi));
    // clang-format off
    return {(1.0 - xi[0]) / 2.0, (1.0 + xi[0]) / 2.0};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, numNodes> getLocalGradients()
	  const {
    // clang-format off
    return {-0.5, 0.5};
    // clang-format on
  } };

struct LinearTriangleShape { static const size_t order = 1; static const size_t
	numNodes = 3; static const size_t numComponentsPerDof = 1; static const
		size_t meshEntDim = 2; constexpr static Mesh_Topology
		DofHolders[1] = {Vertex}; constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, numNodes> getValues(Vector3 const
		  &xi) const { assert(greaterThanOrEqualZero(xi));
	  assert(sumsToOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1], xi[0], xi[1]};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, meshEntDim * numNodes>
	  getLocalGradients() const {
    // clang-format off
    return { -1,-1,  //first vector 1, 0, 0, 1};
    // clang-format on
  } };

struct LinearTriangleCoordinateShape { static const size_t numNodes = 3; static
	const size_t meshEntDim = 2; constexpr static Mesh_Topology
		DofHolders[1] = {Vertex}; constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, numNodes> getValues(Vector3 const
		  &xi) const { assert(greaterThanOrEqualZero(xi));
	  assert(sumsToOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1], xi[0], xi[1]};
    // clang-format on
  } };

struct LinearTetrahedronShape { static const size_t numNodes = 4; static const
	size_t meshEntDim = 3; constexpr static Mesh_Topology DofHolders[1] =
	{Vertex}; constexpr static size_t Order = 1;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, numNodes> getValues(Vector4 const
		  &xi) const { assert(greaterThanOrEqualZero(xi));
	  assert(sumsToOne(xi));
    // clang-format off
    return {1 - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, meshEntDim * numNodes>
	  getLocalGradients() const {
    // clang-format off
    return {-1, -1, -1, 1,  0,  0, 0,  1,  0, 0,  0,  1};
    // clang-format on
  } };

struct QuadraticTriangleShape { static const size_t numNodes = 6; static const
	size_t meshEntDim = 2; constexpr static Mesh_Topology DofHolders[2] =
	{Vertex, Edge}; constexpr static size_t NumDofHolders[2] = {3, 3};
	constexpr static size_t DofsPerHolder[2] = {1, 1}; constexpr static
		size_t Order = 2;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, numNodes> getValues(Vector3 const
		  &xi) const { assert(greaterThanOrEqualZero(xi));
	  assert(sumsToOne(xi)); const Real xi2 = 1 - xi[0] - xi[1];
    // clang-format off
    return {xi2 * (2 * xi2 - 1), xi[0] * (2 * xi[0] - 1), xi[1] * (2 * xi[1] -
		    1), 4 * xi[0] * xi2, 4 * xi[0] * xi[1], 4 * xi[1] * xi2};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Vector2, numNodes>
	  getLocalGradients(Vector3 const &xi) const {
		  assert(greaterThanOrEqualZero(xi)); assert(sumsToOne(xi));
		  const Real xi2 = 1 - xi[0] - xi[1];
    // clang-format off
    return {-4*xi2+1,-4*xi2+1, 4*xi[0]-1,0, 0,4*xi[1]-1,
	    4*(xi2-xi[0]),-4*xi[0], 4*xi[1],4*xi[0], -4*xi[1],4*(xi2-xi[1]) };
    // clang-format on
  } };

struct QuadraticTetrahedronShape { static const size_t numNodes = 10; static
	const size_t meshEntDim = 3; constexpr static Mesh_Topology
		DofHolders[2] = {Vertex, Edge}; constexpr static size_t
		NumDofHolders[2] = {4, 6}; constexpr static size_t
		DofsPerHolder[2] = {1, 1}; constexpr static size_t Order = 2;
  // ordering taken from mfem see mfem/mfem fem/fe/fe_fixed_order.cpp @597cba8
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, numNodes> getValues(Vector4 const
		  &xi) const { assert(greaterThanOrEqualZero(xi));
	  assert(sumsToOne(xi)); const Real xi3 = 1 - xi[0] - xi[1] - xi[2];
    // clang-format off
    return {xi3*(2*xi3-1), xi[0]*(2*xi[0]-1), xi[1]*(2*xi[1]-1),
	    xi[2]*(2*xi[2]-1), 4*xi[0]*xi3, 4*xi[1]*xi3, 4*xi[2]*xi3,
	    4*xi[0]*xi[1], 4*xi[2]*xi[0], 4*xi[1]*xi[2]};
    // clang-format on
  }

  KOKKOS_INLINE_FUNCTION Kokkos::Array<Vector3, numNodes>
	  getLocalGradients(Vector4 const &xi) const {
		  assert(greaterThanOrEqualZero(xi)); assert(sumsToOne(xi));
		  const Real xi3 = 1 - xi[0] - xi[1] - xi[2]; const Real d3 = 1
			  - 4 * xi3;
    // clang-format off
    return {d3,d3,d3, 4*xi[0]-1,0,0, 0,4*xi[1]-1,0, 0,0,4*xi[2]-1,
	    4*xi3-4*xi[0],-4*xi[0],-4*xi[0], -4*xi[1],4*xi3-4*xi[1],-4*xi[1],
	    -4*xi[2],-4*xi[2],4*xi3-4*xi[2], 4*xi[1],4*xi[0],0,
	    4*xi[2],0,4*xi[0], 0,4*xi[2],4*xi[1]};
    // clang-format on
  } };

struct ReducedQuinticImplicitShape {
  static const size_t numNodes = 21;
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

    Real powL1[6], powL2[6], powL3[6];
    powL1[0] = powL2[0] = powL3[0] = 1.0;
    for (int p = 1; p <= 5; ++p) {
      powL1[p] = powL1[p - 1] * L1;
      powL2[p] = powL2[p - 1] * L2;
      powL3[p] = powL3[p - 1] * L3;
    }

    auto fact = [](int n) {
      double r = 1.0;
      for (int i = 2; i <= n; ++i) r *= double(i);
      return r;
    };
    const double f5 = fact(5);

    Kokkos::Array<Real, numNodes> N;
    int idx = 0;
    for (int i = 0; i <= 5; ++i) {
      for (int j = 0; j <= 5 - i; ++j) {
        int k = 5 - i - j;
        double coeff = f5 / (fact(i) * fact(j) * fact(k));
        N[idx++] = coeff * powL1[i] * powL2[j] * powL3[k];
      }
    }
    return N;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Vector2, numNodes> getLocalGradients(Vector3 const &xi) const {
    assert(greaterThanOrEqualZero(xi));
    assert(sumsToOne(xi));

    const Real L1 = 1.0 - xi[0] - xi[1];
    const Real L2 = xi[0];
    const Real L3 = xi[1];

    Real powL1[6], powL2[6], powL3[6];
    powL1[0] = powL2[0] = powL3[0] = 1.0;
    for (int p = 1; p <= 5; ++p) {
      powL1[p] = powL1[p - 1] * L1;
      powL2[p] = powL2[p - 1] * L2;
      powL3[p] = powL3[p - 1] * L3;
    }

    auto fact = [](int n) {
      double r = 1.0;
      for (int i = 2; i <= n; ++i) r *= double(i);
      return r;
    };
    const double f5 = fact(5);

    Kokkos::Array<Vector2, numNodes> dN;
    int idx = 0;
    for (int i = 0; i <= 5; ++i) {
      for (int j = 0; j <= 5 - i; ++j) {
        int k = 5 - i - j;
        double coeff = f5 / (fact(i) * fact(j) * fact(k));

        double dN_dL1 = 0.0;
        double dN_dL2 = 0.0;
        double dN_dL3 = 0.0;
        if (i > 0) dN_dL1 = coeff * double(i) * powL1[i - 1] * powL2[j] * powL3[k];
        if (j > 0) dN_dL2 = coeff * double(j) * powL1[i] * powL2[j - 1] * powL3[k];
        if (k > 0) dN_dL3 = coeff * double(k) * powL1[i] * powL2[j] * powL3[k - 1];

        const double dNdX = -dN_dL1 + dN_dL2; // xi[0] corresponds to L2
        const double dNdY = -dN_dL1 + dN_dL3; // xi[1] corresponds to L3

        dN[idx][0] = dNdX;
        dN[idx][1] = dNdY;
        ++idx;
      }
    }
    return dN;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, numNodes * meshEntDim> getLocalGradients() const {
    Vector3 xi = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    auto gvec = getLocalGradients(xi);
    Kokkos::Array<Real, numNodes * meshEntDim> flat{};
    for (int n = 0; n < (int)numNodes; ++n) {
      flat[n * meshEntDim + 0] = gvec[n][0];
      flat[n * meshEntDim + 1] = gvec[n][1];
    }
    return flat;
  }
};

} // namespace MeshField
#endif
