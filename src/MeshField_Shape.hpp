#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>

// getValues(...) implementation copied from SCOREC/core apf/apfShape.cc @ 7cd76473

namespace MeshField {
struct LinearEdgeShape {
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real,2> getValues(Kokkos::Array<Real, 2> const& xi) const {
    return {
      (1.0-xi[0])/2.0,
      (1.0+xi[0])/2.0
    };
  }
  static const size_t numNodes = 2;
  static const size_t numComponentsPerDof = 1;
  static const size_t meshEntDim = 1;
  constexpr static MeshField::Mesh_Topology DofHolders[1] = {MeshField::Vertex};
};

struct LinearTriangleShape {
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real,3> getValues(Kokkos::Array<Real, 3> const& xi) const {
    return {
      1-xi[0]-xi[1],
      xi[0],
      xi[1]
    };
  }
  static const size_t numNodes = 3;
  static const size_t numComponentsPerDof = 1;
  static const size_t meshEntDim = 2;
  constexpr static MeshField::Mesh_Topology DofHolders[1] = {MeshField::Vertex};
};

struct QuadraticTriangleShape { 
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real,6> getValues(Kokkos::Array<Real, 3> const& xi) const {
    const Real xi2 = 1-xi[0]-xi[1];
    return {
      xi2*(2*xi2-1),
      xi[0]*(2*xi[0]-1),
      xi[1]*(2*xi[1]-1),
      4*xi[0]*xi2,
      4*xi[0]*xi[1],
      4*xi[1]*xi2
    };
  }

  static const size_t numNodes = 6;
  static const size_t numComponentsPerDof = 1;
  static const size_t meshEntDim = 2;
  constexpr static MeshField::Mesh_Topology DofHolders[2] = {MeshField::Vertex, MeshField::Edge};
  constexpr static size_t NumDofHolders[2] = {3,3};
  constexpr static size_t DofsPerHolder[2] = {1,1};
};
} //end MeshField namespace
#endif
