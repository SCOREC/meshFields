#ifndef MESHFIELD_SHAPE_H
#define MESHFIELD_SHAPE_H
#include <MeshField_Defines.hpp>
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
};
}
#endif
