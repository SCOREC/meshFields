#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Shape.hpp>

namespace MeshField {

// hardcoded as a linear triangular element 
template <typename Shape, typename FieldAccessor, typename ElementToFieldMap>
struct FieldElement {
  //TODO add static asserts for functions provided by the templated types
  const size_t numMeshEnts;
  const FieldAccessor field;
  Shape shapeFn;
  ElementToFieldMap e2f;
  size_t meshEntDim() { 
    return shapeFn.meshEntDim;
  }
  FieldElement(size_t in_numMeshEnts, const Shape shapeFnIn, 
      const FieldAccessor& fieldIn, const ElementToFieldMap e2fIn) :
    numMeshEnts(in_numMeshEnts),
    shapeFn(shapeFnIn),
    field(fieldIn),
    e2f(e2fIn) {}
  //FIXME - remove the hardcoded return type, generalize to Shape
  //get value type from FieldAccessor (instead of hardcoding Real)
  //get num components from FieldAccessor or Shape (instead of hardcoding 3)
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> getValue(int ent, Kokkos::Array<Real, 3> localCoord) const {
    Kokkos::Array<Real,3> c;
    const auto shapeValues = shapeFn.getValues(localCoord);
    for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci)
      c[ci] = 0;
    for (int ni = 0; ni < shapeFn.numNodes; ++ni) {
      for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci) {
        auto map = e2f(ni, ci, ent); //map the triangle indices to the vertex based field
        c[ci] += field(map.node, map.component, map.entity) * shapeValues[ni];
      }
    }
    return c;
  }
};

// given an array of parametric coordinates 'localCoords', one per mesh element, evaluate the
// fields value within each element
template <typename Element>
Kokkos::View<Real*> evaluate(Element& fes, Kokkos::View<Real*> localCoords) {
  assert(localCoords.size() == fes.numMeshEnts*(fes.meshEntDim()+1));
  Kokkos::View<Real*> res("result", fes.numMeshEnts);
  Kokkos::parallel_for(fes.numMeshEnts,
    KOKKOS_LAMBDA(const int ent) {
      Kokkos::Array<Real,3> lc{ //not coallesced 
        localCoords[ent*3], 
        localCoords[ent*3+1],
        localCoords[ent*3+2]};
      auto val = fes.getValue(ent, lc);
      res(ent) = val[0]; //hardcoded to be a scalar field 
    }
  );
  return res;
}

}
#endif
