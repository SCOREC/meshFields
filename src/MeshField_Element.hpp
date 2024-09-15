#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Shape.hpp>
#include <iostream>

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

  using ValArray = Kokkos::Array<typename FieldAccessor::BaseType, Shape::numNodes>;
  KOKKOS_INLINE_FUNCTION ValArray getValue(int ent, Kokkos::Array<Real, Shape::meshEntDim+1> localCoord) const {
    ValArray c;
    const auto shapeValues = shapeFn.getValues(localCoord);
    for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci)
      c[ci] = 0;
    for (int ni = 0; ni < shapeFn.numNodes; ++ni) {
      for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci) {
        //map the element indices to the underlying field storage
        //e.g., Element = Triangle and field storage is at mesh vertices
        //e.g., Element = Edge and field storage is at mesh vertices
        auto map = e2f(ni, ci, ent);
        c[ci] += field(map.node, map.component, map.entity) * shapeValues[ni];
      }
    }
    return c;
  }
};

// given an array of parametric coordinates 'localCoords', one per mesh element, evaluate the
// fields value within each element
template <typename Element>
Kokkos::View<Real*> evaluate(Element& fes, Kokkos::View<Real**> localCoords) {
  assert(localCoords.extent(0) == fes.numMeshEnts);
  assert(localCoords.extent(1) == fes.meshEntDim()+1);
  Kokkos::View<Real*> res("result", fes.numMeshEnts);
  Kokkos::parallel_for(fes.numMeshEnts,
    KOKKOS_LAMBDA(const int ent) {
      //FIXME - use rank 2 view of coords
//      Kokkos::Array<Real,Element::Shape::meshEntDim+1> lc{ //not coallesced 
//        localCoords[ent*3], 
//        localCoords[ent*3+1],
//        localCoords[ent*3+2]};
//      auto val = fes.getValue(ent, lc);
//      res(ent) = val[0]; //hardcoded to be a scalar field 
    }
  );
  return res;
}

}
#endif
