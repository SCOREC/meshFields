#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Shape.hpp>
#include <iostream>

namespace MeshField {

template <typename Shape, typename ElementToDofHolderMap>
struct Element {
  //TODO add static asserts for variables and functions provided by the templated types
  Shape shapeFn;
  static const size_t MeshEntDim = Shape::meshEntDim; //better way?
  ElementToDofHolderMap elm2dof;
  Element(const Shape shapeFnIn, const ElementToDofHolderMap elm2dofIn) :
    shapeFn(shapeFnIn),
    elm2dof(elm2dofIn) {}
};

// hardcoded as a linear triangular element 
template <typename FieldAccessor, template<typename, typename> class ElementType, typename ShapeType, typename ElementToDofHolderMap>
struct FieldElement {
  //TODO add static asserts for functions provided by the templated types
  const size_t numMeshEnts;
  const FieldAccessor field;
  ElementType<ShapeType, ElementToDofHolderMap> elm;
  static const size_t MeshEntDim = ShapeType::meshEntDim;
  FieldElement(size_t in_numMeshEnts, const FieldAccessor& fieldIn, const ElementType<ShapeType, ElementToDofHolderMap> elmIn) :
    numMeshEnts(in_numMeshEnts),
    field(fieldIn),
    elm(elmIn) {}

  using ValArray = Kokkos::Array<typename FieldAccessor::BaseType, ShapeType::numNodes>;
  KOKKOS_INLINE_FUNCTION ValArray getValue(int ent, Kokkos::Array<Real, MeshEntDim+1> localCoord) const {
    ValArray c;
    const auto shapeValues = elm.shapeFn.getValues(localCoord);
    for (int ci = 0; ci < elm.shapeFn.numComponentsPerDof; ++ci)
      c[ci] = 0;
    //FIXME - loop over topology based on Element
    for (int ni = 0; ni < elm.shapeFn.numNodes; ++ni) {
      for (int ci = 0; ci < elm.shapeFn.numComponentsPerDof; ++ci) {
        //map the element indices to the underlying field storage
        //e.g., Element = Triangle and field storage is at mesh vertices
        //e.g., Element = Edge and field storage is at mesh vertices
        auto map = elm.elm2dof(ni, ci, ent); //fixme, add topo arg
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
  //TODO add static asserts for values and functions provided by the templated types
  assert(localCoords.extent(0) == fes.numMeshEnts);
  assert(localCoords.extent(1) == fes.MeshEntDim+1);
  Kokkos::View<Real*> res("result", fes.numMeshEnts);
  Kokkos::parallel_for(fes.numMeshEnts,
    KOKKOS_LAMBDA(const int ent) {
      Kokkos::Array<Real,Element::MeshEntDim+1> lc;
      for(int i=0; i<localCoords.extent(1); i++) //better way?
        lc[i] = localCoords(ent,i);
      auto val = fes.getValue(ent, lc);
      res(ent) = val[0]; //hardcoded to be a scalar field 
    }
  );
  return res;
}

}
#endif
