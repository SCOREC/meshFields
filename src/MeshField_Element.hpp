#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Shape.hpp>
#include <iostream>

namespace MeshField {

struct ElementToDofHolderMap {
  LO node;
  LO component;
  LO entity;
  Mesh_Topology topo;
};

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

  //heavily based on SCOREC/core @ 7cd76473 apf/apfElement.cc
  using ValArray = Kokkos::Array<typename FieldAccessor::BaseType, ShapeType::numComponentsPerDof>;
  KOKKOS_INLINE_FUNCTION ValArray getValue(int ent, Kokkos::Array<Real, MeshEntDim+1> localCoord) const {
    ValArray c;
    const auto shapeValues = elm.shapeFn.getValues(localCoord);
    for (int ci = 0; ci < elm.shapeFn.numComponentsPerDof; ++ci)
      c[ci] = 0;
    for (auto topo : elm.elm2dof.Topology) { //element topology
      for (int ni = 0; ni < elm.shapeFn.numNodes; ++ni) {
        for (int ci = 0; ci < elm.shapeFn.numComponentsPerDof; ++ci) {
          //map the element indices to the underlying field storage
          //examples:
          // Element = Triangle and field storage is at mesh vertices (linear shape fn)
          // Element = Edge and field storage is at mesh vertices (linear shape fn)
          // Element = Triangle and field storage is at mesh vertices and edges (quadratic shape fn)
          auto map = elm.elm2dof(ni, ci, ent, topo);
          c[ci] += field(map.node, map.component, map.entity, map.topo) * shapeValues[ni];
        }
      }
    }
    return c;
  }
};

// given an array of parametric coordinates 'localCoords', one per mesh element, evaluate the
// fields value within each element
template <typename Element>
auto evaluate(Element& fes, Kokkos::View<Real**> localCoords) {
  //TODO add static asserts for values and functions provided by the templated types
  assert(localCoords.extent(0) == fes.numMeshEnts);
  assert(localCoords.extent(1) == fes.MeshEntDim+1);
  constexpr const auto numComponents = Element::ValArray::size();
  Kokkos::View<Real*[numComponents]> res("result", fes.numMeshEnts);
  Kokkos::parallel_for(fes.numMeshEnts,
    KOKKOS_LAMBDA(const int ent) {
      Kokkos::Array<Real,Element::MeshEntDim+1> lc;
      for(int i=0; i<localCoords.extent(1); i++) //better way?
        lc[i] = localCoords(ent,i);
      auto val = fes.getValue(ent, lc);
      for(int i=0; i<numComponents; i++)
        res(ent,i) = val[i];
    }
  );
  return res;
}

}
#endif
