#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Shape.hpp>

namespace MeshFields {

// hardcoded as a linear triangular element 
template <typename Shape>
struct FieldElement {
  //prototype as SOA
  const size_t numMeshEnts;
  Kokkos::View<T*> nodeData; //TODO replace with a functor that provides the paren/index operator
  Shape shapeFn;
  size_t meshEntDim() { 
    return shapeFn.meshEntDim;
  }
  FieldElement(size_t in_numMeshEnts) :
    numMeshEnts(in_numMeshEnts),
    nodeData("nodeData", shapeFn.numComponentsPerDof*shapeFn.numNodes*numMeshEnts) {}
  //TODO replace with a functor that provides the paren/index operator
  KOKKOS_INLINE_FUNCTION T& operator() (int comp, int node, int ent) const {
    //simple stub for prototype
    assert(ent < numMeshEnts);
    (void)comp;
    (void)node;
    return nodeData(ent);
  }
  //FIXME - remove the hardcoded return type, generalize to Shape
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> getValue(int ent, Kokkos::Array<Real, 3> localCoord) const {
    Kokkos::Array<Real,3> c;
    const auto shapeValues = shapeFn.getValues(localCoord);
    for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci)
      c[ci] = 0;
    for (int ni = 0; ni < shapeFn.numNodes; ++ni)
      for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci)
        c[ci] += nodeData[ni * shapeFn.numComponentsPerDof + ci] * shapeValues[ni];
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
