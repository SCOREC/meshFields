#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>

namespace MeshFields {

struct LinearEdgeShape {
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<double,2> getValues(Kokkos::Array<double, 2> const& xi) const {
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
  Kokkos::Array<double,3> getValues(Kokkos::Array<double, 3> const& xi) const {
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

// hardcoded as a linear triangular element 
template <typename T, typename Shape>
struct FieldElement {
  //prototype as SOA
  const size_t numMeshEnts;
  Kokkos::View<T*> nodeData; //replaced by a 'meshfield'
  Shape shapeFn;
  size_t meshEntDim() { 
    return shapeFn.meshEntDim;
  }
  FieldElement(size_t in_numMeshEnts) :
    numMeshEnts(in_numMeshEnts),
    nodeData("nodeData", shapeFn.numComponentsPerDof*shapeFn.numNodes*numMeshEnts) {}
  //need accessor here that handles indexing - fieldSlice provides this
  KOKKOS_INLINE_FUNCTION T& operator() (int comp, int node, int ent) const {
    //simple stub for prototype
    assert(ent < numMeshEnts);
    (void)comp;
    (void)node;
    return nodeData(ent);
  }
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
