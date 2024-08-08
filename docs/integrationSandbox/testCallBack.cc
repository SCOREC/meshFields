#include <cassert>
#include <iostream>
#include <type_traits>
#include <Kokkos_Core.hpp>

struct CSR {
  Kokkos::View<int*> vals;
  Kokkos::View<int*> offsets;
};

using Real = double;
using LO = int;

namespace MeshFields {
// functions in this namespace are provided by the library

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

// User passes in an Element functor that has an paren operator function
// that takes two arguments:
// - the key entity index and
// - an index of a mesh element (highest dimension entity in the mesh) that
//   is part of the cavity associated with the key entity.
// and a Cavity functor that has a paren operator function
// that takes one argument; the key entity index.
//
// ##########
//
// Remark: I'm not sure we need to provide the hook for the 'Cavity' functor, the user
// could manage this outside this function independently....  It does reduce the
// need for another parallel_for loop... which shouldn't really matter
// performance wise.
//
template <typename ElementFunctor, typename CavityFunctor>
void applyToCavities(ElementFunctor&& ef, CavityFunctor&& cf, CSR& cavities) { //"universal reference"
    const auto numEnts = cavities.offsets.size()-1;
    const auto elmIdx = cavities.vals;
    const auto offsets = cavities.offsets;
    Kokkos::fence();
    Kokkos::parallel_for(numEnts,
      KOKKOS_LAMBDA(const int ent) {
        for(int elm=offsets(ent); elm<offsets(ent+1); elm++) {
          ef(ent,elmIdx(elm));
        }
        cf(ent);
    });
    Kokkos::fence();
}

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

} //end Meshfields namespace

CSR getCavities() {
  //Hardcoded for three triangles numbered 0, 1, 2 from left to right.
  //    __
  //  /\  /\
  // /__\/__\
  //
  // The centroid of each element is x=[0, 1, 2] respectively.
  // The vertices are numbered clockwise from the top left: 0, ..., 4.
  // The cavities for each vertex are defined as the set of triangles they bound:
  // <vtx> : <list of triangle indices>  <degree> <avgPosOfCentroid>
  // 0: 0,1    2   0.5
  // 1: 1,2    2   1.5
  // 2: 2      1   2.0
  // 3: 0,1,2  3   1.5
  // 4: 0      1   0.0
  std::array<int,9> arr = {0,1,1,2,2,0,1,2,0};
  Kokkos::View<int[9], Kokkos::HostSpace, Kokkos::MemoryUnmanaged> vals_h(arr.data(), arr.size());
  Kokkos::View<int[9]> vals("cavities_vals");
  Kokkos::deep_copy(vals, vals_h);
  std::array<int,6> off_arr = {0,2,4,5,8,9};
  Kokkos::View<int[6], Kokkos::HostSpace, Kokkos::MemoryUnmanaged> off_h(off_arr.data(), off_arr.size());
  auto off = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), off_h);
  return CSR{vals,off};
}

template <typename Element>
void setCentroids(Element& f) {
  Kokkos::parallel_for(f.numMeshEnts,
    KOKKOS_LAMBDA(const int ent) {
      f(0,0,ent) = static_cast<double>(ent);
    }
  );
}

template <typename Element>
struct AvgPosOp {
    Kokkos::View<int*> avgPos; //this would be a field at vertices
    Kokkos::View<int*> count; //this would be a field at vertices
    Element fes;
    AvgPosOp(size_t numVtx, Element& fieldElms) :
      avgPos(Kokkos::View<int*>("avgPos", numVtx)),
      count(Kokkos::View<int*>("count", numVtx)),
      fes(fieldElms) {} //copy
    KOKKOS_INLINE_FUNCTION int operator()(int vtx, int elm) const {
      avgPos(vtx) += fes(0,0,elm);
      Kokkos::atomic_increment(&count(vtx));
      return avgPos(vtx);
    };
    KOKKOS_INLINE_FUNCTION int operator()(int vtx) const {
      avgPos(vtx) /= count(vtx);
      return avgPos(vtx);
    };
};

//for each vertex, compute the average centroid of the triangles that are
//adjacent to it
void triangleCavityOp() {
  const auto numVerts = 5; //provided by the mesh
  const auto numElms = 3; //provided by the mesh
  MeshFields::FieldElement<double, MeshFields::LinearTriangleShape> f(numElms);
  setCentroids(f);
  auto cavities = getCavities();
  { //cavity operation - functor version
    AvgPosOp op(numVerts,f);
    MeshFields::applyToCavities(op,op,cavities);
  }
  { //cavity operation - lambda version
    Kokkos::View<int*> avgPos("avgPos", numVerts); //this would be a field at vertices
    Kokkos::View<int*> count("count", numVerts); //this would be a field at vertices
    auto sum = KOKKOS_LAMBDA (int vtx, int elm) {
      avgPos(vtx) += f(0,0,elm);
      Kokkos::atomic_increment(&count(vtx));
    };
    auto div = KOKKOS_LAMBDA (int vtx) {
      avgPos(vtx) /= count(vtx);
    };
    MeshFields::applyToCavities(sum,div,cavities);
  }
}

//evaluate a field at the specified local coordinate for each element
void triangleLocalPointEval() {
  const auto numElms = 3; //provided by the mesh
  MeshFields::FieldElement<double, MeshFields::LinearTriangleShape> f(numElms);

  std::array<Real,9> localCoords = {0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5};
  Kokkos::View<Real[9], Kokkos::HostSpace, Kokkos::MemoryUnmanaged> lc_h(localCoords.data(), localCoords.size());
  Kokkos::View<Real[9]> lc("localCoords");
  Kokkos::deep_copy(lc, lc_h);
  auto x = MeshFields::evaluate(f, lc);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  triangleCavityOp();
  triangleLocalPointEval();
  std::cerr << "done\n";
  Kokkos::finalize();
  return 0;
}


