#include <cassert>
#include <iostream>
#include <type_traits>
#include <Kokkos_Core.hpp>

struct CSR {
  Kokkos::View<int*> vals;
  Kokkos::View<int*> offsets;
};


namespace MeshFields {
template <typename T>
struct FieldElement {
  //prototype as SOA
  const size_t numCompsPerDof;
  const size_t numNodesPerEnt;
  const size_t numMeshEnts;
  const size_t meshEntDim;
  Kokkos::View<T*> nodeData; //replaced by a 'meshfield' 
  FieldElement(size_t in_numCompsPerDof,
               size_t in_numNodesPerEnt,
               size_t in_numMeshEnts,
               size_t in_meshEntDim) :
    numCompsPerDof(in_numCompsPerDof),
    numNodesPerEnt(in_numNodesPerEnt),
    numMeshEnts(in_numMeshEnts),
    meshEntDim(in_meshEntDim),
    nodeData("nodeData", numCompsPerDof*numNodesPerEnt*numMeshEnts) {}
  //need accessor here that handles indexing - fieldSlice provides this
  KOKKOS_INLINE_FUNCTION T& operator() (int comp, int node, int ent) const {
    //assert(ent < numMeshEnts);
    //simple stub for prototype
    //assert(numCompsPerDof==1);
    //assert(numNodesPerEnt==1);
    //(void)comp;
    //(void)node;
    Kokkos::printf("ent a %d\n", ent);
    nodeData(ent) = 1.0;
    Kokkos::printf("ent b %d\n", ent);
    return nodeData(ent);
  }
};
}

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

template <typename T>
void setCentroids(MeshFields::FieldElement<T>& f) {
  Kokkos::parallel_for(f.numMeshEnts,
    KOKKOS_LAMBDA(const int ent) {
      f(0,0,ent) = static_cast<double>(ent);
    }
  );
}

template <typename T>
struct AvgPosOp {
    Kokkos::View<int*> avgPos; //this would be a field at vertices
    MeshFields::FieldElement<T> fes;
    AvgPosOp(size_t numVtx, MeshFields::FieldElement<T>& fieldElms) : 
      avgPos(Kokkos::View<int*>("avgPos", numVtx)), 
      fes(fieldElms) {} //copy 
    KOKKOS_INLINE_FUNCTION int operator()(int vtx, int elm) const {
      auto x = fes(0,0,elm);  //this causes a cuda sync error
      Kokkos::printf("avgPosOp %d %d\n", vtx, elm);
      avgPos(vtx) = x; 
      return avgPos(vtx);
    };
};

// provided by meshfields
// User passes in a functor that has an paren operator function
// that takes two arguments:
// - the key entity index and
// - an index of a mesh element (highest dimension entity in the mesh) that
//   is part of the cavity associated with the key entity.
template <typename Functor>
void applyToCavities(Functor&& a, CSR& cavities) { //"universal reference"
    const auto numEnts = cavities.offsets.size()-1;
    const auto elmIdx = cavities.vals;
    const auto offsets = cavities.offsets;
    Kokkos::fence();
    Kokkos::printf("%d\n", cavities.offsets.size());
    Kokkos::parallel_for(numEnts,
      KOKKOS_LAMBDA(const int ent) {
        for(int elm=offsets(ent); elm<offsets(ent+1); elm++) {
          auto v = a(ent,elmIdx(elm));
          Kokkos::printf("ent %d elm %d = %f\n", ent, elmIdx(elm), v);
        }
    });
    Kokkos::fence();
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const auto numVerts = 5;
    const auto numComps = 1;
    const auto numNodes = 1;
    const auto numElms = 3;
    const auto dim = 2;
    MeshFields::FieldElement<double> f(numComps,numNodes,numElms,dim);
    setCentroids(f);
    auto cavities = getCavities();
    //AvgPosOp op(numVerts,f);
    Kokkos::View<int*> avgPos("avgPos", numVerts); //this would be a field at vertices
    auto op = KOKKOS_LAMBDA (int vtx, int elm) {
      auto x = f(0,0,elm);  //this causes a cuda sync error
      Kokkos::printf("avgPosOp %d %d\n", vtx, elm);
      avgPos(vtx) = x; 
      return avgPos(vtx);
    };
    applyToCavities(op,cavities);
    std::cerr << "done\n";
  }
  Kokkos::finalize();
  return 0;
}


