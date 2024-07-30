#include <cassert>
#include <iostream>
#include <type_traits>
#include <Kokkos_Core.hpp>

struct CSR {
  Kokkos::View<int*> vals;
  Kokkos::View<int*> offsets;
};

template <typename Functor>
void cavityOp(Functor a, CSR& cavities) {
    const auto numEnts = cavities.offsets.size()-1;
    const auto vals = cavities.vals;
    const auto offsets = cavities.offsets;
    Kokkos::parallel_for(numEnts,
      KOKKOS_LAMBDA(const int ent) {
        for(int elm=offsets[ent]; elm<offsets[ent+1]-offsets[ent]; elm++) {
          Kokkos::printf("ent %d elm %d\n", ent, elm);
          auto val = a(ent,elm);
          Kokkos::printf("ent %d elm %d = %d\n", ent, elm, val);
        }
    });
}

struct AvgPosOp {
    Kokkos::View<int*> avgPos;
    AvgPosOp(size_t numVtx) : avgPos(Kokkos::View<int*>("avgPos", numVtx)) {}
    KOKKOS_INLINE_FUNCTION int operator()(int vtx, int elm) const {
      avgPos(vtx) += vtx+10;
      return avgPos(vtx);
    };
};

namespace MeshFields {
template <typename T>
struct FieldElement {
  //prototype as SOA
  const size_t numCompsPerDof;
  const size_t numNodesPerEnt;
  const size_t numMeshEnts;
  const size_t meshEntDim;
  Kokkos::View<T*> nodeData;
  FieldElement(size_t in_numCompsPerDof,
               size_t in_numNodesPerEnt,
               size_t in_numMeshEnts,
               size_t in_meshEntDim) :
    numCompsPerDof(in_numCompsPerDof),
    numNodesPerEnt(in_numNodesPerEnt),
    numMeshEnts(in_numMeshEnts),
    meshEntDim(in_meshEntDim),
    nodeData("nodeData", numCompsPerDof*numNodesPerEnt*numMeshEnts) {}
};
}

CSR getCavities() {
  //Hardcoded for three triangles numbered 0, 1, 2 from left to right.
  //    __
  //  /\  /\
  // /__\/__\
  //
  // The vertices are numbered clockwise from the top left: 0, ..., 4.
  // The cavities for each vertex are defined as the set of triangles they bound:
  // <vtx> : <list of triangle indices>  <degree>
  // 0: 0,1    2
  // 1: 1,2    2
  // 2: 2      1
  // 3: 0,1,2  3
  // 4: 0      1
  std::array<int,9> arr = {0,1,1,2,2,0,1,2,0};
  Kokkos::View<int[9], Kokkos::HostSpace, Kokkos::MemoryUnmanaged> vals_h(arr.data(), arr.size());
  Kokkos::View<int[9]> vals("cavities_vals");
  Kokkos::deep_copy(vals, vals_h);
  std::array<int,6> off_arr = {0,2,4,5,8,9};
  Kokkos::View<int[6], Kokkos::HostSpace, Kokkos::MemoryUnmanaged> off_h(off_arr.data(), off_arr.size());
  auto off = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), off_h);
  return CSR{vals,off};
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const auto numComps = 1;
    const auto numNodes = 1;
    const auto numEnts = 5;
    const auto dim = 2;
    MeshFields::FieldElement<double>(numComps,numNodes,numEnts,dim);
    auto cavities = getCavities();
    AvgPosOp f(numEnts);
    cavityOp(f,cavities); //this loop is too short - FIXME
  }
  Kokkos::finalize();
  return 0;
}


