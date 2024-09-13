#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "KokkosController.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>

struct LinearTriangleToVertexField {
  struct Map {
    MeshFields::LO node;
    MeshFields::LO component;
    MeshFields::LO entity;
  };

  KOKKOS_FUNCTION Map operator()(MeshFields::LO triNodeIdx, MeshFields::LO triCompIdx, MeshFields::LO tri) const {
    //Need to find which mesh vertex is described by the triangle and one of its 
    //node indices.  This would be implemented using mesh database adjacencies, etc.
    //For the simplicity of the test case, it is hard coded here:
    //     node 
    //tri 0 1 2
    //0   0 1 4
    //1   4 1 2
    //2   4 2 3
    MeshFields::LO triNode2Vtx[3][3] = {{0,1,4},{4,1,2},{4,2,3}};
    const MeshFields::LO vtx = triNode2Vtx[tri][triNodeIdx];
    return {0, 0, vtx};
  }
};

//evaluate a field at the specified local coordinate for each element
void triangleLocalPointEval() {
  const auto numElms = 3; //provided by the mesh
  const int numVerts = 5; //provided by the mesh
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using Ctrlr =
      Controller::KokkosController<MemorySpace, ExecutionSpace, double ***>;
  Ctrlr kk_ctrl({1, 1, numVerts}); //1 dof with 1 component per vtx
  MeshField::MeshField<Ctrlr> kokkosMeshField(kk_ctrl);

  auto field0 = kokkosMeshField.makeField<0>();

  MeshFields::FieldElement f(numElms,
                             MeshFields::LinearTriangleShape(), 
                             field0,
                             LinearTriangleToVertexField());

  std::array<MeshFields::Real,9> localCoords = {0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5};
  Kokkos::View<MeshFields::Real[9], Kokkos::HostSpace, Kokkos::MemoryUnmanaged> lc_h(localCoords.data(), localCoords.size());
  Kokkos::View<MeshFields::Real[9]> lc("localCoords");
  Kokkos::deep_copy(lc, lc_h);
  auto x = MeshFields::evaluate(f, lc);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  triangleLocalPointEval();
  std::cerr << "done\n";
  Kokkos::finalize();
  return 0;
}


