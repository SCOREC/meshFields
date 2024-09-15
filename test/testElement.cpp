#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "KokkosController.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace KE = Kokkos::Experimental;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearTriangleToVertexField {
  struct Map {
    MeshField::LO node;
    MeshField::LO component;
    MeshField::LO entity;
  };

  KOKKOS_FUNCTION Map operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx, MeshField::LO tri) const {
    //Need to find which mesh vertex is described by the triangle and one of its 
    //node indices.  This would be implemented using mesh database adjacencies, etc.
    //For the simplicity of the test case, it is hard coded here:
    //     node 
    //tri 0 1 2
    //0   0 1 4
    //1   4 1 2
    //2   4 2 3
    MeshField::LO triNode2Vtx[3][3] = {{0,1,4},{4,1,2},{4,2,3}};
    const MeshField::LO vtx = triNode2Vtx[tri][triNodeIdx];
    return {0, 0, vtx};
  }
};

//evaluate a field at the specified local coordinate for each triangle
void triangleLocalPointEval() {
  const auto numElms = 3; //provided by the mesh
  const int numVerts = 5; //provided by the mesh
  using Ctrlr =
      Controller::KokkosController<MemorySpace, ExecutionSpace, double ***>;
  Ctrlr kk_ctrl({1, 1, numVerts}); //1 dof with 1 component per vtx
  MeshField::MeshField<Ctrlr> kokkosMeshField(kk_ctrl);

  auto field0 = kokkosMeshField.makeField<0>();

  MeshField::FieldElement f(numElms,
                             MeshField::LinearTriangleShape(), 
                             field0,
                             LinearTriangleToVertexField());

  Kokkos::View<MeshField::Real[3][3]> lc("localCoords");
  Kokkos::deep_copy(lc, 0.5);
  auto x = MeshField::evaluate(f, lc);
}

struct LinearEdgeToVertexField {
  struct Map {
    MeshField::LO node;
    MeshField::LO component;
    MeshField::LO entity;
  };

  KOKKOS_FUNCTION Map operator()(MeshField::LO edgeNodeIdx, MeshField::LO edgeCompIdx, MeshField::LO edge) const {
    //Need to find which mesh vertex is described by the edge and one of its 
    //node indices.  This would be implemented using mesh database adjacencies, etc.
    //For the simplicity of the test case, it is hard coded here:
    //     node 
    //edge 0 1
    //0    0 1 
    //1    1 2 
    //2    2 3 
    //3    3 4 
    //4    4 0 
    //5    4 1 
    //6    4 2 
    MeshField::LO edgeNode2Vtx[7][2] = {{0,1},{1,2},{2,3},{3,4},{4,0},{4,1},{4,2}};
    const MeshField::LO vtx = edgeNode2Vtx[edge][edgeNodeIdx];
    return {0, 0, vtx};
  }
};

//evaluate a field at the specified local coordinate for each edge
void edgeLocalPointEval() {
  const auto numEdges = 7; //provided by the mesh
  const int numVerts = 5; //provided by the mesh
  using Ctrlr =
      Controller::KokkosController<MemorySpace, ExecutionSpace, double ***>;
  Ctrlr kk_ctrl({1,1,numVerts}); //1 dof with 1 component per vtx
  MeshField::MeshField<Ctrlr> kokkosMeshField(kk_ctrl);

  auto field0 = kokkosMeshField.makeField<0>();

  MeshField::FieldElement f(numEdges,
                             MeshField::LinearEdgeShape(), 
                             field0,
                             LinearEdgeToVertexField());

  Kokkos::View<MeshField::Real[7][2]> lc("localCoords");
  Kokkos::deep_copy(lc, 0.5);
  auto x = MeshField::evaluate(f, lc);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  triangleLocalPointEval();
  edgeLocalPointEval();
  std::cerr << "done\n";
  Kokkos::finalize();
  return 0;
}


