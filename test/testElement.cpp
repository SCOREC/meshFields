#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "KokkosController.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>

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
    //node indices.
    //This could be implemented using element-to-dof holder
    //adjacencies, canonical ordering provided by the mesh database, which would
    //provide the index to the vertex in the dof holder array (assuming the dof
    //holder array is in the same order as vertex local numbering in the mesh).
    //For the simplicity of the test case, it is hard coded here using local
    //dof holder numbering:
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

  MeshField::Element elm{ MeshField::LinearTriangleShape(), LinearTriangleToVertexField() };

  MeshField::FieldElement f(numElms, field0, elm);

  Kokkos::View<MeshField::Real[3][3]> lc("localCoords");
  Kokkos::deep_copy(lc, 0.5);
  auto x = MeshField::evaluate(f, lc);
}


struct QuadraticTriangleToVertexField {
  struct Map {
    MeshField::LO node;
    MeshField::LO component;
    MeshField::LO entity;
  };

  KOKKOS_FUNCTION Map operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx, MeshField::LO tri) const {
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


//evaluate a field at the specified local coordinate for each triangle using
//quadratic shape functions
void quadraticTriangleLocalPointEval() {
  const auto numTri = 3;   //provided by the mesh
  const auto numEdges = 7; //provided by the mesh
  const int numVerts = 5;  //provided by the mesh
  using Ctrlr =
      Controller::KokkosController<MemorySpace, ExecutionSpace, double ***, double ***>;
  Ctrlr kk_ctrl({/*field 0*/ 1, 1, numVerts,   //1 dof with 1 component per vtx
                 /*field 1*/ 1, 1, numEdges}); //1 dof with 1 component per edge
  MeshField::MeshField<Ctrlr> kokkosMeshField(kk_ctrl);

  auto vtxField = kokkosMeshField.makeField<0>();
  auto edgeField = kokkosMeshField.makeField<1>();

// FIXME - HERE
//  MeshField::FieldElement f(numTri,
//                             MeshField::QuadraticTriangleShape(),
//                             vtxField,
//                             field0,
//                             LinearTriangleToVertexField(),
//                             LinearTriangleToEdgeField());
//
//  Kokkos::View<MeshField::Real[3][3]> lc("localCoords");
//  Kokkos::deep_copy(lc, 0.5);
//  auto x = MeshField::evaluate(f, lc);
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

  MeshField::Element elm{MeshField::LinearEdgeShape(), LinearEdgeToVertexField()};

  MeshField::FieldElement f(numEdges, field0, elm);

  Kokkos::View<MeshField::Real[7][2]> lc("localCoords");
  Kokkos::deep_copy(lc, 0.5);
  auto x = MeshField::evaluate(f, lc);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  triangleLocalPointEval();
  edgeLocalPointEval();
  quadraticTriangleLocalPointEval();
  std::cerr << "done\n";
  Kokkos::finalize();
  return 0;
}


