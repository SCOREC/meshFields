#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_ShapeField.hpp"
#include "KokkosController.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearTriangleToVertexField {
  constexpr static MeshField::Mesh_Topology Topology[1] = {MeshField::Triangle};
  KOKKOS_FUNCTION MeshField::Map operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx, MeshField::LO tri, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Triangle);
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
  MeshField::MeshInfo meshInfo;
  meshInfo.numVtx = 5;
  meshInfo.numTri = 3;
  auto field = MeshField::CreateLagrangeField<ExecutionSpace, 1>(meshInfo);

//  MeshField::Element elm{ MeshField::LinearTriangleShape(), LinearTriangleToVertexField() };
//
//  MeshField::FieldElement f(numElms, field0, elm);
//
//  Kokkos::View<MeshField::Real[3][3]> lc("localCoords");
//  Kokkos::deep_copy(lc, 0.5);
//  auto x = MeshField::evaluate(f, lc);
}


struct LinearEdgeToVertexField {
  constexpr static MeshField::Mesh_Topology Topology[1] = {MeshField::Edge};
  KOKKOS_FUNCTION MeshField::Map operator()(MeshField::LO edgeNodeIdx, MeshField::LO edgeCompIdx, MeshField::LO edge, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Edge);
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
  MeshField::MeshInfo meshInfo;
  meshInfo.numVtx = 5;
  meshInfo.numEdge = 7;
  auto field = MeshField::CreateLagrangeField<ExecutionSpace, 1>(meshInfo);

//  MeshField::Element elm{MeshField::LinearEdgeShape(), LinearEdgeToVertexField()};
//
//  MeshField::FieldElement f(numEdges, field0, elm);
//
//  Kokkos::View<MeshField::Real[7][2]> lc("localCoords");
//  Kokkos::deep_copy(lc, 0.5);
//  auto x = MeshField::evaluate(f, lc);
}

struct QuadraticTriangleToField {
  constexpr static MeshField::Mesh_Topology Topology[2] = {MeshField::Edge, MeshField::Triangle};
  LinearEdgeToVertexField edge2vtx;
  LinearTriangleToVertexField tri2vtx;

  KOKKOS_FUNCTION MeshField::Map operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx, MeshField::LO ent, MeshField::Mesh_Topology topo) const {
    if(topo == MeshField::Edge ) {
      return edge2vtx(triNodeIdx, triCompIdx, ent, MeshField::Edge);
    } else if (topo == MeshField::Triangle ) {
      return tri2vtx(triNodeIdx, triCompIdx, ent, MeshField::Triangle);
    } else {
      std::cerr << "ERROR: Unsupported topology: " << __func__ << " supports MESH_VERTEX and MESH_TRIANGLE.\n";
      assert(false);
      return {};
    }
  }
};

//evaluate a field at the specified local coordinate for each triangle using
//quadratic shape functions
void quadraticTriangleLocalPointEval() {
  MeshField::MeshInfo meshInfo;
  meshInfo.numVtx = 5;
  meshInfo.numEdge = 7;
  meshInfo.numTri = 3;
  auto field = MeshField::CreateLagrangeField<ExecutionSpace, 2>(meshInfo);

//  MeshField::Element elm{ MeshField::QuadraticTriangleShape(), LinearTriangleToVertexField() };

// FIXME - HERE
//  MeshField::FieldElement f(numTri,
//                             vtxField,
//                             edgeField,
//                             LinearTriangleToVertexField(),
//                             LinearTriangleToEdgeField());
//
//  Kokkos::View<MeshField::Real[3][3]> lc("localCoords");
//  Kokkos::deep_copy(lc, 0.5);
//  auto x = MeshField::evaluate(f, lc);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  triangleLocalPointEval();
  edgeLocalPointEval();
  //quadraticTriangleLocalPointEval();
  std::cerr << "done\n";
  Kokkos::finalize();
  return 0;
}


