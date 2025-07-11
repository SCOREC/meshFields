#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearTriangleToVertexField {
  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Triangle};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx,
             MeshField::LO tri, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Triangle);
    // Need to find which mesh vertex is described by the triangle and one of
    // its node indices. This could be implemented using element-to-dof holder
    // adjacencies, canonical ordering provided by the mesh database, which
    // would provide the index to the vertex in the dof holder array (assuming
    // the dof holder array is in the same order as vertex local numbering in
    // the mesh). For the simplicity of the test case, it is hard coded here
    // using local dof holder numbering:
    //      node
    // tri 0 1 2
    // 0   0 1 4
    // 1   4 1 2
    // 2   4 2 3
    MeshField::LO triNode2Vtx[3][3] = {{0, 1, 4}, {4, 1, 2}, {4, 2, 3}};
    const MeshField::LO vtx = triNode2Vtx[tri][triNodeIdx];
    return {0, 0, vtx, MeshField::Vertex};
  }
};

// evaluate a field at the specified local coordinate for each triangle
void triangleLocalPointEval() {
  const auto numElms = 3; // provided by the mesh
  const MeshField::MeshInfo meshInfo{.numVtx = 5, .numTri = 3};
  auto field = MeshField::CreateLagrangeField<
      ExecutionSpace, MeshField::KokkosController, MeshField::Real, 1, 2, 1>(
      meshInfo);

  MeshField::FieldElement f(numElms, field, MeshField::LinearTriangleShape(),
                            LinearTriangleToVertexField());

  Kokkos::View<MeshField::Real[3][3]> lc("localCoords");
  Kokkos::deep_copy(lc, 1.0 / 3);
  auto x = MeshField::evaluate(f, lc);
}

struct LinearEdgeToVertexField {
  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Edge};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO edgeNodeIdx, MeshField::LO edgeCompIdx,
             MeshField::LO edge, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Edge);
    // Need to find which mesh vertex is described by the edge and one of its
    // node indices.  This would be implemented using mesh database adjacencies,
    // etc. For the simplicity of the test case, it is hard coded here:
    //      node
    // edge 0 1
    // 0    0 1
    // 1    1 2
    // 2    2 3
    // 3    3 4
    // 4    4 0
    // 5    4 1
    // 6    4 2
    MeshField::LO edgeNode2Vtx[7][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4},
                                        {4, 0}, {4, 1}, {4, 2}};
    const MeshField::LO vtx = edgeNode2Vtx[edge][edgeNodeIdx];
    return {0, 0, vtx, MeshField::Vertex};
  }
};

// evaluate a field at the specified local coordinate for each edge
void edgeLocalPointEval() {
  const MeshField::MeshInfo meshInfo{.numVtx = 5, .numEdge = 7, .dim = 1};
  auto field = MeshField::CreateLagrangeField<
      ExecutionSpace, MeshField::KokkosController, MeshField::Real, 1, 1, 1>(
      meshInfo);

  MeshField::FieldElement f(meshInfo.numEdge, field,
                            MeshField::LinearEdgeShape(),
                            LinearEdgeToVertexField());

  Kokkos::View<MeshField::Real[7][2]> lc("localCoords");
  Kokkos::deep_copy(lc, 1.0 / 2);
  auto x = MeshField::evaluate(f, lc);
}

struct QuadraticTriangleToField {
  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Triangle};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx,
             MeshField::LO ent, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Triangle);
    assert(ent == 0);
    // clang-format off
    // hardcoded using the following numbering in MeshFields
    //         2
    //       /   \
    //      5     4
    //    /        \
    //   0 --  3 -- 1
    MeshField::LO triNode2DofHolder[6] = {/*vertices*/ 0, 1, 2,
                                          /*edges*/ 0, 1, 2};
    // clang-format on
    MeshField::Mesh_Topology triNode2DofHolderTopo[6] = {
        /*vertices*/ MeshField::Vertex,
        MeshField::Vertex,
        MeshField::Vertex,
        /*edges*/ MeshField::Edge,
        MeshField::Edge,
        MeshField::Edge};
    const auto dofHolder = triNode2DofHolder[triNodeIdx];
    const auto dofHolderTopo = triNode2DofHolderTopo[triNodeIdx];
    return {0, 0, dofHolder, dofHolderTopo};
  }
};

// evaluate a field at the specified local coordinate for one triangle using
// quadratic shape functions
void quadraticTriangleLocalPointEval() {
  const MeshField::MeshInfo meshInfo{
      .numVtx = 3, .numEdge = 3, .numTri = 1, .dim = 2};
  auto field = MeshField::CreateLagrangeField<
      ExecutionSpace, MeshField::KokkosController, MeshField::Real, 2, 2, 1>(
      meshInfo);

  MeshField::FieldElement f(meshInfo.numTri, field,
                            MeshField::QuadraticTriangleShape(),
                            QuadraticTriangleToField());

  Kokkos::View<MeshField::Real[1][3]> lc("localCoords");
  Kokkos::deep_copy(lc, 1.0 / 3);
  auto x = MeshField::evaluate(f, lc);
}

struct QuadraticTetrahedronToField {
  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Tetrahedron};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO tetNodeIdx, MeshField::LO tetCompIdx,
             MeshField::LO ent, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Tetrahedron);
    assert(ent == 0);
    // clang-format off
    MeshField::LO tetNode2DofHolder[10] = {
        /*vertices*/ 0, 1, 2, 3,
        /*edges*/ 0, 1, 2, 3, 4, 5};
    // clang-format on
    MeshField::Mesh_Topology tetNode2DofHolderTopo[10] = {
        /*vertices*/ MeshField::Vertex,
        MeshField::Vertex,
        MeshField::Vertex,
        MeshField::Vertex,
        /*edges*/ MeshField::Edge,
        MeshField::Edge,
        MeshField::Edge,
        MeshField::Edge,
        MeshField::Edge,
        MeshField::Edge};
    const auto dofHolder = tetNode2DofHolder[tetNodeIdx];
    const auto dofHolderTopo = tetNode2DofHolderTopo[tetNodeIdx];
    return {0, 0, dofHolder, dofHolderTopo};
  }
};

// evaluate a field at the specified local coordinate for one tet using
// linear shape functions
void quadraticTetrahedronLocalPointEval() {
  const int MeshDim = 3;
  const int ShapeOrder = 2;
  const MeshField::MeshInfo meshInfo{
      .numVtx = 4, .numEdge = 6, .numTri = 4, .numTet = 1, .dim = MeshDim};
  auto field =
      MeshField::CreateLagrangeField<ExecutionSpace,
                                     MeshField::KokkosController,
                                     MeshField::Real, ShapeOrder, MeshDim, 1>(
          meshInfo);

  MeshField::FieldElement f(meshInfo.numTet, field,
                            MeshField::QuadraticTetrahedronShape(),
                            QuadraticTetrahedronToField());

  Kokkos::View<MeshField::Real[1][4]> lc("localCoords");
  Kokkos::deep_copy(lc, 1.0 / 4);
  auto x = MeshField::evaluate(f, lc);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  triangleLocalPointEval();
  edgeLocalPointEval();
  quadraticTriangleLocalPointEval();
  quadraticTetrahedronLocalPointEval();
  Kokkos::finalize();
  return 0;
}
