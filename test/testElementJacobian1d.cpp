#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

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
    MeshField::LO edgeNode2Vtx[1][2] = {{0, 1}};
    const MeshField::LO vtx = edgeNode2Vtx[edge][edgeNodeIdx];
    return {0, 0, vtx, MeshField::Vertex};
  }
};

// evaluate a field at the specified local coordinate for each edge
void edgeJacobian() {
  MeshField::MeshInfo meshInfo;
  meshInfo.numVtx = 2;
  meshInfo.numEdge = 1;
  auto field = MeshField::CreateLagrangeField<
      ExecutionSpace, MeshField::KokkosController, MeshField::Real, 1, 1>(
      meshInfo);

  MeshField::FieldElement f(meshInfo.numEdge, field,
                            MeshField::LinearEdgeShape(),
                            LinearEdgeToVertexField());

  Kokkos::View<MeshField::Real*[2]> lc("localCoords",1);
  Kokkos::deep_copy(lc, 1.0 / 2);
  const auto numPtsPerElement = 1;
  auto x = MeshField::getJacobians(f, lc, numPtsPerElement);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  edgeJacobian();
  Kokkos::finalize();
  return 0;
}
