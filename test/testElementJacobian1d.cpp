#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_ShapeField.hpp"
#include "MeshField_For.hpp"
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

template <typename ShapeField>
void setEdgeCoords(size_t numVerts, Kokkos::View<MeshField::Real*> coords, ShapeField field) {
  assert(numVerts == coords.size());
  auto setFieldAtVertices = KOKKOS_LAMBDA(const int &vtx) {
    field(0, 0, vtx, MeshField::Vertex) = coords(vtx);
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {numVerts},
                          setFieldAtVertices, "setFieldAtVertices");
}

// evaluate a field at the specified local coordinate for each edge
void edgeJacobian() {
  MeshField::MeshInfo meshInfo;
  meshInfo.dim = 1;
  meshInfo.numVtx = 2;
  meshInfo.numEdge = 1;
  auto coordField = MeshField::CreateCoordinateField<ExecutionSpace, MeshField::KokkosController>(meshInfo);
  Kokkos::View<MeshField::Real*, Kokkos::HostSpace> coords_h("coords_h", 2);
  coords_h[0] = -1;
  coords_h[1] = 1;
  auto coords = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), coords_h);
  setEdgeCoords(meshInfo.numVtx, coords, coordField);

  MeshField::FieldElement f(meshInfo.numEdge, coordField,
                            MeshField::LinearEdgeShape(),
                            LinearEdgeToVertexField());

  Kokkos::View<MeshField::Real*[2]> lc("localCoords",1);
  Kokkos::deep_copy(lc, 1.0 / 2);
  const auto numPtsPerElement = 1;
  const auto J = MeshField::getJacobians(f, lc, numPtsPerElement);
  const auto J_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),J);
  assert(J_h.size() == 1);
  std::cout << "edge jacobian " << J_h(0,0,0) << std::endl;
  const auto expected = 1.0;
  assert(std::fabs(J_h(0,0,0) - expected) <= MeshField::MachinePrecision);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  edgeJacobian();
  Kokkos::finalize();
  return 0;
}
