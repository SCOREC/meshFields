#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_ShapeField.hpp"
#include "MeshField_For.hpp"
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
    // 0   0 1 2
    MeshField::LO triNode2Vtx[1][3] = {{0, 1, 2}};
    const MeshField::LO vtx = triNode2Vtx[tri][triNodeIdx];
    return {0, 0, vtx, MeshField::Vertex};
  }
};

template <typename ShapeField>
void setVtxCoords(size_t numVerts, size_t meshDim, ShapeField field) {
  Kokkos::View<MeshField::Real*, Kokkos::HostSpace> coords_h("coords_h", numVerts*meshDim);
  coords_h[0] = 0; coords_h[1] = 0;
  coords_h[2] = 1; coords_h[3] = 0;
  coords_h[4] = 0; coords_h[5] = 1;
  auto coords = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), coords_h);
  auto setCoordField = KOKKOS_LAMBDA(const int &vtx) {
    field(0, 0, vtx, MeshField::Vertex) = coords(vtx * meshDim);
    field(0, 1, vtx, MeshField::Vertex) = coords(vtx * meshDim + 1);
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {numVerts},
                          setCoordField, "setCoordField");
}

// evaluate a field at the specified local coordinate for each edge
void triJacobian() {
  MeshField::MeshInfo meshInfo;
  meshInfo.dim = 2;
  meshInfo.numVtx = 3;
  meshInfo.numEdge = 3;
  meshInfo.numTri = 1;
  auto coordField = MeshField::CreateCoordinateField<ExecutionSpace, MeshField::KokkosController>(meshInfo);
  setVtxCoords(meshInfo.numVtx, meshInfo.dim, coordField);

  MeshField::FieldElement f(meshInfo.numTri, coordField,
                            MeshField::LinearTriangleShape(),
                            LinearTriangleToVertexField());

  Kokkos::View<MeshField::Real*[2]> lc("localCoords",1);
  Kokkos::deep_copy(lc, 1.0 / 2);
  const auto numPtsPerElement = 1;
  const auto x = MeshField::getJacobians(f, lc, numPtsPerElement);
  const auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),x);
  assert(x_h.size() == 1);
  std::cout << "tri jacobian\n" 
            << x_h(0,0,0) << " " << x_h(0,0,1) << "\n"
            << x_h(0,1,0) << " " << x_h(0,1,1) << "\n";
  assert(std::fabs(x_h(0,0,0) - 1.0) <= MeshField::MachinePrecision);
  assert(std::fabs(x_h(0,0,1) - 0.0) <= MeshField::MachinePrecision);
  assert(std::fabs(x_h(0,1,0) - 0.0) <= MeshField::MachinePrecision);
  assert(std::fabs(x_h(0,1,1) - 1.0) <= MeshField::MachinePrecision);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  triJacobian();
  Kokkos::finalize();
  return 0;
}
