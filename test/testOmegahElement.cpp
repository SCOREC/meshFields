#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
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
void triangleLocalPointEval(Omega_h::Library& ohLib) {
  auto world = ohLib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  auto mesh = Omega_h::build_box(world, family, len, len, 0.0, 3, 3, 0);
  Omega_h::vtk::write_parallel("square.vtk", &mesh, 2);
  MeshField::MeshInfo meshInfo;
  meshInfo.numVtx = mesh.nverts();
  meshInfo.numTri = mesh.nfaces();
  auto field =
      MeshField::CreateLagrangeField<ExecutionSpace, MeshField::Real, 1, 2>(
          meshInfo);

  MeshField::Element elm{MeshField::LinearTriangleShape(),
                         LinearTriangleToVertexField()};

  MeshField::FieldElement f(meshInfo.numTri, field, elm);

  Kokkos::View<MeshField::Real[3][3]> lc("localCoords");
  Kokkos::deep_copy(lc, 0.5);
  auto x = MeshField::evaluate(f, lc);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  triangleLocalPointEval(lib);
  Kokkos::finalize();
  return 0;
}
