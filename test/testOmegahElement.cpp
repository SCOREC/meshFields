#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearTriangleToVertexField {
  Omega_h::LOs triVerts;
  LinearTriangleToVertexField(Omega_h::Mesh mesh)
      : triVerts(mesh.ask_elem_verts()) {}

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
    // the mesh).
    const auto triDim = 2;
    const auto vtxDim = 0;
    const auto ignored = -1;
    const auto localVtxIdx =
        Omega_h::simplex_down_template(triDim, vtxDim, triNodeIdx, ignored);
    const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
    const MeshField::LO vtx = triVerts[(tri * triToVtxDegree) + localVtxIdx];
    return {0, 0, vtx, MeshField::Vertex}; // node, comp, ent, topo
  }
};

KOKKOS_INLINE_FUNCTION
MeshField::Real linearFunction(MeshField::Real x, MeshField::Real y) {
  return 2.0 * x + y;
}

// evaluate a field at the specified local coordinate for each triangle
void triangleLocalPointEval(Omega_h::Library &ohLib) {
  auto world = ohLib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  auto mesh = Omega_h::build_box(world, family, len, len, 0.0, 3, 3, 0);
  const auto meshDim = 2;
  Omega_h::vtk::write_parallel("square.vtk", &mesh, meshDim);
  MeshField::MeshInfo meshInfo;
  meshInfo.numVtx = mesh.nverts();
  meshInfo.numTri = mesh.nfaces();
  auto field =
      MeshField::CreateLagrangeField<ExecutionSpace, MeshField::Real, 1, 2>(
          meshInfo);

  // set field f based on analytic function
  auto coords = mesh.coords();
  auto setField = KOKKOS_LAMBDA(const int &i) {
    const auto x = coords[i * meshDim];
    const auto y = coords[i * meshDim + 1];
    field(0, 0, i, MeshField::Vertex) = linearFunction(x, y);
  };
  field.meshField.parallel_for({0}, {meshInfo.numVtx}, setField, "setField");

  MeshField::Element elm{MeshField::LinearTriangleShape(),
                         LinearTriangleToVertexField(mesh)};

  MeshField::FieldElement f(meshInfo.numTri, field, elm);

  Kokkos::View<MeshField::Real *[3]> lc("localCoords", meshInfo.numTri);
  Kokkos::deep_copy(lc, 1 / 3.0); // the centroid of the triangle
  auto eval = MeshField::evaluate(f, lc);

  // check the result
  auto elmCentroids = Omega_h::average_field(
      &mesh, meshDim, Omega_h::LOs(mesh.nents(meshDim), 0, 1), meshDim, coords);
  const auto tol = 1e-6;
  auto checkResult = KOKKOS_LAMBDA(const int &i) {
    const auto x = elmCentroids[i * meshDim];
    const auto y = elmCentroids[i * meshDim + 1];
    const auto expected = linearFunction(x, y);
    const auto computed = eval(i, 0);
    if (Kokkos::fabs(computed - expected) > tol) {
      Kokkos::printf(
          "result for elm %d does not match: expected %f computed %f\n", i,
          expected, computed);
    }
  };
  field.meshField.parallel_for({0}, {meshInfo.numTri}, checkResult,
                               "checkResult");
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  triangleLocalPointEval(lib);
  Kokkos::finalize();
  return 0;
}
