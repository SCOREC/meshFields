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

Omega_h::Mesh createMeshTri18(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  return Omega_h::build_box(world, family, len, len, 0.0, 3, 3, 0);
}

MeshField::MeshInfo getMeshInfo(Omega_h::Mesh mesh) {
  MeshField::MeshInfo meshInfo;
  meshInfo.numVtx = mesh.nverts();
  if (mesh.dim() > 1)
    meshInfo.numEdge = mesh.nedges();
  if (mesh.family() == OMEGA_H_SIMPLEX) {
    if (mesh.dim() > 1)
      meshInfo.numTri = mesh.nfaces();
    if (mesh.dim() == 3)
      meshInfo.numTet = mesh.nregions();
  } else { // hypercube
    if (mesh.dim() > 1)
      meshInfo.numQuad = mesh.nfaces();
    if (mesh.dim() == 3)
      meshInfo.numHex = mesh.nregions();
  }
  return meshInfo;
}

// evaluate a field at the specified local coordinate for each triangle
bool triangleLocalPointEval(Omega_h::Library &lib) {
  auto mesh = createMeshTri18(lib);
  const auto meshDim = mesh.dim();

  const auto meshInfo = getMeshInfo(mesh);
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
  const auto tol = 1e-9; // not sure what the upper bound is on compute errors
  MeshField::LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkResult", meshInfo.numTri,
      KOKKOS_LAMBDA(const int &i, MeshField::LO &lerrors) {
        const auto x = elmCentroids[i * meshDim];
        const auto y = elmCentroids[i * meshDim + 1];
        const auto expected = linearFunction(x, y);
        const auto computed = eval(i, 0);
        MeshField::LO isError = 0;
        if (Kokkos::fabs(computed - expected) > tol) {
          isError = 1;
          Kokkos::printf(
              "result for elm %d does not match: expected %f computed %f\n", i,
              expected, computed);
        }
        lerrors += isError;
      },
      numErrors);
  return (numErrors > 0);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  MeshField::Debug = true;
  auto failed = triangleLocalPointEval(lib);
  if (failed) {
    printf("triangleLocalPointEval(...) failed...\n");
    exit(EXIT_FAILURE);
  }
  Kokkos::finalize();
  return 0;
}
