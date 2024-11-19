#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Fail.hpp"
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

struct LinearFunction {
  KOKKOS_INLINE_FUNCTION
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y) const {
    return 2.0 * x + y;
  }
};

Omega_h::Mesh createMeshTri18(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  return Omega_h::build_box(world, family, len, len, 0.0, 3, 3, 0);
}

// TODO - move this into a specialized Omegah interface
MeshField::MeshInfo getMeshInfo(Omega_h::Mesh mesh) {
  MeshField::MeshInfo meshInfo;
  meshInfo.dim = mesh.dim();
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
template <typename AnalyticFunction, int ShapeOrder>
bool triangleLocalPointEval(Omega_h::Mesh mesh,
                            Kokkos::View<MeshField::Real *[3]> localCoords,
                            AnalyticFunction func) {
  const auto MeshDim = 2;
  if (mesh.dim() != MeshDim) {
    MeshField::fail("ERROR: input mesh must be 2d\n");
  }
  const auto meshInfo = getMeshInfo(mesh);
  auto field = MeshField::CreateLagrangeField<ExecutionSpace, MeshField::Real,
                                              ShapeOrder, 2>(meshInfo);

  // set field f based on analytic function
  auto coords = mesh.coords();
  auto setField = KOKKOS_LAMBDA(const int &i) {
    const auto x = coords[i * MeshDim];
    const auto y = coords[i * MeshDim + 1];
    field(0, 0, i, MeshField::Vertex) = func(x, y);
  };
  field.meshField.parallel_for({0}, {meshInfo.numVtx}, setField, "setField");

  auto coordField = MeshField::CreateCoordinateField<ExecutionSpace>(meshInfo);
  auto setCoordField = KOKKOS_LAMBDA(const int &i) {
    coordField(0, 0, i, MeshField::Vertex) = coords[i * MeshDim];
    coordField(0, 1, i, MeshField::Vertex) = coords[i * MeshDim + 1];
  };
  coordField.meshField.parallel_for({0}, {meshInfo.numVtx}, setCoordField,
                                    "setCoordField");

  MeshField::Element elm{MeshField::LinearTriangleShape(),
                         LinearTriangleToVertexField(mesh)};

  MeshField::FieldElement f(meshInfo.numTri, field, elm);
  auto eval = MeshField::evaluate(f, localCoords);

  MeshField::FieldElement fcoords(meshInfo.numTri, coordField, elm);
  auto globalCoords = MeshField::evaluate(fcoords, localCoords);

  // check the result
  MeshField::LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkResult", meshInfo.numTri,
      KOKKOS_LAMBDA(const int &i, MeshField::LO &lerrors) {
        const auto x = globalCoords(i, 0);
        const auto y = globalCoords(i, 1);
        const auto expected = func(x, y);
        const auto computed = eval(i, 0);
        MeshField::LO isError = 0;
        if (Kokkos::fabs(computed - expected) > MeshField::MachinePrecision) {
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
  {
    auto mesh = createMeshTri18(lib);
    Kokkos::View<MeshField::Real *[3]> lc("localCoords", mesh.nfaces());
    Kokkos::deep_copy(lc, 1 / 3.0); // the centroid of the triangle
    auto failed =
        triangleLocalPointEval<LinearFunction, 1>(mesh, lc, LinearFunction{});
    if (failed) {
      MeshField::fail("ERROR: triangleLocalPointEval(...)\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
