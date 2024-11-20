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
#include <sstream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearTriangleToVertexField {
  Omega_h::LOs triVerts;
  LinearTriangleToVertexField(Omega_h::Mesh mesh)
      : triVerts(mesh.ask_elem_verts()) {
    if (mesh.dim() != 2 && mesh.family() != OMEGA_H_SIMPLEX) {
      MeshField::fail(
          "The mesh passed to %s must be 2D and simplex (triangles)\n",
          __func__);
    }
  }

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
    return {0, triCompIdx, vtx, MeshField::Vertex}; // node, comp, ent, topo
  }
};

struct LinearFunction {
  KOKKOS_INLINE_FUNCTION
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y) const {
    return 2.0 * x + y;
  }
};

struct QuadraticTriangleToField {
  Omega_h::LOs triVerts;
  Omega_h::LOs triEdges;
  QuadraticTriangleToField(Omega_h::Mesh mesh)
      : triVerts(mesh.ask_elem_verts()),
        triEdges(mesh.ask_down(mesh.dim(), 1).ab2b) {
    if (mesh.dim() != 2 && mesh.family() != OMEGA_H_SIMPLEX) {
      MeshField::fail(
          "The mesh passed to %s must be 2D and simplex (triangles)\n",
          __func__);
    }
  }

  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Triangle};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx,
             MeshField::LO tri, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Triangle);
    // Omega_h has no concept of nodes so we can define the map from
    // triNodeIdx to the dof holder index
    const MeshField::LO triNode2DofHolder[6] = {
        /*vertices*/ 0, 1, 2,
        /*edges*/ 0,    1, 2};
    const MeshField::Mesh_Topology triNode2DofHolderTopo[6] = {
        /*vertices*/
        MeshField::Vertex, MeshField::Vertex, MeshField::Vertex,
        /*edges*/
        MeshField::Edge, MeshField::Edge, MeshField::Edge};
    const auto dofHolderIdx = triNode2DofHolder[triNodeIdx];
    const auto dofHolderTopo = triNode2DofHolderTopo[triNodeIdx];
    // Given the topo index and type find the Omega_h vertex or edge index that
    // bounds the triangle
    Omega_h::LO osh_ent;
    if (dofHolderTopo == MeshField::Vertex) {
      const auto triDim = 2;
      const auto vtxDim = 0;
      const auto ignored = -1;
      const auto localVtxIdx =
          Omega_h::simplex_down_template(triDim, vtxDim, dofHolderIdx, ignored);
      const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
      osh_ent = triVerts[(tri * triToVtxDegree) + localVtxIdx];
    } else if (dofHolderTopo == MeshField::Edge) {
      const auto triDim = 2;
      const auto edgeDim = 1;
      const auto triToEdgeDegree = Omega_h::simplex_degree(triDim, edgeDim);
      // passing dofHolderIdx as Omega_h_simplex.hpp does not provide
      // a function that maps a triangle and edge index to a 'canonical' edge
      // index. This may need to be revisited...
      osh_ent = triEdges[(tri * triToEdgeDegree) + dofHolderIdx];
    } else {
      assert(false);
    }
    return {0, triCompIdx, osh_ent, dofHolderTopo};
  }
};

struct QuadraticFunction {
  KOKKOS_INLINE_FUNCTION
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y) const {
    return (x * x) + (2.0 * y);
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

template <int ShapeOrder> auto getTriangleElement(Omega_h::Mesh mesh) {
  static_assert(ShapeOrder == 1 || ShapeOrder == 2);
  if constexpr (ShapeOrder == 1) {
    return MeshField::Element{MeshField::LinearTriangleShape(),
                              LinearTriangleToVertexField(mesh)};
  } else if constexpr (ShapeOrder == 2) {
    return MeshField::Element{MeshField::QuadraticTriangleShape(),
                              QuadraticTriangleToField(mesh)};
  }
}

// evaluate a field at the specified local coordinate for each triangle
template <typename AnalyticFunction, int ShapeOrder>
bool triangleLocalPointEval(Omega_h::Mesh mesh,
                            Kokkos::View<MeshField::Real *[3]> localCoords,
                            AnalyticFunction func) {
  const auto MeshDim = 2;
  if (mesh.dim() != MeshDim) {
    MeshField::fail("input mesh must be 2d\n");
  }
  if (ShapeOrder != 1 && ShapeOrder != 2) {
    MeshField::fail("field order must be 1 or 2\n");
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
  if (ShapeOrder == 2) {
    const auto edgeDim = 1;
    const auto vtxDim = 0;
    const auto edge2vtx = mesh.ask_down(edgeDim, vtxDim).ab2b;
    auto setFieldAtEdges = KOKKOS_LAMBDA(const int &edge) {
      // get dofholder position at the midpoint of edge
      // - TODO should be encoded in the field?
      const auto left = edge2vtx[edge * 2];
      const auto right = edge2vtx[edge * 2 + 1];
      const auto x = (coords[left * MeshDim] + coords[right * MeshDim]) / 2.0;
      const auto y =
          (coords[left * MeshDim + 1] + coords[right * MeshDim + 1]) / 2.0;
      field(0, 0, edge, MeshField::Edge) = func(x, y);
    };
    field.meshField.parallel_for({0}, {meshInfo.numEdge}, setFieldAtEdges,
                                 "setFieldAtEdges");
  }

  auto coordField = MeshField::CreateCoordinateField<ExecutionSpace>(meshInfo);
  auto setCoordField = KOKKOS_LAMBDA(const int &i) {
    coordField(0, 0, i, MeshField::Vertex) = coords[i * MeshDim];
    coordField(0, 1, i, MeshField::Vertex) = coords[i * MeshDim + 1];
  };
  coordField.meshField.parallel_for({0}, {meshInfo.numVtx}, setCoordField,
                                    "setCoordField");

  const auto elm = getTriangleElement<ShapeOrder>(mesh);

  MeshField::FieldElement f(meshInfo.numTri, field, elm);
  auto eval = MeshField::evaluate(f, localCoords);

  MeshField::Element coordElm{MeshField::LinearTriangleCoordinateShape(),
                              LinearTriangleToVertexField(mesh)};
  MeshField::FieldElement fcoords(meshInfo.numTri, coordField, coordElm);
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

struct TestCoords {
  Kokkos::View<MeshField::Real *[3]> coords;
  std::string name;
};

Kokkos::View<MeshField::Real *[3]>
createElmAreaCoords(size_t numElements,
                    Kokkos::Array<MeshField::Real, 3> coord) {
  Kokkos::View<MeshField::Real *[3]> lc("localCoords", numElements);
  Kokkos::parallel_for(
      "setLocalCoords", numElements, KOKKOS_LAMBDA(const int &i) {
        lc(i, 0) = coord[0];
        lc(i, 1) = coord[1];
        lc(i, 2) = coord[2];
      });
  return lc;
}

void doFail(std::string_view order, std::string_view function,
            std::string_view location) {
  std::stringstream ss;
  ss << order << " field evaluation with " << function
     << " analytic function at " << location << " points failed\n";
  std::string msg = ss.str();
  MeshField::fail(msg);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  MeshField::Debug = true;
  {
    auto mesh = createMeshTri18(lib);
    auto centroids =
        createElmAreaCoords(mesh.nfaces(), {1 / 3.0, 1 / 3.0, 1 / 3.0});
    auto interior = createElmAreaCoords(mesh.nfaces(), {0.1, 0.4, 0.5});
    auto vertex = createElmAreaCoords(mesh.nfaces(), {0.0, 0.0, 1.0});
    // clang-format off
    const auto cases = {TestCoords{centroids, "centroids"},
                        TestCoords{interior,  "interior"},
                        TestCoords{vertex,    "vertex"}};
    // clang-format on
    for (auto testCase : cases) {
      auto failed = triangleLocalPointEval<LinearFunction, 1>(
          mesh, testCase.coords, LinearFunction{});
      if (failed)
        doFail("linear", "linear", testCase.name);
      failed = triangleLocalPointEval<QuadraticFunction, 2>(
          mesh, testCase.coords, QuadraticFunction{});
      if (failed)
        doFail("quadratic", "quadratic", testCase.name);
      failed = triangleLocalPointEval<LinearFunction, 2>(mesh, testCase.coords,
                                                         LinearFunction{});
      if (failed)
        doFail("quadratic", "linear", testCase.name);
    }
  }
  Kokkos::finalize();
  return 0;
}
