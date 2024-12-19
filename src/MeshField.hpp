#ifndef MESHFIELD_MESHFIELD_HPP
#define MESHFIELD_MESHFIELD_HPP

#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Fail.hpp"
#include "MeshField_For.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_file.hpp"    //move
#include "Omega_h_mesh.hpp"    //move
#include "Omega_h_simplex.hpp" //move

namespace {

MeshField::MeshInfo getMeshInfo(Omega_h::Mesh &mesh) {
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

template <typename ExecutionSpace, template <typename...> typename Controller =
                                       MeshField::KokkosController>
decltype(MeshField::CreateCoordinateField<ExecutionSpace, Controller>(
    MeshField::MeshInfo()))
createCoordinateField(MeshField::MeshInfo mesh_info, Omega_h::Reals coords) {
  const auto meshDim = mesh_info.dim;
  auto coordField =
      MeshField::CreateCoordinateField<ExecutionSpace, Controller>(mesh_info);
  auto setCoordField = KOKKOS_LAMBDA(const int &i) {
    coordField(0, 0, i, MeshField::Vertex) = coords[i * meshDim];
    coordField(0, 1, i, MeshField::Vertex) = coords[i * meshDim + 1];
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh_info.numVtx},
                          setCoordField, "setCoordField");
  return coordField;
}

struct LinearTriangleToVertexField {
  Omega_h::LOs triVerts;
  LinearTriangleToVertexField(Omega_h::Mesh &mesh)
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

struct QuadraticTriangleToField {
  Omega_h::LOs triVerts;
  Omega_h::LOs triEdges;
  QuadraticTriangleToField(Omega_h::Mesh &mesh)
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

template <int ShapeOrder> auto getTriangleElement(Omega_h::Mesh &mesh) {
  static_assert(ShapeOrder == 1 || ShapeOrder == 2);
  if constexpr (ShapeOrder == 1) {
    struct result {
      MeshField::LinearTriangleShape shp;
      LinearTriangleToVertexField map;
    };
    return result{MeshField::LinearTriangleShape(),
                  LinearTriangleToVertexField(mesh)};
  } else if constexpr (ShapeOrder == 2) {
    struct result {
      MeshField::QuadraticTriangleShape shp;
      QuadraticTriangleToField map;
    };
    return result{MeshField::QuadraticTriangleShape(),
                  QuadraticTriangleToField(mesh)};
  }
}

} // namespace

namespace MeshField {

template <typename ExecutionSpace, template <typename...> typename Controller =
                                       MeshField::KokkosController>
class OmegahMeshField {
private:
  Omega_h::Mesh &mesh;
  const MeshField::MeshInfo meshInfo;
  using CoordField = decltype(createCoordinateField<ExecutionSpace, Controller>(
      MeshField::MeshInfo(), Omega_h::Reals()));
  CoordField coordField;

public:
  OmegahMeshField(Omega_h::Mesh &mesh_in)
      : mesh(mesh_in), meshInfo(getMeshInfo(mesh)),
        coordField(createCoordinateField<ExecutionSpace, Controller>(
            getMeshInfo(mesh_in), mesh_in.coords())) {}

  template <typename DataType, size_t order, size_t dim>
  auto CreateLagrangeField() {
    return MeshField::CreateLagrangeField<ExecutionSpace, Controller, DataType,
                                          order, dim>(meshInfo);
  }

  template <typename Field> void writeVtk(Field &field) {
    using FieldDataType = typename decltype(field.vtxField)::BaseType;
    // HACK assumes there is a vertex field.. in the Field Mixin object
    auto field_view = field.vtxField.serialize();
    Omega_h::Write<FieldDataType> field_write(field_view);
    mesh.add_tag(0, "field", 1, Omega_h::read(field_write));
    Omega_h::vtk::write_parallel("foo.vtk", &mesh, mesh.dim());
  }

  template <typename ViewType = Kokkos::View<MeshField::LO *>>
  ViewType createOffsets(size_t numTri, size_t numPtsPerElem) {
    ViewType offsets("offsets", numTri + 1);
    Kokkos::parallel_for(
        "setOffsets", numTri,
        KOKKOS_LAMBDA(int i) { offsets(i) = i * numPtsPerElem; });
    Kokkos::deep_copy(Kokkos::subview(offsets, offsets.size() - 1),
                      numTri * numPtsPerElem);
    return offsets;
  }

  template <typename AnalyticFunction, typename ShapeField>
  void setEdges(Omega_h::Mesh &mesh, AnalyticFunction func, ShapeField field) {
    const auto MeshDim = mesh.dim();
    const auto edgeDim = 1;
    const auto vtxDim = 0;
    const auto edge2vtx = mesh.ask_down(edgeDim, vtxDim).ab2b;
    auto coords = mesh.coords();
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
    MeshField::parallel_for(ExecutionSpace(), {0}, {meshInfo.numEdge},
                            setFieldAtEdges, "setFieldAtEdges");
  }

  // evaluate a field at the specified local coordinate for each triangle
  template <typename AnalyticFunction, typename ViewType, typename ShapeField>
  auto triangleLocalPointEval(ViewType localCoords, size_t NumPtsPerElem,
                              AnalyticFunction func, ShapeField field) {
    const auto MeshDim = 2;
    if (mesh.dim() != MeshDim) {
      MeshField::fail("input mesh must be 2d\n");
    }
    const size_t ShapeOrder = 1; // typename ShapeField::order; //wrong
    if (ShapeOrder != 1 && ShapeOrder != 2) {
      MeshField::fail("input field order must be 1 or 2\n");
    }

    if (ShapeOrder == 2) {
      setEdges(mesh, func, field);
    }

    //    if (ShapeOrder == 1) {
    //      writeVtk(mesh, field);
    //    }

    const auto [shp, map] = getTriangleElement<ShapeOrder>(mesh);

    MeshField::FieldElement f(meshInfo.numTri, field, shp, map);
    auto offsets = createOffsets(meshInfo.numTri, NumPtsPerElem);
    auto eval = MeshField::evaluate(f, localCoords, offsets);
    return eval;
  }
};

} // namespace MeshField

#endif
