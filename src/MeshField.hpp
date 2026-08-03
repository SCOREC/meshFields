#ifndef MESHFIELD_MESHFIELD_HPP
#define MESHFIELD_MESHFIELD_HPP

#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Fail.hpp"
#include "MeshField_For.hpp"
#include "MeshField_ShapeField.hpp"
#include "MeshField_ReducedQuintic.hpp"
#include "Omega_h_file.hpp"    //move
#include "Omega_h_mesh.hpp"    //move
#include "Omega_h_simplex.hpp" //move
#include <vector>

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

template <typename ExecutionSpace, size_t dim,
          template <typename...>
          typename Controller = MeshField::KokkosController>
decltype(MeshField::CreateCoordinateField<ExecutionSpace, Controller, dim>(
    MeshField::MeshInfo()))
createCoordinateField(const MeshField::MeshInfo &mesh_info,
                      Omega_h::Reals coords) {
  const auto meshDim = mesh_info.dim;
  auto coordFieldWithCtrlr =
      MeshField::CreateCoordinateField<ExecutionSpace, Controller, dim>(
          mesh_info);
  auto coordField = coordFieldWithCtrlr.field;
  auto setCoordField = KOKKOS_LAMBDA(const int &i) {
    coordField(i, 0, 0, MeshField::Vertex) = coords[i * meshDim];
    coordField(i, 0, 1, MeshField::Vertex) = coords[i * meshDim + 1];
    if constexpr (dim == 3) {
      coordField(i, 0, 2, MeshField::Vertex) = coords[i * meshDim + 2];
    }
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh_info.numVtx},
                          setCoordField, "setCoordField");
  return coordFieldWithCtrlr;
}

} // anonymous namespace

namespace MeshField {

namespace Omegah {
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

  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
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
        (Omega_h::simplex_down_template(triDim, vtxDim, triNodeIdx, ignored) +
         2) %
        3;
    const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
    const MeshField::LO vtx = triVerts[(tri * triToVtxDegree) + localVtxIdx];
    return {0, triCompIdx, vtx, MeshField::Vertex}; // node, comp, ent, topo
  }
};
struct LinearTetrahedronToVertexField {
  Omega_h::LOs tetVerts;
  LinearTetrahedronToVertexField(Omega_h::Mesh &mesh)
      : tetVerts(mesh.ask_elem_verts()) {
    if (mesh.dim() != 3 && mesh.family() != OMEGA_H_SIMPLEX) {
      MeshField::fail(
          "The mesh passed to %s must be 3D and simplex (tetrahedron)\n",
          __func__);
    }
  }
  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Tetrahedron};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO tetNodeIdx, MeshField::LO tetCompIdx,
             MeshField::LO tet, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Tetrahedron);
    const auto tetDim = 3;
    const auto vtxDim = 0;
    const auto ignored = -1;
    const auto localVtxIdx =
        (Omega_h::simplex_down_template(tetDim, vtxDim, tetNodeIdx, ignored) +
         3) %
        4;
    const auto tetToVtxDegree = Omega_h::simplex_degree(tetDim, vtxDim);
    const MeshField::LO vtx = tetVerts[(tet * tetToVtxDegree) + localVtxIdx];
    return {0, tetCompIdx, vtx, MeshField::Vertex}; // node, comp, ent, topo
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

  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
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
      const auto localVtxIdx = (Omega_h::simplex_down_template(
                                    triDim, vtxDim, dofHolderIdx, ignored) +
                                2) %
                               3;
      const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
      osh_ent = triVerts[(tri * triToVtxDegree) + localVtxIdx];
    } else if (dofHolderTopo == MeshField::Edge) {
      const auto triDim = 2;
      const auto edgeDim = 1;
      const auto triToEdgeDegree = Omega_h::simplex_degree(triDim, edgeDim);
      // passing dofHolderIdx as Omega_h_simplex.hpp does not provide
      // a function that maps a triangle and edge index to a 'canonical' edge
      // index. This may need to be revisited...
      osh_ent = triEdges[(tri * triToEdgeDegree) + (dofHolderIdx + 2) % 3];
    } else {
      assert(false);
    }
    return {0, triCompIdx, osh_ent, dofHolderTopo};
  }
};

struct QuadraticTetrahedronToField {
  Omega_h::LOs tetVerts;
  Omega_h::LOs tetEdges;
  QuadraticTetrahedronToField(Omega_h::Mesh &mesh)
      : tetVerts(mesh.ask_elem_verts()),
        tetEdges(mesh.ask_down(mesh.dim(), 1).ab2b) {
    if (mesh.dim() != 3 && mesh.family() != OMEGA_H_SIMPLEX) {
      MeshField::fail(
          "The mesh passed to %s must be 3D and simplex (tetrahedron)",
          __func__);
    }
  }

  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Tetrahedron};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO tetNodeIdx, MeshField::LO tetCompIdx,
             MeshField::LO tet, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Tetrahedron);
    const MeshField::LO tetNode2DofHolder[10] = {0, 1, 2, 3, 3, 4, 5, 0, 1, 2};
    const MeshField::Mesh_Topology tetNode2DofHolderTopo[10] = {
        MeshField::Vertex, MeshField::Vertex, MeshField::Vertex,
        MeshField::Vertex, MeshField::Edge,   MeshField::Edge,
        MeshField::Edge,   MeshField::Edge,   MeshField::Edge,
        MeshField::Edge};
    const auto dofHolderIdx = tetNode2DofHolder[tetNodeIdx];
    const auto dofHolderTopo = tetNode2DofHolderTopo[tetNodeIdx];
    Omega_h::LO osh_ent;
    if (dofHolderTopo == MeshField::Vertex) {
      const auto tetDim = 3;
      const auto vtxDim = 0;
      const auto ignored = -1;
      // cyclic rotation of the omegah vertex order to map to the meshfields order
      // defined by the shape functions in MeshField_Shape.hpp
      const auto localVtxIdx = (Omega_h::simplex_down_template(
                                    tetDim, vtxDim, dofHolderIdx, ignored) +
                                3) %
                               4;
      const auto tetToVtxDegree = Omega_h::simplex_degree(tetDim, vtxDim);
      osh_ent = tetVerts[(tet * tetToVtxDegree) + localVtxIdx];
    } else if (dofHolderTopo == MeshField::Edge) {
      const auto tetDim = 3;
      const auto edgeDim = 1;
      const auto tetToEdgeDegree = Omega_h::simplex_degree(tetDim, edgeDim);
      osh_ent = tetEdges[(tet * tetToEdgeDegree) + dofHolderIdx];
    } else {
      assert(false);
    }
    return {0, tetCompIdx, osh_ent, dofHolderTopo};
  }
};

struct ReducedQuinticTriangleToField {
  Omega_h::LOs triVerts;

  ReducedQuinticTriangleToField(Omega_h::Mesh& mesh)
      : triVerts(mesh.ask_elem_verts()) {

    if (mesh.dim() != 2 || mesh.family() != OMEGA_H_SIMPLEX) {
      MeshField::fail(
          "The mesh passed to %s must be 2D and simplex (triangles)\n",
          __func__);
    }
  }

  static constexpr KOKKOS_FUNCTION
  Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Triangle};
  }

  KOKKOS_FUNCTION
  MeshField::ElementToDofHolderMap
  operator()(MeshField::LO triNodeIdx,
             MeshField::LO triCompIdx,
             MeshField::LO tri,
             MeshField::Mesh_Topology topo) const {

    assert(topo == MeshField::Triangle);

    // shape function index -> vertex
    const MeshField::LO localVtxIdx = triNodeIdx / 6;

    // dof component within vertex
    const MeshField::LO vertexDof = triNodeIdx % 6;

    const auto triDim = 2;
    const auto vtxDim = 0;
    const auto ignored = -1;

    const auto canonicalVtxIdx =
        (Omega_h::simplex_down_template(
             triDim, vtxDim, localVtxIdx, ignored) +
         2) %
        3;

    const auto triToVtxDegree =
        Omega_h::simplex_degree(triDim, vtxDim);

    const MeshField::LO vtx =
        triVerts[(tri * triToVtxDegree) + canonicalVtxIdx];

    return {
        0,                // node within vertex dof holder
        vertexDof,        // field component (the DOF type)
        vtx,
        MeshField::Vertex
    };
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

// ReducedQuintic triangle element with precomputed coefficients
// Functor for extracting triangle vertex coordinates on device
struct ExtractTriCoords {
  Omega_h::LOs triVerts_d;
  Omega_h::Reals coords_d;
  Kokkos::View<Omega_h::Matrix<2,3>*> triCoords_d; // one Matrix per triangle

  ExtractTriCoords(Omega_h::LOs triVerts_d_, Omega_h::Reals coords_d_,
                   Kokkos::View<Omega_h::Matrix<2,3>*> triCoords_d_)
    : triVerts_d(triVerts_d_), coords_d(coords_d_), triCoords_d(triCoords_d_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int tri) const {
    Omega_h::Few<Omega_h::LO, 3> vtxIds;
    for (int vi = 0; vi < 3; vi++) {
      const auto localVtxIdx = (vi + 2) % 3;
      vtxIds[vi] = triVerts_d[tri * 3 + localVtxIdx];
    }
    triCoords_d(tri) = Omega_h::gather_vectors<3, 2>(coords_d, vtxIds);
  }
};

// Result type for getReducedQuinticTriangleElement
struct ReducedQuinticTriangleElementResult {
  MeshField::ReducedQuinticTriangleShape shp;
  Omegah::ReducedQuinticTriangleToField map;
  Kokkos::View<MeshField::Real**> coeffs;
};

inline ReducedQuinticTriangleElementResult
getReducedQuinticTriangleElement(Omega_h::Mesh &mesh) {
  if (mesh.dim() != 2 || mesh.family() != OMEGA_H_SIMPLEX) {
    MeshField::fail("getReducedQuinticTriangleElement requires 2D simplex mesh\n");
  }
  
  const auto numTri = mesh.nfaces();
  const auto coords_d = mesh.coords();
  const auto triVerts_d = mesh.ask_elem_verts();
  
  // Allocate device view: one Matrix<2,3> per triangle
  // Matrix<dim, neev> = neev columns (vertices) of dim-dimensional vectors
  Kokkos::View<Omega_h::Matrix<2,3>*> triCoords_d("triCoords_device", numTri);
  
  // Extract triangle vertex coordinates using Omega_h gather_vectors
  Kokkos::parallel_for(
      "extractTriCoords", numTri,
      ExtractTriCoords(triVerts_d, coords_d, triCoords_d));
  
  // Precompute coefficients for all triangles entirely on device
  auto elemCoeffs = MeshField::precomputeReducedQuinticCoefficients(numTri, triCoords_d);

  return ReducedQuinticTriangleElementResult{
    MeshField::ReducedQuinticTriangleShape(),
    Omegah::ReducedQuinticTriangleToField(mesh),
    elemCoeffs};
}
template <int ShapeOrder> auto getTetrahedronElement(Omega_h::Mesh &mesh) {
  static_assert(ShapeOrder == 1 || ShapeOrder == 2);
  if constexpr (ShapeOrder == 1) {
    struct result {
      MeshField::LinearTetrahedronShape shp;
      LinearTetrahedronToVertexField map;
    };
    return result{MeshField::LinearTetrahedronShape(),
                  LinearTetrahedronToVertexField(mesh)};
  } else if constexpr (ShapeOrder == 2) {
    struct result {
      MeshField::QuadraticTetrahedronShape shp;
      QuadraticTetrahedronToField map;
    };
    return result{MeshField::QuadraticTetrahedronShape(),
                  QuadraticTetrahedronToField(mesh)};
  }
}

} // namespace Omegah

template <typename ExecutionSpace, size_t dim,
          template <typename...> typename Controller =
              MeshField::KokkosController>
class OmegahMeshField {
private:
  Omega_h::Mesh &mesh;
  const MeshField::MeshInfo meshInfo;
  using CoordField =
      decltype(createCoordinateField<ExecutionSpace, dim, Controller>(
          MeshField::MeshInfo(), Omega_h::Reals()));
  CoordField coordField;

public:
  OmegahMeshField(Omega_h::Mesh &mesh_in)
      : mesh(mesh_in), meshInfo(getMeshInfo(mesh)),
        coordField(createCoordinateField<ExecutionSpace, dim, Controller>(
            getMeshInfo(mesh_in), mesh_in.coords())) {
    static_assert(dim == 1 || dim == 2 || dim == 3);
  }

  template <typename DataType, size_t order, size_t numComp>
  // Ordering of field indexing changed to 'entity, node, component'
  auto CreateLagrangeField() const {
    return MeshField::CreateLagrangeField<ExecutionSpace, Controller, DataType,
                                          order, dim, numComp>(meshInfo);
  }

  auto getCoordField() { return coordField; }

  // FIXME support 2d and 3d and fields with order>1
  template <typename Field> void writeVtk(Field &field) const {
    using FieldDataType = typename decltype(field.vtxField)::BaseType;
    // HACK assumes there is a vertex field.. in the Field Mixin object
    auto field_view = field.vtxField.serialize();
    Omega_h::Write<FieldDataType> field_write(field_view);
    mesh.add_tag(0, "field", 1, Omega_h::read(field_write), false,
                 Omega_h::ArrayType::VectorND);
    Omega_h::vtk::write_parallel("foo.vtk", &mesh, mesh.dim());
  }

  template <typename ViewType = Kokkos::View<MeshField::LO *>>
  ViewType createOffsets(size_t numTri, size_t numPtsPerElem) const {
    ViewType offsets("offsets", numTri + 1);
    Kokkos::parallel_for(
        "setOffsets", numTri,
        KOKKOS_LAMBDA(int i) { offsets(i) = i * numPtsPerElem; });
    Kokkos::deep_copy(Kokkos::subview(offsets, offsets.size() - 1),
                      numTri * numPtsPerElem);
    return offsets;
  }

  // evaluate a field at the specified local coordinate for each triangle
  template <typename ViewType, typename ShapeField>
  auto triangleLocalPointEval(const ViewType &localCoords, size_t NumPtsPerElem,
                              const ShapeField &field) const {
    auto offsets = createOffsets(meshInfo.numTri, NumPtsPerElem);
    auto eval = triangleLocalPointEval<ViewType, ShapeField>(localCoords,
                                                             offsets, field);
    return eval;
  }

  // evaluate a field at the specified local coordinates for each triangle
  template <typename ViewType, typename ShapeField>
  auto triangleLocalPointEval(const ViewType &localCoords,
                              Kokkos::View<LO *> offsets,
                              const ShapeField &field) const {
    const auto MeshDim = 2;
    if (mesh.dim() != MeshDim) {
      MeshField::fail("input mesh must be 2d\n");
    }
    const auto ShapeOrder = ShapeField::Order;
    if (ShapeOrder != 1 && ShapeOrder != 2) {
      MeshField::fail("input field order must be 1 or 2\n");
    }

    const auto [shp, map] = Omegah::getTriangleElement<ShapeOrder>(mesh);

    MeshField::FieldElement<ShapeField, decltype(shp), decltype(map)> f(
        meshInfo.numTri, field, shp, map);
    auto eval = MeshField::evaluate(f, localCoords, offsets);
    return eval;
  }

  // Evaluate a ReducedQuintic field at the specified local coordinates for each triangle.
  //
  // NOTE: This is separate from triangleLocalPointEval because the ReducedQuintic
  // element has a different DOF layout and shape function. triangleLocalPointEval
  // handles standard elements, while ReducedQuintic uses
  // a different set of degrees of freedom and shape functions.
  template <typename ViewType, typename ShapeField>
  auto triangleReducedQuinticEval(const ViewType &localCoords,
                                  Kokkos::View<LO *> offsets,
                                  const ShapeField &field) const {
    const auto MeshDim = 2;
    if (mesh.dim() != MeshDim) {
      MeshField::fail("input mesh must be 2d\n");
    }

    const auto [shp, map, coeffs] = Omegah::getReducedQuinticTriangleElement(mesh);

    MeshField::FieldElement<ShapeField, decltype(shp), decltype(map)> f(
        meshInfo.numTri, field, shp, map, coeffs);
    auto eval = MeshField::evaluate(f, localCoords, offsets);
    return eval;
  }

  // evaluate a ReducedQuintic field at the specified local coordinate for each triangle
  template <typename ViewType, typename ShapeField>
  auto triangleReducedQuinticEval(const ViewType &localCoords,
                                  size_t NumPtsPerElem,
                                  const ShapeField &field) const {
    auto offsets = createOffsets(meshInfo.numTri, NumPtsPerElem);
    return triangleReducedQuinticEval<ViewType, ShapeField>(localCoords, offsets, field);
  }

  template <typename ViewType, typename ShapeField>
  auto tetrahedronLocalPointEval(const ViewType &localCoords,
                                 size_t NumPtsPerElem,
                                 const ShapeField &field) const {
    auto offsets = createOffsets(meshInfo.numTet, NumPtsPerElem);
    auto eval = tetrahedronLocalPointEval(localCoords, offsets, field);
    return eval;
  }

  template <typename ViewType, typename ShapeField>
  auto tetrahedronLocalPointEval(const ViewType &localCoords,
                                 Kokkos::View<LO *> offsets,
                                 const ShapeField &field) const {
    const auto MeshDim = 3;
    if (mesh.dim() != MeshDim) {
      MeshField::fail("input mesh must be 3d\n");
    }
    const auto ShapeOrder = ShapeField::Order;
    if (ShapeOrder != 1 && ShapeOrder != 2) {
      MeshField::fail("input field order must be 1 or 2\n");
    }
    const auto [shp, map] = Omegah::getTetrahedronElement<ShapeOrder>(mesh);
    MeshField::FieldElement<ShapeField, decltype(shp), decltype(map)> f(
        meshInfo.numTet, field, shp, map);
    auto eval = MeshField::evaluate(f, localCoords, offsets);
    return eval;
  }
};

} // namespace MeshField

#endif
