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
// we use functors because cannot use auto return type with KOKKOS_LAMBDAS
template <int dim, typename Field, typename Coords, typename CoordinateMap>
struct SetCoordFieldFunctor
{
  SetCoordFieldFunctor(Field& field, Coords& coords, int meshDim,
                       const CoordinateMap& map) :
    field_(field), coords_(coords), meshDim_(meshDim), map_(map) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t vtx) const {
    Kokkos::Array<MeshField::Real, dim> x{};
    for (int d = 0; d < meshDim_; ++d)
      x[d] = coords_[vtx * meshDim_ + d];
    const auto mapped = map_(x);
    for (int d = 0; d < meshDim_; ++d)
      field_(vtx, 0, d, MeshField::Vertex) = mapped[d];
  }
  Field field_;
  Coords coords_;
  int meshDim_;
  const CoordinateMap map_;
};


template <int dim, typename Element, typename CoordinateMap>
struct InterpolateElementFunctor
{
  InterpolateElementFunctor(const Element& element, const CoordinateMap& map)
      : element_(element), map_(map) {}

  KOKKOS_INLINE_FUNCTION
  auto operator()(const int ent, Kokkos::Array<MeshField::Real, dim> const& xi) const {
        return map_(element_.getValue(ent, xi));
  }

  const Element element_;
  const CoordinateMap map_;
};

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
        Omega_h::simplex_down_template(triDim, vtxDim, triNodeIdx, ignored);
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
    const auto localVtxIdx = Omega_h::simplex_down_template(tetDim, vtxDim, tetNodeIdx, ignored);
    const auto tetToVtxDegree = Omega_h::simplex_degree(tetDim, vtxDim);
    const MeshField::LO vtx = tetVerts[(tet * tetToVtxDegree) + localVtxIdx];
    return {0, tetCompIdx, vtx, MeshField::Vertex}; // node, comp, ent, topo
  }
};
//! [QuadraticTriangleToField]
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
      const auto localVtxIdx = Omega_h::simplex_down_template(
                                    triDim, vtxDim, dofHolderIdx, ignored);
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
//! [QuadraticTriangleToField]

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
    const MeshField::LO tetNode2DofHolder[10] = {0, 1, 2, 3, 0, 2, 3, 1, 5, 4};
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
      const auto localVtxIdx = Omega_h::simplex_down_template(
                                    tetDim, vtxDim, dofHolderIdx, ignored);
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

//! [getTriangleElement]
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
//! [getTriangleElement]
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

/**
 * @brief Maps the nodes of LinearWedgeShape wedges, extruded from a 2D
 * triangle mesh, to vertices
 * @details
 * The 2D mesh is stacked into numLayers layers. Every plane has the vertex
 * numbering and triangle connectivity of the 2D mesh; only the coordinates,
 * which the caller sets, differ, so the triangles may change shape from plane
 * to plane. Vertex v on plane p is vertex p * nVtx2d + v. Wedge w is in layer
 * k = w / nTri2d and comes from triangle t = w % nTri2d. Layer k runs from
 * plane k to plane k + 1, or to plane (k + 1) % numLayers when periodic.
 * Wedge corner a is local vertex a of triangle t, or local vertex {0, 2, 1}[a]
 * when reverseOrientation is true. Node a is corner a on plane k and node
 * a + 3 is corner a on the top plane of the layer. These vertex ids, and the
 * vertical edge ids of ExtrudedLinearCubicWedgeToField, number the extruded
 * mesh; they are not entity ids of an Omega_h mesh.
 *
 * With coordinates (R, phi, Z), phi increasing with the layer and triangles
 * counterclockwise in (R, Z), det J < 0. Setting reverseOrientation = true
 * makes it positive.
 */
struct ExtrudedLinearWedgeToVertexField {
  Omega_h::LOs triVerts;
  int nVtx2d;
  int nTri2d;
  int numLayers;
  bool periodic;
  bool reverseOrientation;
  ExtrudedLinearWedgeToVertexField(Omega_h::Mesh &mesh2d, int numLayersIn,
                                   bool periodicIn, bool reverseOrientationIn)
      : triVerts(mesh2d.ask_elem_verts()), nVtx2d(mesh2d.nverts()),
        nTri2d(mesh2d.nfaces()), numLayers(numLayersIn), periodic(periodicIn),
        reverseOrientation(reverseOrientationIn) {
    if (mesh2d.dim() != 2 || mesh2d.family() != OMEGA_H_SIMPLEX) {
      MeshField::fail(
          "The mesh passed to %s must be 2D and simplex (triangles)\n",
          __func__);
    }
    if (numLayers < 1 || (periodic && numLayers < 2)) {
      MeshField::fail("%s needs numLayers >= 1, and >= 2 when periodic; got "
                      "%d\n",
                      __func__, numLayers);
    }
  }

  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Wedge};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO wedgeNodeIdx, MeshField::LO wedgeCompIdx,
             MeshField::LO wedge, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Wedge);
    const MeshField::LO layer = wedge / nTri2d;
    const MeshField::LO tri = wedge % nTri2d;
    const MeshField::LO corner = wedgeNodeIdx % 3;
    // reverseOrientation swaps corners 1 and 2
    const MeshField::LO triNodeIdx =
        (reverseOrientation && corner != 0) ? 3 - corner : corner;
    const auto triDim = 2;
    const auto vtxDim = 0;
    const auto ignored = -1;
    const auto localVtxIdx =
        Omega_h::simplex_down_template(triDim, vtxDim, triNodeIdx, ignored);
    const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
    const MeshField::LO vtx2d = triVerts[(tri * triToVtxDegree) + localVtxIdx];
    const MeshField::LO top = periodic ? (layer + 1) % numLayers : layer + 1;
    const MeshField::LO plane = (wedgeNodeIdx < 3) ? layer : top;
    return {0, wedgeCompIdx, plane * nVtx2d + vtx2d, MeshField::Vertex};
  }
};

/**
 * @brief Maps the nodes of LinearCubicWedgeShape wedges, extruded from a 2D
 * triangle mesh, to vertices and vertical edges
 * @details
 * Nodes 0-5 map as in ExtrudedLinearWedgeToVertexField. The vertical edge
 * above 2D vertex v in layer k is edge k * nVtx2d + v. Nodes 6 + 2a and
 * 7 + 2a are DOFs 0 (zeta = -1/3) and 1 (zeta = +1/3) of the edge above
 * corner a. The orientation note of ExtrudedLinearWedgeToVertexField applies.
 */
struct ExtrudedLinearCubicWedgeToField {
  Omega_h::LOs triVerts;
  int nVtx2d;
  int nTri2d;
  int numLayers;
  bool periodic;
  bool reverseOrientation;
  ExtrudedLinearCubicWedgeToField(Omega_h::Mesh &mesh2d, int numLayersIn,
                                  bool periodicIn, bool reverseOrientationIn)
      : triVerts(mesh2d.ask_elem_verts()), nVtx2d(mesh2d.nverts()),
        nTri2d(mesh2d.nfaces()), numLayers(numLayersIn), periodic(periodicIn),
        reverseOrientation(reverseOrientationIn) {
    if (mesh2d.dim() != 2 || mesh2d.family() != OMEGA_H_SIMPLEX) {
      MeshField::fail(
          "The mesh passed to %s must be 2D and simplex (triangles)\n",
          __func__);
    }
    if (numLayers < 1 || (periodic && numLayers < 2)) {
      MeshField::fail("%s needs numLayers >= 1, and >= 2 when periodic; got "
                      "%d\n",
                      __func__, numLayers);
    }
  }

  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Wedge};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO wedgeNodeIdx, MeshField::LO wedgeCompIdx,
             MeshField::LO wedge, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Wedge);
    const MeshField::LO layer = wedge / nTri2d;
    const MeshField::LO tri = wedge % nTri2d;
    const bool isVtx = wedgeNodeIdx < 6;
    const MeshField::LO corner =
        isVtx ? wedgeNodeIdx % 3 : (wedgeNodeIdx - 6) / 2;
    // reverseOrientation swaps corners 1 and 2
    const MeshField::LO triNodeIdx =
        (reverseOrientation && corner != 0) ? 3 - corner : corner;
    const auto triDim = 2;
    const auto vtxDim = 0;
    const auto ignored = -1;
    const auto localVtxIdx =
        Omega_h::simplex_down_template(triDim, vtxDim, triNodeIdx, ignored);
    const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
    const MeshField::LO vtx2d = triVerts[(tri * triToVtxDegree) + localVtxIdx];
    if (isVtx) {
      const MeshField::LO top = periodic ? (layer + 1) % numLayers : layer + 1;
      const MeshField::LO plane = (wedgeNodeIdx < 3) ? layer : top;
      return {0, wedgeCompIdx, plane * nVtx2d + vtx2d, MeshField::Vertex};
    }
    return {(wedgeNodeIdx - 6) % 2, wedgeCompIdx, layer * nVtx2d + vtx2d,
            MeshField::Edge};
  }
};

/**
 * @brief MeshInfo of the wedges from getExtrudedWedgeElement
 * @details numEdge counts only the vertical edges, which hold the edge DOFs
 * of LinearCubicWedgeShape
 */
inline MeshField::MeshInfo getExtrudedWedgeMeshInfo(Omega_h::Mesh &mesh2d,
                                                    int numLayers,
                                                    bool periodic = false) {
  const int numPlanes = periodic ? numLayers : numLayers + 1;
  MeshField::MeshInfo meshInfo;
  meshInfo.dim = 3;
  meshInfo.numVtx = numPlanes * mesh2d.nverts();
  meshInfo.numEdge = numLayers * mesh2d.nverts();
  meshInfo.numWedge = numLayers * mesh2d.nfaces();
  return meshInfo;
}

/**
 * @brief Shape and element map for wedges extruded from a 2D triangle mesh
 * @tparam ShapeOrder 1 for LinearWedgeShape, 3 for LinearCubicWedgeShape
 * @details See ExtrudedLinearWedgeToVertexField for the numbering and
 * orientation. The caller fills the coordinate field, sized by
 * getExtrudedWedgeMeshInfo.
 */
template <int ShapeOrder>
auto getExtrudedWedgeElement(Omega_h::Mesh &mesh2d, int numLayers,
                             bool periodic = false,
                             bool reverseOrientation = false) {
  static_assert(ShapeOrder == 1 || ShapeOrder == 3);
  if constexpr (ShapeOrder == 1) {
    struct result {
      MeshField::LinearWedgeShape shp;
      ExtrudedLinearWedgeToVertexField map;
    };
    return result{MeshField::LinearWedgeShape(),
                  ExtrudedLinearWedgeToVertexField(mesh2d, numLayers, periodic,
                                                   reverseOrientation)};
  } else if constexpr (ShapeOrder == 3) {
    struct result {
      MeshField::LinearCubicWedgeShape shp;
      ExtrudedLinearCubicWedgeToField map;
    };
    return result{MeshField::LinearCubicWedgeShape(),
                  ExtrudedLinearCubicWedgeToField(mesh2d, numLayers, periodic,
                                                  reverseOrientation)};
  }
}

} // namespace Omegah

/**
 * @brief the CoordinateMap that leaves the mesh geometry unchanged
 */
struct IdentityCoordinateMap {
  template <typename Point>
  KOKKOS_INLINE_FUNCTION Point operator()(Point const &x) const {
    return x;
  }
};

template <typename ExecutionSpace, size_t dim,
          template <typename...> typename Controller =
              MeshField::KokkosController>
class OmegahMeshField {
private:
  Omega_h::Mesh &mesh;
  const MeshField::MeshInfo meshInfo;
  using CoordField =
      decltype(MeshField::CreateCoordinateField<ExecutionSpace, Controller,
                                                dim>(MeshField::MeshInfo()));
  CoordField coordField;

public:
  OmegahMeshField(Omega_h::Mesh &mesh_in)
      : mesh(mesh_in), meshInfo(getMeshInfo(mesh)),
        coordField(CreateLagrangeCoordinateField<1>()) {
    static_assert(dim == 1 || dim == 2 || dim == 3);
  }

  template <typename DataType, size_t order, size_t numComp>
  // Ordering of field indexing changed to 'entity, node, component'
  auto CreateLagrangeField() const {
    return MeshField::CreateLagrangeField<ExecutionSpace, Controller, DataType,
                                          order, dim, numComp>(meshInfo);
  }

  /**
   * @brief create a Lagrange coordinate field of the given order
   *
   * @details
   * Each node is placed at map(x), where x is the node's position on the
   * linear (straight-sided) mesh geometry.  The linear field is set from the
   * mesh coordinates and higher order fields are interpolated from the
   * linear coordinate field, so with the default IdentityCoordinateMap the
   * geometry is the mesh's own and is straight-sided.  A nonlinear map gives
   * curved geometry for orders above one.
   *
   * @tparam order the order of the coordinate field
   * @param map a device callable taking and returning a
   * Kokkos::Array<Real, dim> of physical coordinates
   */
  template <size_t order, typename CoordinateMap = IdentityCoordinateMap>
  auto CreateLagrangeCoordinateField(
      const CoordinateMap &map = CoordinateMap()) const {
    auto fieldWithCtrlr = CreateLagrangeField<Real, order, dim>();
    auto field = fieldWithCtrlr.field;
    // note order 1 is called in the constructor to initialize the coordinate field
    // it must be called before higher order coordinate fields are constructed
    if constexpr (order == 1) {
      const auto meshDim = meshInfo.dim;
      const auto coords = mesh.coords();
      MeshField::parallel_for<decltype(fieldWithCtrlr.ctrlr)>(
          {0}, {meshInfo.numVtx},
          SetCoordFieldFunctor<dim, decltype(field), decltype(coords),
                               CoordinateMap>(field, coords, meshDim, map),
          "setCoordField");
    } else {
      const auto linear = CreateLagrangeElement(coordField.field);
      auto target = CreateLagrangeElement(field);
      MeshField::setDofByInterpolation(
          target,
          InterpolateElementFunctor<dim, decltype(linear), CoordinateMap>(
              linear, map));
    }
    return fieldWithCtrlr;
  }

  auto getCoordField() { return coordField; }

  /**
   * @brief create a FieldElement over the mesh elements for a Lagrange field
   *
   */
  template <typename Field>
  auto CreateLagrangeElement(const Field &field) const {
    static_assert(dim == 2 || dim == 3,
                  "CreateLagrangeElement supports triangles and tetrahedra");
    constexpr int order = Field::Order;
    if constexpr (dim == 2) {
      const auto [shp, map] = Omegah::getTriangleElement<order>(mesh);
      return MeshField::FieldElement(meshInfo.numTri, field, shp, map);
    } else {
      const auto [shp, map] = Omegah::getTetrahedronElement<order>(mesh);
      return MeshField::FieldElement(meshInfo.numTet, field, shp, map);
    }
  }

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
