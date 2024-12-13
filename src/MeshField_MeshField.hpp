#include "KokkosController.hpp"

namespace {
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





} //namespace

namespace MeshField {


template <typename Controller>
class MeshField {
  private:
  Controller ctrl;

  public:
  MeshField(Controller ctrl_in) : ctrl(ctrl_in) {}
  template <typename Field>
    virtual void writeVtk(Field &field) = 0;
  template <typename AnalyticFunction, typename Field>
    virtual Field triangleLocalPointEval(Kokkos::View<MeshField::Real *[3]> localCoords,
        size_t NumPtsPerElem, AnalyticFunction func, Field field);
};


template <typename Controller>
class OmegahMeshField : public MeshField {
  private:
    Omega_h::Mesh& mesh;

  public:
  MeshField(Omega_h::Mesh mesh_in, Controller ctrl_in) : mesh(mesh_in), ctrl(ctrl_in) {}

  template <typename Field> void writeVtk(Field &field) {
    using FieldDataType = typename decltype(field.vtxField)::BaseType;
    // HACK assumes there is a vertex field.. in the Field Mixin object
    auto field_view = field.vtxField.serialize();
    mesh.writeVtk("foo.vtk", &mesh, mesh.dim());
    //Omega_h::Write<FieldDataType> field_write(field_view);
    //mesh.add_tag(0, "field", 1, Omega_h::read(field_write));
    //Omega_h::vtk::write_parallel("foo.vtk", &mesh, mesh.dim());
  }

  // evaluate a field at the specified local coordinate for each triangle
  template <typename AnalyticFunction, typename Field>
    Field triangleLocalPointEval(Kokkos::View<MeshField::Real *[3]> localCoords,
        size_t NumPtsPerElem, AnalyticFunction func, Field field) {
      const auto MeshDim = 2;
      if (mesh.dim() != MeshDim) {
        MeshField::fail("input mesh must be 2d\n");
      }
      if (ShapeOrder != 1 && ShapeOrder != 2) {
        MeshField::fail("field order must be 1 or 2\n");
      }
      const auto meshInfo = getMeshInfo(mesh);
      auto field = MeshField::CreateLagrangeField<ExecutionSpace, MeshField::Real,
           ShapeOrder, MeshDim>(meshInfo);

      // TODO - define interface that takes function of cartesian coords / real
      //  space and gets the cartesian position of 'nodes' and sets the field
      //  at the node using the result of the function
      //  - mfem calls this 'project'
      // set field based on analytic function
      auto coords = mesh.coords();
      auto setField = KOKKOS_LAMBDA(const int &i) {
        const auto x = coords[i * MeshDim];
        const auto y = coords[i * MeshDim + 1];
        field(0, 0, i, MeshField::Vertex) = func(x, y);
      };
      MeshField::parallel_for(ExecutionSpace(), {0}, {meshInfo.numVtx}, setField,
          "setField");
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
        MeshField::parallel_for(ExecutionSpace(), {0}, {meshInfo.numEdge},
            setFieldAtEdges, "setFieldAtEdges");
      }

      if (ShapeOrder == 1) {
        writeVtk(mesh, field);
      }

      auto coordField = MeshField::CreateCoordinateField<ExecutionSpace>(meshInfo);
      auto setCoordField = KOKKOS_LAMBDA(const int &i) {
        coordField(0, 0, i, MeshField::Vertex) = coords[i * MeshDim];
        coordField(0, 1, i, MeshField::Vertex) = coords[i * MeshDim + 1];
      };
      MeshField::parallel_for(ExecutionSpace(), {0}, {meshInfo.numVtx},
          setCoordField, "setCoordField");

      const auto [shp, map] = getTriangleElement<ShapeOrder>(mesh);

      MeshField::FieldElement f(meshInfo.numTri, field, shp, map);
      Kokkos::View<MeshField::LO *> offsets("offsets", meshInfo.numTri + 1);
      Kokkos::parallel_for(
          "setOffsets", meshInfo.numTri,
          KOKKOS_LAMBDA(int i) { offsets(i) = i * NumPtsPerElem; });
      Kokkos::deep_copy(Kokkos::subview(offsets, offsets.size() - 1),
          meshInfo.numTri * NumPtsPerElem);
      auto eval = MeshField::evaluate(f, localCoords, offsets);

      MeshField::FieldElement fcoords(meshInfo.numTri, coordField,
          MeshField::LinearTriangleCoordinateShape(),
          LinearTriangleToVertexField(mesh));
      auto globalCoords = MeshField::evaluate(fcoords, localCoords, offsets);
    }
};

} //MeshField

