#ifndef MESHFIELD_SHAPEFIELD_HPP
#define MESHFIELD_SHAPEFIELD_HPP

#include "KokkosController.hpp"
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif
#include "MeshField_Field.hpp"
#include "MeshField_Shape.hpp"
#include <type_traits> //decltype

namespace MeshField {

/**
 * @brief
 * On-process mesh metadata
 */
struct MeshInfo {
  int numVtx;     // entDim = 0
  int numEdge;    // entDim = 1
  int numTri;     // entDim = 2
  int numQuad;    // entDim = 2
  int numTet;     // entDim = 3
  int numHex;     // entDim = 3
  int numPrism;   // entDim = 3
  int numPyramid; // entDim = 3
  int dim;
};

/**
 * @brief
 * Enable definition of field classes with multiple inheritance.
 * @details
 * The field definition (e.g., linear triangle, quadratic tet, etc.) dictates
 * what combination of interfaces need to be exposed.
 * For example, the functions that provide the coefficients (name?) (e.g.,
 * QuadraticTriangleShape) needs to return the values for each dof holder and
 * associated metadata to define their association with mesh entities, the
 * mesh entity topology, and how many components exist per dof.
 * Likewise, the interface that allows getting and setting field values needs to
 * be customized to accomodate the field interface.
 *
 * To avoid inheritance, virtual functions, and to allow RAII, the 'mixin'
 * technique enables definition of a class that inherites from multiple
 * parent classes.
 *
 * @tparam MeshFieldType MeshField type templated on a Controller (i.e.,
 * KokkosController or CabanaController)
 * @tparam Shape Defines the shape function order and mesh topology (i.e.,
 * QuadraticTriangleShape, QuadraticTetrahedronShape, ...)
 * @tparam Mixins Accessor type that provides paren operator to underlying slice
 * (i.e., LinearAccessor, QuadraticAccessor)
 *
 * @param meshFieldIn see MeshFieldType
 * @param meshInfoIn defines on-process mesh metadata
 * @param mixins object(s) needed to construct the Accessor
 */
template <typename MeshFieldType, typename Shape, typename... Mixins>
struct ShapeField : public Mixins... {
  MeshFieldType meshField;
  Shape shape;
  MeshInfo meshInfo;
  constexpr static auto Order = Shape::Order;
  ShapeField(MeshFieldType &meshFieldIn, MeshInfo meshInfoIn, Mixins... mixins)
      : meshField(meshFieldIn), meshInfo(meshInfoIn), Mixins(mixins)... {};
};

/**
 * @brief
 * Defines a parenthesis operator to provide dof component read/write access
 * for fields on simplex (i.e., tri and tet) and hypercube (i.e., quad and hex)
 * meshes using quadratic shape functions
 * @details
 * The dof holders for fields using quadratic shape functions are associated
 * with mesh edges and mesh vertices.
 * Values associated with each topological mesh order (edge, vtx, etc.) are
 * stored in their own structure (a MeshField slice).
 * As such, the QuadraticAccessor is a composition of a VtxAccessor and
 * EdgeAccessor that respectively define the parenthesis operators for vertices
 * and edges.
 *
 * @tparam VtxAccessor defines parenthesis operator for dofs associated with
 * mesh vertices
 * @tparam EdgeAccessor defines parenthesis operator for dofs associated with
 * mesh edges
 */
template <typename VtxAccessor, typename EdgeAccessor>
struct QuadraticAccessor {
  constexpr static const Mesh_Topology topo[2] = {Vertex, Edge};
  VtxAccessor vtxField;
  EdgeAccessor edgeField;
  using BaseType = typename VtxAccessor::BaseType;

  KOKKOS_FUNCTION
  auto &operator()(int node, int component, int entity, Mesh_Topology t) const {
    if (t == Vertex) {
      return vtxField(node, component, entity);
    } else if (t == Edge) {
      return edgeField(node, component, entity);
    } else {
      Kokkos::printf("%d is not a support topology\n", t);
      assert(false);
    }
  }
};

/**
 * @brief
 * Defines a parenthesis operator to provide dof component read/write access
 * for fields on simplex (i.e., tri and tet) and hypercube (i.e., quad and hex)
 * meshes using linear shape functions
 * @details
 * The dof holders for fields using linear shape functions are associated
 * with mesh vertices.
 * As such, the LinearAccessor has a VtxAccessor that defines the parenthesis
 * operators for vertices.
 *
 * @tparam VtxAccessor defines parenthesis operator for dofs associated with
 * mesh vertices
 */
template <typename VtxAccessor> struct LinearAccessor {
  constexpr static const Mesh_Topology topo[1] = {Vertex};
  VtxAccessor vtxField;
  using BaseType = typename VtxAccessor::BaseType;

  KOKKOS_FUNCTION
  auto &operator()(int node, int component, int entity, Mesh_Topology t) const {
    if (t == Vertex) {
      return vtxField(node, component, entity);
    } else {
      Kokkos::printf("%d is not a support topology\n", t);
      assert(false);
    }
  }
};

/**
 * @brief
 * Create a field using linear or quadratic Lagrange shape functions
 * @details
 * The key to this function is creation of a ShapeField instance from
 * the MeshField, the fields it provides, and the required accessors to those
 * fields.
 *
 * @todo test with the Cabana Controller
 *
 * @tparam ExecutionSpace a Kokkos ExecutionSpace (i.e., Cuda, Serial, etc.)
 * @tparam DataType the primative datatype for storing field entries; 32b or 64b
 * floats are supported
 * @tparam order the order of the shape functions; linear(1) and quadratic(2)
 * are supported
 * @tparam dim the dimension of the mesh
 *
 * @param meshInfo defines on-process mesh metadata
 * @return a linear or quadratic ShapeField
 */
template <typename ExecutionSpace,
          template <typename...>
          typename Controller = MeshField::KokkosController,
          typename DataType, size_t order, size_t dim>
auto CreateLagrangeField(const MeshInfo &meshInfo) {
  static_assert((std::is_same_v<Real4, DataType> == true ||
                 std::is_same_v<Real8, DataType> == true),
                "CreateLagrangeField only supports single and double precision "
                "floating point fields\n");
  static_assert(
      (order == 1 || order == 2),
      "CreateLagrangeField only supports linear and quadratic fields\n");
  static_assert((dim == 1 || dim == 2 || dim == 3),
                "CreateLagrangeField only supports 1d, 2d, and 3d meshes\n");
  using MemorySpace = typename ExecutionSpace::memory_space;
  if constexpr (order == 1 && (dim == 1 || dim == 2)) {
    if (meshInfo.numVtx <= 0) {
      fail("mesh has no vertices\n");
    }
    using Ctrlr = Controller<MemorySpace, ExecutionSpace, DataType ***>;
    // 1 dof with 1 component per vtx
    Ctrlr kk_ctrl({/*field 0*/ 1, 1, meshInfo.numVtx});
    auto vtxField = MeshField::makeField<Ctrlr, 0>(kk_ctrl);
    using LA = LinearAccessor<decltype(vtxField)>;
    using LinearLagrangeShapeField = ShapeField<Ctrlr, LinearTriangleShape, LA>;
    LinearLagrangeShapeField llsf(kk_ctrl, meshInfo, {vtxField});
    return llsf;
  } else if constexpr (order == 2 && (dim == 2 || dim == 3)) {
    if (meshInfo.numVtx <= 0) {
      fail("mesh has no vertices\n");
    }
    if (meshInfo.numEdge <= 0) {
      fail("mesh has no edges\n");
    }
    using Ctrlr =
        Controller<MemorySpace, ExecutionSpace, DataType ***, DataType ***>;
    // 1 dof with 1 comp per vtx/edge
    Ctrlr kk_ctrl({/*field 0*/ 1, 1, meshInfo.numVtx,
                   /*field 1*/ 1, 1, meshInfo.numEdge});
    auto vtxField = MeshField::makeField<Ctrlr, 0>(kk_ctrl);
    auto edgeField = MeshField::makeField<Ctrlr, 1>(kk_ctrl);
    using QA = QuadraticAccessor<decltype(vtxField), decltype(edgeField)>;
    using QuadraticLagrangeShapeField =
        ShapeField<Ctrlr, QuadraticTriangleShape, QA>;
    QuadraticLagrangeShapeField qlsf(kk_ctrl, meshInfo, {vtxField, edgeField});
    return qlsf;
  } else {
    fail("CreateLagrangeField does not support the specified "
         "combination of order %d and dimension %d.\n",
         order, dim);
    return nullptr; // silence compiler warning
  }
};

/**
 * @brief
 * Create a coordinate field using linear Lagrange shape functions
 * @details
 * The key to this function is creation of a ShapeField instance from
 * the MeshField, a vertex field it provides, and a LinearAccessor to it.
 * The field's primative datatype is hardcoded to use 64b floats.
 * Note, the user must set the field entries with coordinates from the mesh.
 *
 * @todo test with Cabana Controller
 *
 * @tparam ExecutionSpace a Kokkos ExecutionSpace (i.e., Cuda, Serial, etc.)
 *
 * @param meshInfo defines on-process mesh metadata
 * @return a linear ShapeField
 */
template <typename ExecutionSpace, template <typename...> typename Controller =
                                       MeshField::KokkosController>
auto CreateCoordinateField(const MeshInfo &meshInfo) {
  if (meshInfo.numVtx <= 0) {
    fail("mesh has no vertices\n");
  }
  using DataType = Real;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using Ctrlr = Controller<MemorySpace, ExecutionSpace, DataType ***>;
  const int numComp = meshInfo.dim;
  Ctrlr kk_ctrl({/*field 0*/ 1, numComp, meshInfo.numVtx});
  auto vtxField = MeshField::makeField<Ctrlr, 0>(kk_ctrl);
  using LA = LinearAccessor<decltype(vtxField)>;
  using LinearLagrangeShapeField = ShapeField<Ctrlr, LinearTriangleShape, LA>;
  LinearLagrangeShapeField llsf(kk_ctrl, meshInfo, {vtxField});
  return llsf;
};

} // namespace MeshField

#endif
