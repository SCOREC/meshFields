#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>

namespace MeshField {

// clang-format off
  /**
   * @mainpage
   * ## Nomenclature
   *
   * - Mesh - conforming discretization of a domain using simplices (triangles
   *   and tetrahedrons), hypercubes (quads and hexahedrons), pyramids and prisms
   * - Field - "describe the distribution of input and solution tensors over the
   *   mesh entities" [Beall 1999; Simmetrix Web 2015]
   * - Field Element - localization of the field values on the closure of a
   *   mesh entity (e.g., the vertices and edges bounding a mesh face) to
   *   support operations that require those values. For example,
   *   integration, evaluation of the field at a location within the entity.
   * - DOF - degree of freedom, exists at a dof holder, can be scalar, vector, matrix, etc.
   * - DOF holder
   *   - can contain a DOF
   *   - possible holders: mesh entities, quadrature points, element centroids, etc.
   * - Node - a location on a mesh entitiy that is a dof holder. Multiple nodes can
   *   exist per mesh entity.  For example, a mesh edge could have multiple
   *   nodes for a high order shape function.
   *
   * ## References
   *
   * - PUMI users guide: https://www.scorec.rpi.edu/pumi/PUMI.pdf
   * - PUMI TOMS paper: https://www.scorec.rpi.edu/REPORTS/2015-4.pdf
   * - PUMI apf source code
   * - PUMI doxygen: https://www.scorec.rpi.edu/pumi/doxygen/
   *   - with internal headers: https://www.scorec.rpi.edu/~cwsmith/SCOREC/pumiDocs/html/
   * - PUMI APF library documentation:
   *   - The APF Library: https://github.com/SCOREC/core/blob/bcfbd128b65a629241b629c90e3665b539e2e9ae/apf/apf.tex
   *   - The Cavity Operator: https://github.com/SCOREC/core/blob/8959c599cc05e21d0fb470f941d9f892da62aa02/apf/cavity.tex
   *   - Interface For Field Access: https://github.com/SCOREC/core/blob/8959c599cc05e21d0fb470f941d9f892da62aa02/apf/attach.tex
   *   - Hierarchic Shape Functions: https://github.com/SCOREC/core/blob/8959c599cc05e21d0fb470f941d9f892da62aa02/apf/hierarchic.tex
   * - Mark Beall's thesis, Chapter 8: https://scorec.rpi.edu/REPORTS/1999-6.pdf
   *   - An object-oriented field API used within the framework ('Trellis') of
   *     other objects described in the thesis (mesh, model, solution, etc.).
   *   - Designed to support p-adaptivity (i.e., non-uniform field order)
   */
// clang-format on

/**
 * @brief
 * Return type used by structs/classes that implement the
 * ElementDofHolderAccessor parenthesis operator
 */
struct ElementToDofHolderMap {
  LO node;
  LO component;
  LO entity;
  Mesh_Topology topo;
};

/**
 * @brief
 * Combines the shape function definition and the type that provides the mapping
 * from the element indices to the underlying field storage
 * @detail
 * Examples:
 *  Element = Triangle and field storage is at mesh vertices (linear
 *   shape fn)
 *  Element = Edge and field storage is at mesh vertices
 *   (linear shape fn)
 *  Element = Triangle and field storage is at mesh vertices and
 *   edges (quadratic shape fn)
 *
 * @todo
 * consider removing this, the extra object/type isn't needed outside the
 * FieldElement
 *
 * @tparam Shape Defines the shape function order and mesh topology (i.e.,
 * QuadraticTriangleShape, QuadraticTetrahedronShape, ...)
 * @tparam ElementDofHolderAccessor provides the mapping from
 * the element indices to the underlying field storage
 *
 * @param shapeFnIn see Shape
 * @param elm2dofIn see ElementDofHolderAccessor
 */
template <typename Shape, typename ElementDofHolderAccessor> struct Element {
  // TODO add static asserts for variables and functions provided by the
  // templated types
  Shape shapeFn;
  static const size_t MeshEntDim = Shape::meshEntDim; // better way?
  ElementDofHolderAccessor elm2dof;
  Element(const Shape shapeFnIn, const ElementDofHolderAccessor elm2dofIn)
      : shapeFn(shapeFnIn), elm2dof(elm2dofIn) {}
};

/**
 * @brief
 * Supports the evaluation of a field, and other per-element
 * operations, given the definition of the element ((the topological type of
 * mesh entity to operate over (i.e., edge, tri, quad, tet,
 * etc.)) and shape functions used by the field.
 * Operations are expected to be executed concurrently across all (or a subset
 * of) elements in the on-process mesh (e.g., within a parallel for loop).
 *
 * @tparam FieldAccessor provides parenthesis operator to access the field (see
 * CreateLagrangeField and CreateCoordinateField)
 * @tparam ElementType provides the definition of an element in terms of its
 * shape function (ShapeType) and the mapping from mapping from
 * the element indices to the underlying field storage (ElementDofHolderAccesor)
 * @tparam ShapeType see ElementType
 * @tparam ElementDofHolderAccesor see ElementType
 *
 * @param in_numMeshEnts number of mesh entities that are associated with the
 * ElementType
 * @param fieldIn see FieldAccessor
 * @param elmIn see ElementType
 */
template <typename FieldAccessor,
          template <typename, typename> class ElementType, typename ShapeType,
          typename ElementDofHolderAccessor>
struct FieldElement {
  // TODO add static asserts for functions provided by the templated types
  const size_t numMeshEnts;
  const FieldAccessor field;
  ElementType<ShapeType, ElementDofHolderAccessor> elm;
  static const size_t MeshEntDim = ShapeType::meshEntDim;
  FieldElement(size_t in_numMeshEnts, const FieldAccessor &fieldIn,
               const ElementType<ShapeType, ElementDofHolderAccessor> elmIn)
      : numMeshEnts(in_numMeshEnts), field(fieldIn), elm(elmIn) {}

  using ValArray = Kokkos::Array<typename FieldAccessor::BaseType,
                                 ShapeType::numComponentsPerDof>;
  static const size_t NumComponents = ShapeType::numComponentsPerDof;

  /**
   * @brief
   * evaluate the field in the specified element at the specified
   * parametric/local/area coordinate
   *
   * @todo add expression in documetation for summation of shape function *
   * field values
   *
   * @detail
   * heavily based on SCOREC/core @ 7cd76473 apf/apfElement.cc
   *
   * @param ent the mesh entity index
   * @param localCoord the parametric coordinate
   * @return the result of evaluation
   */
  KOKKOS_INLINE_FUNCTION ValArray
  getValue(int ent, Kokkos::Array<Real, MeshEntDim + 1> localCoord) const {
    assert(ent < numMeshEnts);
    ValArray c;
    const auto shapeValues = elm.shapeFn.getValues(localCoord);
    for (int ci = 0; ci < elm.shapeFn.numComponentsPerDof; ++ci)
      c[ci] = 0;
    for (auto topo : elm.elm2dof.getTopology()) { // element topology
      for (int ni = 0; ni < elm.shapeFn.numNodes; ++ni) {
        for (int ci = 0; ci < elm.shapeFn.numComponentsPerDof; ++ci) {
          auto map = elm.elm2dof(ni, ci, ent, topo);
          c[ci] += field(map.node, map.component, map.entity, map.topo) *
                   shapeValues[ni];
        }
      }
    }
    return c;
  }
};

/**
 * @brief
 * Given an array of parametric coordinates 'localCoords', one per mesh element,
 * evaluate the fields value within each element.
 *
 * @todo consider making this a member function of FieldElement
 * @todo support passing a CSR for more than one point eval per element - SPR
 * needs this? - at least need uniform number of points for each element
 *
 * @tparam FieldElement see FieldElement struct
 *
 * @param fes see FieldElement
 * @param localCoords 2D Kokkos::View containing the local/parametric/area
 * coordinates for each element
 * @param offsets 1D Kokkos::View containing the offsets into localCoords,
 *                size = localCoords.extent(0)+1
 * @return Kokkos::View of evaluation results for all the mesh elements
 */
template <typename FieldElement>
Kokkos::View<Real *[FieldElement::NumComponents]>
evaluate(FieldElement &fes, Kokkos::View<Real **> localCoords,
         Kokkos::View<LO *> offsets) {
  // TODO add static asserts for values and functions provided by the templated
  // types
  if (Debug) {
    // check input parametric coords are positive and sum to one
    LO numErrors = 0;
    Kokkos::parallel_reduce(
        "checkCoords", fes.numMeshEnts,
        KOKKOS_LAMBDA(const int &ent, LO &lerrors) {
          Real sum = 0;
          LO isError = 0;
          for (int i = 0; i < localCoords.extent(1); i++) {
            if (localCoords(ent, i) < 0)
              isError++;
            sum += localCoords(ent, i);
          }
          if (Kokkos::fabs(sum - 1) > MachinePrecision)
            isError++;
          lerrors += isError;
        },
        numErrors);
    if (numErrors) {
      fail("One or more of the parametric coordinates passed "
           "to evaluate(...) were invalid\n");
    }
  }
  if (localCoords.extent(0) < fes.numMeshEnts) {
    fail("The size of dimension 0 of the local coordinates input array "
         "must be at least %zu.\n",
         fes.numMeshEnts);
  }
  if (localCoords.extent(1) != fes.MeshEntDim + 1) {
    fail("Dimension 1 of the input array of local coordinates "
         "must have size = %zu.\n",
         fes.MeshEntDim + 1);
  }
  if (offsets.size() != fes.numMeshEnts + 1) {
    fail("The input array of offsets must have size = %zu\n",
         fes.numMeshEnts + 1);
  }
  constexpr const auto numComponents = FieldElement::ValArray::size();
  const auto numPts = MeshFieldUtil::getLastValue(offsets);
  Kokkos::View<Real *[numComponents]> res("result", numPts);
  Kokkos::parallel_for(
      fes.numMeshEnts, KOKKOS_LAMBDA(const int ent) {
        Kokkos::Array<Real, FieldElement::MeshEntDim + 1> lc;
        // TODO use nested parallel for?
        for (auto pt = offsets(ent); pt < offsets(ent + 1); pt++) {
          for (int i = 0; i < localCoords.extent(1); i++) // better way?
            lc[i] = localCoords(pt, i);
          const auto val = fes.getValue(ent, lc);
          for (int i = 0; i < numComponents; i++)
            res(ent, i) = val[i];
        }
      });
  return res;
}

/**
 * @brief
 * Given an array of parametric coordinates 'localCoords', one per mesh element,
 * evaluate the fields value within each element.
 *
 * @detail
 * see evaluate function accepting offsets
 */
template <typename FieldElement>
Kokkos::View<Real *[FieldElement::NumComponents]> evaluate(
    FieldElement &fes, Kokkos::View<Real **> localCoords) {
  Kokkos::View<LO *> offsets("offsets", fes.numMeshEnts + 1);
  Kokkos::parallel_for(
      fes.numMeshEnts + 1,
      KOKKOS_LAMBDA(const int ent) { offsets(ent) = ent; });
  return evaluate(fes, localCoords, offsets);
}

} // namespace MeshField
#endif
