#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>

namespace MeshField {

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
 * Supports the evaluation of a field, and other per-element
 * operations, given the definition of the element ((the topological type of
 * mesh entity to operate over (i.e., edge, tri, quad, tet,
 * etc.)) and shape functions used by the field.
 * Operations are expected to be executed concurrently across all (or a subset
 * of) elements in the on-process mesh (e.g., within a parallel for loop).
 *
 * @tparam FieldAccessor provides parenthesis operator to access the field (see
 * CreateLagrangeField and CreateCoordinateField)
 * @tparam ShapeType Defines the shape function order and mesh topology (i.e.,
 * QuadraticTriangleShape, QuadraticTetrahedronShape, ...)
 * @tparam ElementDofHolderAccessor provides the mapping from
 * the element indices to the underlying field storage
 *
 * @todo add static asserts for functions provided by the templated types
 *
 * @param in_numMeshEnts number of mesh entities that are associated with the
 * ElementType
 * @param fieldIn see FieldAccessor
 * @param elmIn see ElementType
 */
template <typename FieldAccessor, typename ShapeType,
          typename ElementDofHolderAccessor>
struct FieldElement {
  const size_t numMeshEnts;
  const FieldAccessor field;
  ShapeType shapeFn;
  ElementDofHolderAccessor elm2dof;

  static const size_t MeshEntDim = ShapeType::meshEntDim;
  FieldElement(size_t numMeshEntsIn, const FieldAccessor &fieldIn,
               const ShapeType shapeFnIn,
               const ElementDofHolderAccessor elm2dofIn)
      : numMeshEnts(numMeshEntsIn), field(fieldIn), shapeFn(shapeFnIn),
        elm2dof(elm2dofIn) {}
  /* general template for baseType which simply sets type
   */
  template <typename T> struct baseType {
    using type = T;
  };
  /* template specialization to recursively strip type to get base type
   * Example: int[5][6] => int[6] => int
   */
  template <typename T, size_t N> struct baseType<T[N]> {
    using type = typename baseType<T>::type;
  };
  using ValArray =
      Kokkos::Array<typename baseType<typename FieldAccessor::BaseType>::type,
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
   * @details
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
    const auto shapeValues = shapeFn.getValues(localCoord);
    for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci)
      c[ci] = 0;
    for (auto topo : elm2dof.getTopology()) { // element topology
      for (int ni = 0; ni < shapeFn.numNodes; ++ni) {
        for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci) {
          auto map = elm2dof(ni, ci, ent, topo);
          const auto fval =
              field(map.entity, map.node, map.component, map.topo);
          c[ci] += fval * shapeValues[ni];
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
 * @todo add static asserts for values and functions provided by the templated
 * types
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
 * @details
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

/**
 * @brief
 * Given an array of parametric coordinates 'localCoords',
 * with numPtsPerElement points per mesh element,
 * evaluate the fields value within each element.
 *
 * @param (in) numPtsPerElement
 *
 * @details
 * see evaluate function accepting offsets
 */
template <typename FieldElement>
Kokkos::View<Real *[FieldElement::NumComponents]> evaluate(
    FieldElement &fes, Kokkos::View<Real **> localCoords,
    size_t numPtsPerElement) {
  Kokkos::View<LO *> offsets("offsets", fes.numMeshEnts + 1);
  Kokkos::parallel_for(
      fes.numMeshEnts + 1,
      KOKKOS_LAMBDA(const int ent) { offsets(ent) = ent * numPtsPerElement; });
  return evaluate(fes, localCoords, offsets);
}

} // namespace MeshField
#endif
