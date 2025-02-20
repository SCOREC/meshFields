#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>

namespace {
  //FIXME - tensorProduct(...) and add(...) need performance and type safety improvements
  /** \brief tensor product of two vectors */
  template <typename VecA, typename VecB>
  KOKKOS_INLINE_FUNCTION
  auto tensorProduct(VecA const& a, VecB const& b) {
    const auto N = VecA::size();
    const auto M = VecB::size();
    static_assert(std::is_same_v< typename VecA::value_type, typename VecB::value_type >);
    Kokkos::Array< Kokkos::Array< typename VecA::value_type, M>, N> matrix;
    //FIXME - is there a better type for 'matrix'?
    for (std::size_t i=0; i < M; ++i) {
      for (std::size_t j=0; j < M; ++j) {
        matrix[i][j] = b[j] * a[i];
      }
    }
    return matrix;
  }

  /** \brief sum two matrices */
  template <typename MatrixA, typename MatrixB>
  KOKKOS_INLINE_FUNCTION
  MatrixA add(MatrixA const& a, MatrixB const& b) {
    static_assert(std::is_same_v< MatrixA, MatrixB >);
    //FIXME ensure that MatrixA is a Kokkos::Array<Kokkos::Array>
    MatrixA matrix;
    for (std::size_t i=0; i < a.size(); ++i) {
      for (std::size_t j=0; j < a[i].size(); ++j) {
        matrix[i][j] = a[i][j] + b[i][j];
      }
    }
    return matrix;
  }
}

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
              field(map.node, map.component, map.entity, map.topo);
          c[ci] += fval * shapeValues[ni];
        }
      }
    }
    return c;
  }

  using NodeArray = Kokkos::Array< typename FieldAccessor::BaseType,
                                   ShapeType::numNodes *
                                   ShapeType::numComponentsPerDof >;
  KOKKOS_INLINE_FUNCTION NodeArray
  getNodeValues(int ent) const {
    NodeArray c;
    for (auto topo : elm2dof.getTopology()) { // element topology
      for (int ni = 0; ni < shapeFn.numNodes; ++ni) {
        for (int ci = 0; ci < shapeFn.numComponentsPerDof; ++ci) {
          auto map = elm2dof(ni, ci, ent, topo);
          const auto fval =
            field(map.node, map.component, map.entity, map.topo);
          c[ni*shapeFn.numComponentsPerDof + ci] = fval;
        }
      }
    }
    return c;
  }

  /**
   * @brief
   * compute the Jacobian of an edge
   *
   * @details
   * heavily based on SCOREC/core @ 7cd76473 apf/apfVectorElement.cc
   *
   * @param ent the mesh entity index
   * @return the result of evaluation
   */
  KOKKOS_INLINE_FUNCTION Real
  getJacobian1d(int ent) const {
    assert(ent < numMeshEnts);
    const auto nodalGradients = shapeFn.getLocalGradients();
    auto nodeValues = getNodeValues(ent);
    auto g = nodalGradients[0]*nodeValues[0];
    for (int i=1; i < shapeFn.numNodes; ++i) {
      g = g + nodalGradients[i]*nodeValues[i];
    }
    return g;
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

/**
 * @brief
 * Given an array of parametric coordinates 'localCoords', one per mesh element,
 * compute the jacobian within each element.
 *
 * @todo add static asserts for values and functions provided by the templated
 * types
 * @todo consider making this a member function of FieldElement
 *
 * @tparam FieldElement see FieldElement struct
 *
 * @param fes see FieldElement
 * @param localCoords 2D Kokkos::View containing the local/parametric/area
 * coordinates for each element
 * @param offsets 1D Kokkos::View containing the offsets into localCoords,
 *                size = localCoords.extent(0)+1
 * @return Kokkos::View containing the jacobian for all the mesh elements
 */
template <typename FieldElement>
Kokkos::View<Real *[FieldElement::NumComponents]>
getJacobians(FieldElement &fes, Kokkos::View<Real **> localCoords,
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
  if (fes.MeshEntDim != 1) {
    fail("getJacobians only currently supports 1d meshes.  Input mesh has %zu dimensions.\n",
         fes.numMeshEnts);
  }
  if (fes.MeshEntDim == 1) {
    constexpr const auto numComponents = FieldElement::ValArray::size();
    const auto numPts = MeshFieldUtil::getLastValue(offsets);
    Kokkos::View<Real *[1]> res("result", numPts);
    Kokkos::parallel_for(
        fes.numMeshEnts, KOKKOS_LAMBDA(const int ent) {
        // TODO use nested parallel for?
        for (auto pt = offsets(ent); pt < offsets(ent + 1); pt++) {
        const auto val = fes.getJacobian1d(ent);
        res(pt,0) = val;
        }
        });
    return res;
  }
}

/**
 * @brief
 * Given an array of parametric coordinates 'localCoords',
 * with numPtsPerElement points per mesh element,
 * compute the jacobian for each element at the specified point
 *
 * @param (in) numPtsPerElement
 *
 * @details
 * see evaluate function accepting offsets
 */
template <typename FieldElement>
Kokkos::View<Real *[FieldElement::NumComponents]> getJacobians(
    FieldElement &fes, Kokkos::View<Real **> localCoords,
    size_t numPtsPerElement) {
  Kokkos::View<LO *> offsets("offsets", fes.numMeshEnts + 1);
  Kokkos::parallel_for(
      fes.numMeshEnts + 1,
      KOKKOS_LAMBDA(const int ent) { offsets(ent) = ent * numPtsPerElement; });
  return getJacobians(fes, localCoords, offsets);
}

} // namespace MeshField
#endif
