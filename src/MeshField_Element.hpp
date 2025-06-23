#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>
#include <type_traits> // has_static_size helper

namespace {
Kokkos::View<MeshField::LO *> getOffsets(MeshField::LO numItems,
                                         MeshField::LO numEntriesPerItem) {
  Kokkos::View<MeshField::LO *> offsets("offsets", numItems + 1);
  auto first = Kokkos::subview(offsets, 0);
  Kokkos::deep_copy(first, 0); // write 0 to the first item
  Kokkos::parallel_for(
      numItems, KOKKOS_LAMBDA(const int i) {
        offsets(i + 1) = (i + 1) * numEntriesPerItem;
      });
  return offsets;
}

// chatgpt prompt 2/20/2025:
//  c++ static assert that checks that a type
//  provides a function named size
template <typename T> class has_size_method {
private:
  template <typename U>
  static auto test(int) -> decltype(std::declval<U>().size(), std::true_type());
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> class has_static_rank {
private:
  template <typename U>
  static auto test(int) -> decltype(U::rank(), std::true_type());
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> class has_extent_method {
private:
  template <typename U>
  static auto test(int)
      -> decltype(std::declval<U>().extent(std::declval<std::size_t>()),
                  std::true_type());
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

// FIXME - addTensorProduct(...) likely needs performance improvements
/** \brief tensor product of two vectors */
template <typename VecA, typename VecB, typename Matrix>
KOKKOS_INLINE_FUNCTION auto addTensorProduct(VecA const &a, VecB const &b,
                                             Matrix &A) {
  static_assert(has_size_method<VecA>::value,
                "VecA must have a size() method.");
  static_assert(has_size_method<VecB>::value,
                "VecB must have a size() method.");
  static_assert(has_static_rank<VecA>::value,
                "VecA must have a static rank() method.");
  static_assert(has_static_rank<VecB>::value,
                "VecB must have a static rank() method.");
  static_assert(
      std::is_same_v<typename VecA::value_type, typename VecB::value_type>);
  static_assert(VecA::rank() == 1);
  static_assert(VecB::rank() == 1);
  static_assert(has_extent_method<Matrix>::value,
                "Matrix must have an extent(size_t) method.");
  const auto M = a.size();
  const auto N = b.size();
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      A(i, j) += b[j] * a[i];
    }
  }
}
} // namespace

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

  using NodeArray =
      Kokkos::Array<typename baseType<typename FieldAccessor::BaseType>::type,
                    ShapeType::meshEntDim * ShapeType::numNodes>;
  KOKKOS_INLINE_FUNCTION NodeArray getNodeValues(int ent) const {
    NodeArray c;
    for (auto topo : elm2dof.getTopology()) { // element topology
      for (int ni = 0; ni < ShapeType::numNodes; ++ni) {
        for (int d = 0; d < ShapeType::meshEntDim; ++d) {
          auto map = elm2dof(ni, d, ent, topo);
          const auto fval =
              field(map.entity, map.node, map.component, map.topo);
          c[ni * ShapeType::meshEntDim + d] = fval;
        }
      }
    }
    return c;
  }

  /**
   * @brief
   * compute the Jacobian of an edge
   *
   * @TODO use the same approach taken for 2d
   *
   * @details
   * heavily based on SCOREC/core @ 7cd76473 apf/apfVectorElement.cc
   *
   * @param ent the mesh entity index
   * @return the result of evaluation
   */
  KOKKOS_INLINE_FUNCTION Real getJacobian1d(int ent) const {
    assert(ent < numMeshEnts);
    const auto nodalGradients = shapeFn.getLocalGradients();
    const auto nodeValues = getNodeValues(ent);
    auto g = nodalGradients[0] * nodeValues[0];
    for (int i = 1; i < shapeFn.numNodes; ++i) {
      g = g + nodalGradients[i] * nodeValues[i];
    }
    return g;
  }

  /**
   * @details
   * heavily based on SCOREC/core @ 7cd76473 apf/apfVectorElement.cc
   */
  template <typename Matrices>
  Kokkos::View<Real *> getJacobianDeterminants(Matrices const &J) {
    static_assert(has_static_rank<Matrices>::value,
                  "Matrices must have a static rank() method.");
    static_assert(has_extent_method<Matrices>::value,
                  "Matrices must have an extent(size_t) method.");
    static_assert(Matrices::rank() == 3); // array of rank two matrices
    if (J.extent(1) != J.extent(2)) {
      fail("getJacobianDeterminant only supports square matrices.  "
           "The given matrices have dimension %lu x %lu \n",
           J.extent(1), J.extent(2));
    }
    const auto dimension = J.extent(1);
    if (dimension > 3 || dimension < 1) {
      fail("getJacobianDeterminant: invalid dimension of input matrix.  "
           "The given matrices have dimension %lu x %lu \n",
           J.extent(0), J.extent(1));
    }
    if (dimension == 3) {
      /* det(J) is also the triple product of the
         "tangent vectors" in 3D, the volume of their
         parallelpiped, which is the differential volume
         of the coordinate field */
      fail("getJacobianDeterminant doesn't yet support 3d.  "
           "The given matrices have dimension %lu x %lu \n",
           J.extent(0), J.extent(1));
    }
    if (dimension == 2) {
      if (J.extent(1) != 2) {
        fail("getJacobianDeterminant only supports 2x2 matrices in 2d.  "
             "The given matrices have dimension %lu x %lu \n",
             J.extent(1), J.extent(2));
      }
      /* |\frac{\partial x}{\partial s}\times
         \frac{\partial x}{\partial t}|,
         the area spanned by the tangent vectors
         at this point, surface integral. */
      Kokkos::View<Real *> determinants("2dJacobianDeterminants", J.extent(0));
      // compute the cross product of the 2x2 jacobian matrix
      Kokkos::parallel_for(
          J.extent(0), KOKKOS_LAMBDA(const int i) {
            // TODO use nested parallel for?
            auto Ji = Kokkos::subview(J, i, Kokkos::ALL(), Kokkos::ALL());
            const auto cross = Ji(0, 0) * Ji(1, 1) - Ji(1, 0) * Ji(0, 1);
            const auto magnitude = Kokkos::fabs(cross);
            determinants(i) = magnitude;
          });
      return determinants;
    }
    if (dimension == 1) {
      fail("getJacobianDeterminant doesn't yet support 1d.  "
           "The given matrices have dimension %lu x %lu \n",
           J.extent(0), J.extent(1));
    }
    // assuming at this point dimension=1
    /* \|\vec{x}_{,\xi}\| the length
       of the tangent vector at this point.
       line integral:
       ds = sqrt(dx^2 + dy^2 + dz^2) */
    return Kokkos::View<Real *>("foo", J.extent(0));
  }

  /**
   * @brief
   * Given an array of parametric coordinates 'localCoords', one per mesh
   * element, compute the jacobian within each element.
   *
   * @todo add static asserts for values and functions provided by the templated
   * types
   * @todo consider making this a member function of FieldElement
   *
   * @param localCoords 2D Kokkos::View containing the local/parametric/area
   * coordinates for each element
   * @param offsets 1D Kokkos::View containing the offsets into localCoords,
   *                size = localCoords.extent(0)+1
   * @return Kokkos::View containing the jacobian for all the mesh elements
   */
  Kokkos::View<Real ***> getJacobians(Kokkos::View<Real **> localCoords,
                                      Kokkos::View<LO *> offsets) {
    if (Debug) {
      // check input parametric coords are positive and sum to one
      LO numErrors = 0;
      Kokkos::parallel_reduce(
          "checkCoords", numMeshEnts,
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
    if (localCoords.extent(0) < numMeshEnts) {
      fail("The size of dimension 0 of the local coordinates input array "
           "must be at least %zu.\n",
           numMeshEnts);
    }
    if (localCoords.extent(1) != MeshEntDim + 1) {
      fail("Dimension 1 of the input array of local coordinates "
           "must have size = %zu.\n",
           MeshEntDim + 1);
    }
    if (offsets.size() != numMeshEnts + 1) {
      fail("The input array of offsets must have size = %zu\n",
           numMeshEnts + 1);
    }
    if (MeshEntDim != 1 && MeshEntDim != 2) {
      fail("getJacobians only currently supports 1d and 2d meshes.  Input mesh "
           "has %zu dimensions.\n",
           numMeshEnts);
    }
    if constexpr (MeshEntDim == 1) {
      const auto numPts = MeshFieldUtil::getLastValue(offsets);
      Kokkos::View<Real ***> res("result", numPts, 1, 1);
      Kokkos::parallel_for(
          numMeshEnts, KOKKOS_CLASS_LAMBDA(const int ent) {
            // TODO use nested parallel for?
            for (auto pt = offsets(ent); pt < offsets(ent + 1); pt++) {
              const auto val = getJacobian1d(ent);
              res(pt, 0, 0) = val;
            }
          });
      return res;
    } else if constexpr (MeshEntDim == 2) {
      const auto numPts = MeshFieldUtil::getLastValue(offsets);
      // one matrix per point
      Kokkos::View<Real ***> res("result", numPts, MeshEntDim, MeshEntDim);
      Kokkos::deep_copy(res, 0.0); // initialize all entries to zero

      // fill the views of node coordinates and node gradients
      Kokkos::View<Real * [ShapeType::numNodes][MeshEntDim]> nodeCoords(
          "nodeCoords", numPts);
      Kokkos::View<Real * [ShapeType::numNodes][MeshEntDim]> nodalGradients(
          "nodalGradients", numPts);
      const auto grad = shapeFn.getLocalGradients();
      Kokkos::parallel_for(
          numMeshEnts, KOKKOS_CLASS_LAMBDA(const int ent) {
            const auto vals = getNodeValues(ent);
            assert(vals.size() == MeshEntDim * ShapeType::numNodes);
            for (auto pt = offsets(ent); pt < offsets(ent + 1); pt++) {
              for (size_t node = 0; node < ShapeType::numNodes; node++) {
                for (size_t d = 0; d < MeshEntDim; d++) {
                  nodeCoords(pt, node, d) = vals[node * MeshEntDim + d];
                  nodalGradients(pt, node, d) = grad[node * MeshEntDim + d];
                }
              }
            }
          });

      Kokkos::parallel_for(
          numMeshEnts, KOKKOS_LAMBDA(const int ent) {
            // TODO use nested parallel for?
            for (auto pt = offsets(ent); pt < offsets(ent + 1); pt++) {
              auto A = Kokkos::subview(res, pt, Kokkos::ALL(), Kokkos::ALL());
              for (size_t node = 0; node < ShapeType::numNodes; node++) {
                auto a =
                    Kokkos::subview(nodalGradients, pt, node, Kokkos::ALL());
                auto b = Kokkos::subview(nodeCoords, pt, node, Kokkos::ALL());
                addTensorProduct(a, b, A);
              }
            }
          });
      return res;
    }
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

  if (localCoords.extent(1) != fes.MeshEntDim + 1) {
    fail("Dimension 1 of the input array of local coordinates "
         "must have size = %zu.\n",
         fes.MeshEntDim + 1);
  }
  if (offsets.size() != fes.numMeshEnts + 1) {
    fail("The input array of offsets must have size = %zu\n",
         fes.numMeshEnts + 1);
  }
  LO numLocalCoords;
  Kokkos::deep_copy(numLocalCoords,
                    Kokkos::subview(offsets, offsets.size() - 1));
  if (localCoords.extent(0) != numLocalCoords) {
    fail("The size of dimension 0 of the local coordinates input array (%zu) "
         "does not match the last entry of the offsets array (%zu).\n",
         localCoords.extent(0), numLocalCoords);
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
  const auto numPtsPerElement = 1;
  return evaluate(fes, localCoords, numPtsPerElement);
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
  const auto offsets = getOffsets(fes.numMeshEnts, numPtsPerElement);
  return evaluate(fes, localCoords, offsets);
}

/**
 * @brief
 * Given an array of parametric coordinates 'localCoords',
 * with numPtsPerElement points per mesh element,
 * compute the jacobian for each element at the specified point
 *
 * @param (in) numPtsPerElement
 *
 * @return an array of rank2 matrices stored in a Kokkos 3d View.  The dimension
 * are sized (from left to right):
 *  - number of points where the jacobian is evaluated
 *  - number of rows in each rank2 matrix
 *  - number of columns in each rank2 matrix
 */
template <typename FieldElement>
Kokkos::View<Real ***> getJacobians(FieldElement &fes,
                                    Kokkos::View<Real **> localCoords,
                                    size_t numPtsPerElement) {
  const auto offsets = getOffsets(fes.numMeshEnts, numPtsPerElement);
  return fes.getJacobians(localCoords, offsets);
}

/**
 * @brief
 * Given an array of Jacobian matrices
 * compute the determinant for each
 *
 * @param (in) array of Jacobian matrices
 *
 * @return an array of scalars that are the jacobian determinant for each
 * input Jacobian matrix
 */
template <typename FieldElement, typename Matrices>
Kokkos::View<Real *> getJacobianDeterminants(FieldElement &fes, Matrices J) {
  return fes.getJacobianDeterminants(J);
}

} // namespace MeshField
#endif
