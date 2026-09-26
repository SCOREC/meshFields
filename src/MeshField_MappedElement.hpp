#ifndef MESHFIELD_MAPPEDELEMENT_HPP
#define MESHFIELD_MAPPEDELEMENT_HPP

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Element.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_LinearAlgebra.hpp>
#include <utility> // std::declval

/**
 * @file MeshField_MappedElement.hpp
 * @brief Elements whose geometry, dofs, or values need mapping between the
 * reference element and the physical element.
 *
 * A MappedElement composes three things:
 *
 *  - **a geometry element**: a FieldElement over the coordinate field that
 *    supplies the map from the reference element to the physical element.  
 *  - **a DofTransformation**: converts the dofs stored on mesh entities to
 *    coefficients on the reference basis. See:
 *    R.C. Kirby, "A general approach to transforming finite
 *    elements", SMAI J. Comput. Math. 4 (2018) 197-224 , and https://defelement.org/
 *  - **a MapType**: converts values and derivatives on the reference element
 *    to the physical element.  See: https://defelement.org
 */

namespace MeshField {

struct IdentityDofTransformation {
  template <typename FieldElem, typename GeomElem>
  KOKKOS_INLINE_FUNCTION auto coefficients(FieldElem const &field,
                                           GeomElem const &, int ent) const {
    return field.coefficients(ent);
  }
};

struct IdentityMap {
  template <typename ValArray>
  KOKKOS_INLINE_FUNCTION ValArray value(ValArray const &ref) const {
    return ref;
  }

  /**
   * @brief computes the physical gradient
   *
   * @details
   * With J[i][j] = dx_j/dxi_i the chain rule gives grad_xi = J grad_x, so
   * grad_x = J^{-1} grad_xi 
   */
  template <typename GradArray, typename JacobianArray>
  KOKKOS_INLINE_FUNCTION GradArray gradient(GradArray const &refGrad,
                                            JacobianArray const &J) const {
    // sqrt not constexpr until C++26. Is there a more sane way to do this?
    constexpr size_t Dim = JacobianArray::size() == 1   ? 1
                           : JacobianArray::size() == 4 ? 2
                                                        : 3;
    static_assert(Dim * Dim == JacobianArray::size());
    constexpr auto numComp = GradArray::size() / Dim;
    const auto Jinv = invert<Dim>(J);
    GradArray g;
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t ci = 0; ci < numComp; ++ci) {
        Real sum = 0;
        for (size_t j = 0; j < Dim; ++j)
          sum += Jinv[i * Dim + j] * refGrad[j * numComp + ci];
        g[i * numComp + ci] = sum;
      }
    }
    return g;
  }
};

/**
 * @brief Element type used for non-isoparametric, and other elements that require mapping
 *
 * @details
 * The field element supplies the basis and the stored dofs.  The geometry
 * element supplies the map from the reference element to the physical
 * element.  I
 *
 * Evaluation:
 * @code
 * const auto c = elm.coefficients(ent);              // once per element
 * for (pt : points) elm.valueFromCoefficients(c, xi); // once per point
 * @endcode
 *
 * @tparam FieldElem a FieldElement over the field being evaluated
 * @tparam GeomElem a FieldElement over the coordinate field
 * @tparam DofTransformation see IdentityDofTransformation
 * @tparam MapType see IdentityMap
 */
template <typename FieldElem, typename GeomElem,
          typename DofTransformation = IdentityDofTransformation,
          typename MapType = IdentityMap>
struct MappedElement {
  static_assert(FieldElem::MeshEntDim == GeomElem::MeshEntDim,
                "the field and geometry elements must have the same "
                "parametric dimension");
  static_assert(FieldElem::getTopology()[0] == GeomElem::getTopology()[0],
                "the field and geometry elements must have the same topology");

  FieldElem field;
  GeomElem geom;
  DofTransformation dofTransformation;
  MapType mapType;
  const size_t numMeshEnts;

  static constexpr size_t MeshEntDim = FieldElem::MeshEntDim;
  static constexpr size_t NumComponents = FieldElem::NumComponents;
  using ValArray = typename FieldElem::ValArray;

  /** @brief the topology of the reference element */
  static constexpr KOKKOS_FUNCTION auto getTopology() {
    return GeomElem::getTopology();
  }

  /** @brief physical gradients 
   *  [direction * NumComponents + component] */
  using GradArray = typename FieldElem::LocalGradArray;
  /** @brief the Jacobian, J[i][j] = dx_j/dxi_i */
  using JacobianArray = typename GeomElem::JacobianArray;

  /** @brief storage for per-element data*/
  struct Coefficients {
    /** coefficients on the reference basis, from the DofTransformation */
    decltype(std::declval<DofTransformation>().coefficients(
        std::declval<FieldElem>(), std::declval<GeomElem>(), 0)) field;
    /** the geometry element's node coordinates */
    typename GeomElem::NodeArray geometry;
  };

  MappedElement(
      FieldElem const &fieldIn, GeomElem const &geomIn,
      DofTransformation const &dofTransformationIn = DofTransformation(),
      MapType const &mapTypeIn = MapType())
      : field(fieldIn), geom(geomIn), dofTransformation(dofTransformationIn),
        mapType(mapTypeIn), numMeshEnts(fieldIn.numMeshEnts) {
    if (geomIn.numMeshEnts != fieldIn.numMeshEnts) {
      fail("the field and geometry elements have the same number of "
           "mesh entities (%zu != %zu)\n",
           fieldIn.numMeshEnts, geomIn.numMeshEnts);
    }
  }

  KOKKOS_INLINE_FUNCTION Coefficients coefficients(int ent) const {
    return {dofTransformation.coefficients(field, geom, ent),
            geom.getNodeValues(ent)};
  }

  /** @brief the physical value of the field at a parametric coordinate */
  KOKKOS_INLINE_FUNCTION ValArray valueFromCoefficients(
      Coefficients const &c, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return mapType.value(field.valueFromCoefficients(c.field, localCoord));
  }

  /** @brief the Jacobian of the geometry at a parametric coordinate */
  KOKKOS_INLINE_FUNCTION JacobianArray jacobianFromCoefficients(
      Coefficients const &c, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return geom.jacobianFromNodeValues(c.geometry, localCoord);
  }

  /** @brief the physical gradient of the field at a parametric coordinate */
  KOKKOS_INLINE_FUNCTION GradArray gradientFromCoefficients(
      Coefficients const &c, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return mapType.gradient(
        field.localGradientFromCoefficients(c.field, localCoord),
        jacobianFromCoefficients(c, localCoord));
  }

  /**
   * @brief evaluate the field at a single parametric coordinate
   *
   * Prefer coefficients and valueFromCoefficients when evaluating an element
   * at more than one point.
   */
  KOKKOS_INLINE_FUNCTION ValArray
  getValue(int ent, Kokkos::Array<Real, MeshEntDim> localCoord) const {
    return valueFromCoefficients(coefficients(ent), localCoord);
  }

  /** @brief Jacobians, from the geometry element */
  Kokkos::View<Real ***> getJacobians(Kokkos::View<Real **> localCoords,
                                      Kokkos::View<LO *> offsets) const {
    return geom.getJacobians(localCoords, offsets);
  }

  /** @brief Jacobian determinants, from the geometry element */
  template <typename Matrices>
  Kokkos::View<Real *> getJacobianDeterminants(Matrices const &J) const {
    return geom.getJacobianDeterminants(J);
  }
};

} // namespace MeshField

#endif
