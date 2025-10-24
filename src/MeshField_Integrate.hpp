#ifndef MESHFIELD_INTEGRATE_H
#define MESHFIELD_INTEGRATE_H

#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>
#include <type_traits> // has_static_size helper

namespace MeshField {
// directly copied from SCOREC/core @ 7cd76473 apf/apfIntegrate.[h|cc]
template <size_t pointSize> struct IntegrationPoint {
  // template parameter pointSize specifies the length of the integration point
  // array for one point
  IntegrationPoint(Kokkos::Array<Real, pointSize> const &p, double w)
      : param(p), weight(w) {}
  Kokkos::Array<Real, pointSize> param;
  double weight;
};
template <size_t pointSize> class Integration {
public:
  virtual ~Integration() {}
  virtual int countPoints() const = 0;
  virtual std::vector<IntegrationPoint<pointSize>> getPoints() const = 0;
  virtual int getAccuracy() const = 0;
};
template <size_t pointSize> class EntityIntegration {
public:
  virtual ~EntityIntegration() {}
  Integration<pointSize> const *getAccurate(int minimumAccuracy) const {
    int n = countIntegrations();
    for (int i = 0; i < n; ++i) {
      Integration<pointSize> const *integration = getIntegration(i);
      if (integration->getAccuracy() >= minimumAccuracy)
        return integration;
    }
    return NULL;
  }
  virtual int countIntegrations() const = 0;
  virtual Integration<pointSize> const *getIntegration(int i) const = 0;
};

class TriangleIntegration : public EntityIntegration<3> {
public:
  class N1 : public Integration<3> {
  public:
    virtual int countPoints() const { return 1; }
    virtual std::vector<IntegrationPoint<3>> getPoints() const {
      return {IntegrationPoint(Vector3{1. / 3., 1. / 3., 1. / 3.}, 1.0 / 2.0)};
    }
    virtual int getAccuracy() const { return 1; }
  }; // end N1
  class N2 : public Integration<3> {
  public:
    virtual int countPoints() const { return 3; }
    virtual std::vector<IntegrationPoint<3>> getPoints() const {
      return {IntegrationPoint(Vector3{0.666666666666667, 0.166666666666667,
                                       0.166666666666667},
                               1. / 3. / 2.0),
              IntegrationPoint(Vector3{0.166666666666667, 0.666666666666667,
                                       0.166666666666667},
                               1. / 3. / 2.0),
              IntegrationPoint(Vector3{0.166666666666667, 0.166666666666667,
                                       0.666666666666667},
                               1. / 3. / 2.0)};
    }
    virtual int getAccuracy() const { return 2; }
  }; // end N2
    class N6 : public Integration<3> {
  public:
    virtual int countPoints() const { return 12; }
    virtual std::vector<IntegrationPoint<3>> getPoints() const {
      return {
          IntegrationPoint(Vector3{0.873821971, 0.063089014, 0.063089014},
                           0.0508449063702 / 2.0),
          IntegrationPoint(Vector3{0.063089014, 0.873821971, 0.063089014},
                           0.0508449063702 / 2.0),
          IntegrationPoint(Vector3{0.063089014, 0.063089014, 0.873821971},
                           0.0508449063702 / 2.0),

          IntegrationPoint(Vector3{0.501426509, 0.249286745, 0.249286745},
                           0.1167862757264 / 2.0),
          IntegrationPoint(Vector3{0.249286745, 0.501426509, 0.249286745},
                           0.1167862757264 / 2.0),
          IntegrationPoint(Vector3{0.249286745, 0.249286745, 0.501426509},
                           0.1167862757264 / 2.0),

          IntegrationPoint(Vector3{0.636502499, 0.310352451, 0.053145050},
                           0.0828510756184 / 2.0),
          IntegrationPoint(Vector3{0.636502499, 0.053145050, 0.310352451},
                           0.0828510756184 / 2.0),
          IntegrationPoint(Vector3{0.310352451, 0.636502499, 0.053145050},
                           0.0828510756184 / 2.0),
          IntegrationPoint(Vector3{0.310352451, 0.053145050, 0.636502499},
                           0.0828510756184 / 2.0),
          IntegrationPoint(Vector3{0.053145050, 0.636502499, 0.310352451},
                           0.0828510756184 / 2.0),
          IntegrationPoint(Vector3{0.053145050, 0.310352451, 0.636502499},
                           0.0828510756184 / 2.0)};
    }
    virtual int getAccuracy() const { return 6; }
  }; // end N6
  virtual int countIntegrations() const { return 3; }
  virtual Integration<3> const *getIntegration(int i) const {
    static N1 i1;
    static N2 i2;
    static N6 i6;
    static Integration<3> *integrations[3] = {&i1, &i2, &i6};
    return integrations[i];
  }
};

class TetrahedronIntegration : public EntityIntegration<4> {
public:
  class N1 : public Integration<4> {
  public:
    virtual int countPoints() const { return 1; }
    virtual std::vector<IntegrationPoint<4>> getPoints() const {
      return {IntegrationPoint(Vector4{0.25, 0.25, 0.25, 0.25}, 1.0 / 6.0)};
    }
    virtual int getAccuracy() const { return 1; }
  };
  class N2 : public Integration<4> {
  public:
    virtual int countPoints() const { return 4; }
    virtual std::vector<IntegrationPoint<4>> getPoints() const {

      return {IntegrationPoint(Vector4{0.138196601125011, 0.138196601125011,
                                       0.138196601125011, 0.585410196624969},
                               0.25 / 6.0),
              IntegrationPoint(Vector4{0.585410196624969, 0.138196601125011,
                                       0.138196601125011, 0.138196601125011},
                               0.25 / 6.0),
              IntegrationPoint(Vector4{0.138196601125011, 0.585410196624969,
                                       0.138196601125011, 0.138196601125011},
                               0.25 / 6.0),
              IntegrationPoint(Vector4{0.138196601125011, 0.138196601125011,
                                       0.585410196624969, 0.138196601125011},
                               0.25 / 6.0)};
    }
    virtual int getAccuracy() const { return 2; }
  };
  virtual int countIntegrations() const { return 2; }
  virtual Integration<4> const *getIntegration(int i) const {
    static N1 i1;
    static N2 i2;
    static Integration<4> *integrations[2] = {&i1, &i2};
    return integrations[i];
  }
};
template <Mesh_Topology topo> auto const getIntegration() {
  if constexpr (topo == Triangle) {
    return std::make_shared<TriangleIntegration>();
  } else if constexpr (topo == Tetrahedron) {
    return std::make_shared<TetrahedronIntegration>();
  }
  fail("getIntegration does not support given topology\n");
}
template <Mesh_Topology topo> auto getIntegrationPoints(int order) {
  auto ip = getIntegration<topo>()->getAccurate(order)->getPoints();
  return ip;
}

template <typename FieldElement>
Kokkos::View<MeshField::Real **> getIntegrationPointLocalCoords(
    FieldElement &fes,
    std::vector<IntegrationPoint<FieldElement::MeshEntDim + 1>> ip) {
  const auto numPtsPerElm = ip.size();
  const auto numMeshEnts = fes.numMeshEnts;
  const auto meshEntDim = fes.MeshEntDim;
  const auto numParametricCoords = meshEntDim + 1;
  Kokkos::View<MeshField::Real **> localCoords(
      "localCoords", numMeshEnts * numPtsPerElm, numParametricCoords);
  // broadcast the points into the view - FIXME this is an inefficient use of
  // memory
  for (size_t pt = 0; pt < numPtsPerElm; pt++) {
    const auto point = ip.at(pt);
    const auto param = point.param;
    assert(numParametricCoords == param.size());
    Kokkos::parallel_for(
        numMeshEnts, KOKKOS_LAMBDA(const int ent) {
          for (size_t d = 0; d < numParametricCoords; d++) {
            localCoords(ent * numPtsPerElm + pt, d) = param[d];
          }
        });
  }
  return localCoords;
}

template <typename FieldElement>
Kokkos::View<Real *> getIntegrationPointWeights(
    FieldElement &fes,
    std::vector<IntegrationPoint<FieldElement::MeshEntDim + 1>> ip) {
  const auto numPtsPerElm = ip.size();
  const auto numMeshEnts = fes.numMeshEnts;
  const auto meshEntDim = fes.MeshEntDim;
  Kokkos::View<Real *> weights("weights", numMeshEnts * numPtsPerElm);
  // broadcast the points into the view - FIXME this is an inefficient use of
  // memory
  for (size_t pt = 0; pt < numPtsPerElm; pt++) {
    const auto point = ip.at(pt);
    const auto w = point.weight;
    Kokkos::parallel_for(
        numMeshEnts,
        KOKKOS_LAMBDA(const int ent) { weights(ent * numPtsPerElm + pt) = w; });
  }
  return weights;
}

template <typename FieldElement>
auto getJacobianDeterminants(FieldElement &fes,
                             Kokkos::View<Real **> localCoords,
                             size_t numIntegrationPoints) {
  auto J = getJacobians(fes, localCoords, numIntegrationPoints);
  auto dV = getJacobianDeterminants(fes, J);
  return dV;
}

/** \brief A virtual base for user-defined integrators.
 *
 * The `Integrator` class provides an interface for implementing custom
 * integration routines over elements. Users can define specific behavior
 * for integration points, weights, and differential volumes.
 *
 * This class is directly adapted from SCOREC/core @ 7cd76473 apf/apf.h.
 *
 * \details
 * The `Integrator` class operates on elements and provides hooks
 * for user-defined callbacks:
 * - `pre`: Called before the integration process begins.
 * - `post`: Called after the integration process ends.
 * - `atPoints`: Called for each integration point to perform user-defined
 *   computations.
 *
 * To use this class, derive from it and implement the `atPoints` method.
 * Optionally, override `pre` and `post` for additional setup or cleanup.
 */
class Integrator {
public:
  /** \brief Construct an Integrator given an order of accuracy. */
  Integrator(int o) : order(o) {}
  virtual ~Integrator(){};
  /** \brief User callback: process entry
   *
   * \details do something at the start of the process call
   */
  virtual void pre() {}
  /** \brief User callback: process exit
   *
   * \details do something at the end of the process call
   */
  virtual void post() {}
  /** \brief User callback: accumulation.
   *
   * \details called for all integration points
   * Users should evaluate their expression and accumulate
   * the value.
   *
   * \param p The local coordinates of integration points
   * \param w The integration weight of the points.
   * \param dV The differential volume at that points.
   */
  virtual void atPoints(Kokkos::View<Real **> p, Kokkos::View<Real *> w,
                        Kokkos::View<Real *> dV) = 0;
  /** \brief Run the Integrator over the local field elements.
   * \param fes FieldElement
   * FIXME make the sensible
   * */
  template <typename FieldElement> void process(FieldElement &fes) {
    constexpr auto topo = decltype(FieldElement::elm2dof)::getTopology();
    pre();
    auto ip = getIntegrationPoints<topo[0]>(order);
    auto localCoords = getIntegrationPointLocalCoords(fes, ip);
    auto weights = getIntegrationPointWeights(fes, ip);
    auto dV = getJacobianDeterminants(fes, localCoords, ip.size());
    atPoints(localCoords, weights, dV);
    post();
    // TODO support distributed meshes by running a parallel reduction with user
    // functor
  }

protected:
  int order;
};

} // namespace MeshField

#endif
