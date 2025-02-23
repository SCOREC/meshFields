#ifndef MESHFIELD_INTEGRATE_H
#define MESHFIELD_INTEGRATE_H

#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>
#include <type_traits> // has_static_size helper

namespace MeshField { //FIXME move some of the helper funcs to anonymous namespace
  // directly copied from SCOREC/core @ 7cd76473 apf/apfIntegrate.[h|cc]
  struct IntegrationPoint
  {
    IntegrationPoint(Vector3 const& p, double w):
      param(p),weight(w)
    {}
    Vector3 param;
    double weight;
  };

  class Integration
  {
    public:
      virtual ~Integration() {}
      virtual int countPoints() const = 0;
      virtual std::vector<IntegrationPoint> getPoints() const = 0;
      virtual int getAccuracy() const = 0;
  };

  class EntityIntegration
  {
    public:
      virtual ~EntityIntegration() {}
      Integration const* getAccurate(int minimumAccuracy) const {
        int n = countIntegrations();
        for (int i=0; i < n; ++i) {
          Integration const* integration = getIntegration(i);
          if (integration->getAccuracy() >= minimumAccuracy)
            return integration;
        }
        return NULL;
      }
      virtual int countIntegrations() const = 0;
      virtual Integration const* getIntegration(int i) const = 0;
  };

  class TriangleIntegration : public EntityIntegration
  {
    public:
      class N1 : public Integration {
        public:
          virtual int countPoints() const {return 1;}
          virtual std::vector<IntegrationPoint> getPoints() const
          {
            return {IntegrationPoint(Vector3{1./3.,1./3.,0},1.0/2.0)};
          }
          virtual int getAccuracy() const {return 1;}
      }; //end N1
      class N2 : public Integration {
        public:
          virtual int countPoints() const {return 3;}
          virtual std::vector<IntegrationPoint> getPoints() const
          {
            return { IntegrationPoint(Vector3{0.666666666666667,0.166666666666667,0},1./3./2.0),
                     IntegrationPoint(Vector3{0.166666666666667,0.666666666666667,0},1./3./2.0),
                     IntegrationPoint(Vector3{0.166666666666667,0.166666666666667,0},1./3./2.0) };
          }
          virtual int getAccuracy() const {return 2;}
      }; //end N2
      virtual int countIntegrations() const {return 2;}
      virtual Integration const* getIntegration(int i) const
      {
        static N1 i1;
        static N2 i2;
        static Integration* integrations[2] = {&i1,&i2};
        return integrations[i];
      }
  };

  EntityIntegration const* getIntegration(Mesh_Topology topo)
  {
    if(topo != Triangle) {
      fail("getIntegration only supports triangles (Mesh_Topology::Triangle)\n");
    }
    return new TriangleIntegration();
  }

  std::vector<IntegrationPoint> getIntegrationPoints(Mesh_Topology topo, int order) {
    auto ip = getIntegration(topo)->getAccurate(order)->getPoints();
    return ip;
  }

  template <typename FieldElement>
  auto getIntegrationPointLocalCoords(FieldElement& fes, std::vector<IntegrationPoint> ip) {
    const auto numPtsPerElm = ip.size();
    const auto numMeshEnts = fes.numMeshEnts;
    const auto meshEntDim = fes.MeshEntDim;
    const auto numParametricCoords = meshEntDim+1;
    Kokkos::View<MeshField::Real **> localCoords("localCoords", numMeshEnts*numPtsPerElm, numParametricCoords);
    //broadcast the points into the view - FIXME this is an inefficient use of memory
    for(size_t pt=0; pt < numPtsPerElm; pt++) {
      const auto point = ip.at(pt);
      const auto param = point.param;
      Kokkos::parallel_for(numMeshEnts, KOKKOS_LAMBDA(const int ent) {
          for(size_t i=0; i<param.size(); i++) {
            for(size_t d=0; d<numParametricCoords; d++) {
              localCoords(ent*numPtsPerElm+pt,d) = param[d];
            }
          }
      });
    }
    return localCoords;
  }

  template <typename FieldElement>
  auto getIntegrationPointWeights(FieldElement& fes, std::vector<IntegrationPoint> ip) {
    const auto numPtsPerElm = ip.size();
    const auto numMeshEnts = fes.numMeshEnts;
    const auto meshEntDim = fes.MeshEntDim;
    Kokkos::View<Real *> weights("weights", numMeshEnts*numPtsPerElm);
    //broadcast the points into the view - FIXME this is an inefficient use of memory
    for(size_t pt=0; pt < numPtsPerElm; pt++) {
      const auto point = ip.at(pt);
      const auto w = point.weight;
      Kokkos::parallel_for(numMeshEnts, KOKKOS_LAMBDA(const int ent) {
          weights(ent*numPtsPerElm+pt) = w;
      });
    }
    return weights;
  }

  template <typename FieldElement>
  auto getJacobianDeterminants(FieldElement& fes, Kokkos::View<Real **> localCoords,
      size_t numIntegrationPoints) {
    auto J = getJacobians(fes, localCoords, numIntegrationPoints);
    auto dV = getJacobianDeterminants(fes, J); //FIXME fails here
    return dV;
  }

  /** \brief A virtual base for user-defined integrators.
   * directly copied from SCOREC/core @ 7cd76473 apf/apf.h
   *
   * FIXME add documentation
   */
  class Integrator
  {
    public:
      /** \brief Construct an Integrator given an order of accuracy. */
      Integrator(int o): order(o) {}
      virtual ~Integrator() {};
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
      virtual void atPoints(Kokkos::View<Real**> p, Kokkos::View<Real*> w, Kokkos::View<Real*> dV) = 0;
      /** \brief Run the Integrator over the local field elements.
       * \param fes FieldElement
       * FIXME make the sensible
       * */
      template <typename FieldElement>
      void process(FieldElement &fes)
      {
        //TODO add check to only support triangles
        pre();
        const auto topo = fes.elm2dof.getTopology();
        auto ip = getIntegrationPoints(topo[0],order);
        auto localCoords = getIntegrationPointLocalCoords(fes,ip);
        auto weights = getIntegrationPointWeights(fes,ip);
        auto dV = getJacobianDeterminants(fes,localCoords,ip.size());
        atPoints(localCoords,weights,dV);
        post();
        //TODO support distributed meshes by running a parallel reduction with user functor
      }
    protected:
      int order;
  };

} // namespace MeshField

#endif
