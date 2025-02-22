#ifndef MESHFIELD_INTEGRATE_H
#define MESHFIELD_INTEGRATE_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>
#include <type_traits> // has_static_size helper

namespace MeshField {
  /** \brief A virtual base for user-defined integrators.
   * directly copied from SCOREC/core @ 7cd76473 apf/apf.h
   *
   * \details Users of APF can define an Integrator object to handle
   * integrating expressions over elements and over meshes.
   * Users specify the accuracy of the integration and provide
   * accumulation callbacks which APF uses at each integration
   * point. The APF-provided process functions will perform the
   * integration over an element or mesh using the callbacks.
   * In parallel, users must provide a reduction callback
   * to turn locally accumulated values into a globally integrated
   * value.
   */
  class Integrator
  {
    public:
      /** \brief Construct an Integrator given an order of accuracy. */
      Integrator(int o);
      virtual ~Integrator();
      /** \brief Run the Integrator over the local Mesh.
       * \param m mesh to integrate over
       * \param dim optional dimension to integrate over. This defaults to
       * integration over the mesh dimesion which may not be correct e.g. in the case
       * of a 1D element embeded in 3D space.
       * */
      void process(Mesh* m, int dim=-1);
      /** \brief Run the Integrator over a Mesh Element. */
      void process(MeshElement* e);
      /** \brief User callback: element entry.
       *
       * \details APF will call this function every time the
       * Integrator begins operating over a new element.
       * Users can then construct Field Elements, for example.
       */
      virtual void inElement(MeshElement*);
      /** \brief User callback: element exit.
       *
       * \details APF will call this function once an Integrator
       * is done operating over an element. This can be used
       * to destroy Field Elements, for example.
       */
      virtual void outElement();
      /** \brief User callback: accumulation.
       *
       * \details APF will call this function at each integration
       * point. Users should evaluate their expression and accumulate
       * the value.
       *
       * \param p The local coordinates of the point.
       * \param w The integration weight of the point.
       * \param dV The differential volume at that point.
       */
      virtual void atPoint(Vector3 const& p, double w, double dV) = 0;
      /** \brief User callback: parallel reduction.
       *
       * \details This function should use communication to reduce
       * process-local integrations into a global mesh integration,
       * if that is the user's goal.
       */
      virtual void parallelReduce(pcu::PCU*);
    protected:
      int order;
      int ipnode;
  };

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
      virtual std::vector<IntegrationPoint> getPoint() const = 0;
      virtual int getAccuracy() const = 0;
  };

  class EntityIntegration
  {
    public:
      virtual ~EntityIntegration() {}
      Integration const* getAccurate(int minimumAccuracy) const;
      virtual int countIntegrations() const = 0;
      virtual Integration const* getIntegration(int i) const = 0;
  };

  EntityIntegration const* getIntegration(int meshEntityType);

  class TriangleIntegration : public EntityIntegration
  {
    public:
      class N1 : public Integration {
        public:
          virtual int countPoints() const {return 1;}
          virtual std::vector<IntegrationPoint> getPoint() const
          {
            return {IntegrationPoint point(Vector3(1./3.,1./3.,0),1.0/2.0)};
          }
          virtual int getAccuracy() const {return 1;}
      }; //end N1
      class N2 : public Integration {
        public:
          virtual int countPoints() const {return 3;}
          virtual std::vector<IntegrationPoint> getPoints() const
          {
            return { IntegrationPoint(Vector3(0.666666666666667,0.166666666666667,0),1./3./2.0),
                     IntegrationPoint(Vector3(0.166666666666667,0.666666666666667,0),1./3./2.0),
                     IntegrationPoint(Vector3(0.166666666666667,0.166666666666667,0),1./3./2.0) };
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
} // namespace MeshField

