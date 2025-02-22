#ifndef MESHFIELD_ELEMENT_H
#define MESHFIELD_ELEMENT_H

#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Utility.hpp> // getLastValue
#include <iostream>
#include <type_traits> // has_static_size helper

namespace MeshField {
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
      virtual IntegrationPoint const* getPoint(int i) const = 0;
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
      class N1 : public Integration
    {
      public:
        virtual int countPoints() const {return 1;}
        virtual IntegrationPoint const* getPoint(int) const
        {
          static IntegrationPoint point(
              Vector3(1./3.,1./3.,0),1.0/2.0);
          return &point;
        }
        virtual int getAccuracy() const {return 1;}
    };
      class N2 : public Integration
    {
      public:
        virtual int countPoints() const {return 3;}
        virtual IntegrationPoint const* getPoint(int i) const
        {
          static IntegrationPoint points[3]=
          { IntegrationPoint(Vector3(0.666666666666667,0.166666666666667,0),1./3./2.0),
            IntegrationPoint(Vector3(0.166666666666667,0.666666666666667,0),1./3./2.0),
            IntegrationPoint(Vector3(0.166666666666667,0.166666666666667,0),1./3./2.0), };
          return points+i;
        }
        virtual int getAccuracy() const {return 2;}
    };
  };
} // namespace MeshField

