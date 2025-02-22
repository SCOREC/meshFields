
#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Integrate.hpp>

namespace MeshField {

void getIntPoint(MeshElement* e, int order, int point, Vector3& param)
{
  IntegrationPoint const* p = 
    getIntegration(e->getType())->getAccurate(order)->getPoint(point);
  param = p->param;
}

double getIntWeight(MeshElement* e, int order, int point)
{
  IntegrationPoint const* p = 
    getIntegration(e->getType())->getAccurate(order)->getPoint(point);
  return p->weight;
}

double getDV(MeshElement* e, Vector3 const& param)
{
  return e->getDV(param); //FIXME return determinant of Jacobian - for linear triangles the paramteric coord ('param') is ignored
}

Integrator::Integrator(int o):
  order(o),
  ipnode(0)
{
}

Integrator::~Integrator()
{
}

void Integrator::inElement(MeshElement*)
{
}

void Integrator::outElement()
{
}

void Integrator::parallelReduce(pcu::PCU*)
{
}

void Integrator::process(Mesh* m, int d) //FIXME needs to take FieldElement
{
  if(d<0)
    d = m->getDimension();
  PCU_DEBUG_ASSERT(d<=m->getDimension());
  MeshEntity* entity;
  MeshIterator* elements = m->begin(d);
  while ((entity = m->iterate(elements)))
  {
    if ( ! m->isOwned(entity)) continue;
    MeshElement* e = createMeshElement(m,entity);
    this->process(e);
    destroyMeshElement(e);
  }
  m->end(elements);
  this->parallelReduce(m->getPCU());
}

void Integrator::process(MeshElement* e) //FIXME move this into process? hard to see how this will work as a user function
{
  this->inElement(e);
  int np = countIntPoints(e,this->order);
  for (int p=0; p < np; ++p)
  {
    ipnode = p;
    Vector3 point;
    getIntPoint(e,this->order,p,point); //IntegrationPoint[p].param
    double w = getIntWeight(e,this->order,p); //IntegrationPoint[p].weight
    double dV = getDV(e,point); //determinant of jacobian
    this->atPoint(point,w,dV); //callback/functor
  }
  this->outElement();
}

} // namespace MeshField
