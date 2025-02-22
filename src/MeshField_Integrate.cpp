
#include <Kokkos_Core.hpp>
#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <MeshField_Integrate.hpp>

namespace MeshField {

EntityIntegration const* getIntegration(Mesh_Topology topo)
{
  if(topo != Triangle) {
    fail("getIntegration only supports triangles (Mesh_Topology::Triangle)\n");
  }
  return TriangleIntegration(); //I don't think this will work...
}

Integration const* EntityIntegration::getAccurate(int minimumAccuracy) const
{
  int n = countIntegrations();
  for (int i=0; i < n; ++i) {
    Integration const* integration = getIntegration(i);
    if (integration->getAccuracy() >= minimumAccuracy)
      return integration;
  }
  return NULL;
}

std::vector<IntegrationPoint> getIntegrationPoints(Mesh_Topology topo, int order) {
  auto ip = getIntegration(topo)->getAccurate(order)->getPoints();
  return ip;
}

double getIntegrationPointWeights(MeshElement* e, int order, int point)
{
  IntegrationPoint const* p =
    getIntegration(e->getType())->getAccurate(order)->getPoint(point);
  return p->weight;
}

template <typename FieldElement>
auto getJacobianDeterminants(FieldElement& fes, std::vector<IntegrationPoint> ip) {
  const auto numPtsPerElm = ip.size();
  const auto numMeshEnts = fes.numMeshEnts;
  const auto meshEntDim = fes.MeshEntDim;
  Kokkos::View<Real **> localCoords(numMeshEnts*numPtsPerElm, meshEntDim);
  //broadcast the points into the view - FIXME this is an inefficient use of memory
  for(size_t pt=0; pt < numPtsPerElm; pt++) {
    const auto point = ip.at(pt);
    const auto param = point.param;
    Kokkos::parallel_for(numMeshEnts, KOKKOS_LAMBDA(const int ent) {
        for(size_t i=0; i<param.size(); i++) {
          for(size_t d=0; i<meshEntDim; d++) {
            localCoords(ent*numPtsPerElm+pt,d) = param[d];
          }
        }
    });
  }
  auto J = getJacobians(fes, localCoords, ip.size());
  auto dV = getJacobianDeterminants(fes, J);
  return dV;
}

Integrator::Integrator(int o):
  order(o),
{
}

Integrator::~Integrator()
{
}

void Integrator::inElements()
{
}

void Integrator::outElements()
{
}

template <typename FieldElement>
void Integrator::process(FieldElement &fes)
{
  //TODO add check to only support triangles
  inElements();
  const auto topo = fes.elm2dof.getTopology();
  auto ip = getIntegrationPoints(topo,order);
  auto dV = getJacobianDeterminants(fes,ip);
  atPoints(fes,ip,dV);
  outElements();
  //TODO support distributed meshes by running a parallel reduction with user functor
}

} // namespace MeshField
