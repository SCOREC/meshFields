#include <gmi_mesh.h>
#include <apfMDS.h>
#include <apfBox.h>
#include <apfMesh2.h>
#include <apfShape.h>
#include <apfDynamicVector.h>
#include <PCU.h>
#include <iostream>
#include <cassert>
void getComp(apf::Element* element, apf::Vector3 p) {
  double val;
  apf::getComponents(element, p, &val);
  std::cout << "val " << val << "\n";
}

int main(int argc, char** argv) {
  MPI_Init(&argc,&argv);
  {
  pcu::PCU PCUObj;
  gmi_register_mesh();
  apf::Mesh2* m = apf::makeMdsBox(1,1,0,1,1,0,1,&PCUObj);
  std::cout << "num tri " << m->count(2) << " num vtx " << m->count(0) << "\n";
  //field on vertices
  apf::Field* f = apf::createLagrangeField(m, "vals", apf::SCALAR, 1);
  //set field
  size_t i = 0;
  apf::MeshIterator* vtxIt = m->begin(0);
  apf::MeshEntity* v;
  while ((v = m->iterate(vtxIt))) {
    apf::setScalar(f, v, 0, i); // set as 100
    std::cout << "setting vtx to " << i << "\n";
    i++;
  }
  m->end(vtxIt);
  //get components of the first triangle
  apf::MeshIterator* rIt = m->begin(m->getDimension());
  apf::MeshEntity* tri = m->iterate(rIt);
  assert(tri);
  apf::Element* element = apf::createElement(f, tri);
  apf::MeshEntity* triVerts[3];
  int numTriVerts = m->getDownward(tri, 0, triVerts);
  assert(numTriVerts == 3);
  for(int i=0; i<numTriVerts; i++) {
    std::cout << i << " expected " << apf::getScalar(f, triVerts[i], 0) << " ";
    apf::Vector3 xi = {0,0,0};
    xi[i] = 1;
    getComp(element, xi);
  }
  //write to vtk files that can be visualized in paraview
  apf::writeVtkFiles("twoTri", m);
  }
  MPI_Finalize();
  return 0;
}
