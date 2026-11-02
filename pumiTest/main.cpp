#include <apf.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfBox.h>
#include <pcu_util.h>
#include <pumi_version.h>
#include <PCU.h>
#include <iostream>
#include <gmi_mesh.h>
#include <gmi_null.h>
#include <chrono>

class CountIntegrator : public apf::Integrator {
  protected:
    double numEnt;
    apf::MeshElement* currentMeshElement;
  public:
    double getCount() {return numEnt;}
    void resetCount() { numEnt = 0; }
    CountIntegrator(size_t order) : Integrator(order), numEnt(0) {};
    void inElement(apf::MeshElement* me) override {
      currentMeshElement = me;
    }
    void atPoint(apf::Vector3 const& point, double weight, double jacobian) override {
      apf::Vector3 globalCoord;
      apf::mapLocalToGlobal(currentMeshElement, point, globalCoord);
      numEnt += weight * globalCoord[0] * jacobian;
    }
};
int main(int argc, char ** argv) {
  pcu::Init(&argc, &argv);
  {
  pcu::PCU PCUObj;
  size_t order = atoi(argv[1]);
  size_t size = atoi(argv[2]);
  size_t numberOfElems = atoi(argv[3]);
  size_t dim = atoi(argv[4]);
  size_t runs = atoi(argv[5]);
  gmi_register_mesh();
  gmi_register_null();
  apf::Mesh2* mesh = apf::makeMdsBox(numberOfElems, numberOfElems, dim == 3 ? numberOfElems : 0, size, size, dim == 3 ? size : 0, 1, &PCUObj);
  CountIntegrator * countInt = new CountIntegrator(order);
  for (int i = 0; i < runs; ++i) {
    countInt->resetCount();
    auto start = std::chrono::high_resolution_clock::now();
    countInt->process(mesh);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "RESULT: " << pumi_git_sha() << "," << mesh->count(mesh->getDimension()) << "," << order << "," << countInt->getCount() << "," << std::chrono::duration<double>(end - start).count() << std::endl;
  }
  delete countInt;
  mesh->destroyNative();
  apf::destroyMesh(mesh);
  }
  pcu::Finalize();
  return 0;
}
