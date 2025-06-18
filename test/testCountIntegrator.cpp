#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"    //remove?
#include "MeshField_Fail.hpp"       //remove?
#include "MeshField_For.hpp"        //remove?
#include "MeshField_ShapeField.hpp" //remove?
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <MeshField_Integrate.hpp>
#include <iostream>
#include <sstream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

Omega_h::Mesh createMeshTri18(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  return Omega_h::build_box(world, family, len, len, 0.0, 3, 3, 0);
}

template <typename AnalyticFunction, typename ShapeField>
void setVertices(Omega_h::Mesh &mesh, AnalyticFunction func, ShapeField field) {
  const auto MeshDim = mesh.dim();
  auto coords = mesh.coords();
  auto setFieldAtVertices = KOKKOS_LAMBDA(const int &vtx) {
    // get dofholder position at the midpoint of edge
    // - TODO should be encoded in the field?
    const auto x = coords[vtx * MeshDim];
    const auto y = coords[vtx * MeshDim + 1];
    field(vtx, 0, 0, MeshField::Vertex) = func(x, y);
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nverts()},
                          setFieldAtVertices, "setFieldAtVertices");
}

template <typename FieldElement>
class CountIntegrator : public MeshField::Integrator {
private:
  CountIntegrator(){};

protected:
  size_t count;
  FieldElement &fes;

public:
  unsigned int getCount() { return count; }
  CountIntegrator(FieldElement &fes_in)
      : Integrator(1), count(0), fes(fes_in){};
  void atPoints(Kokkos::View<MeshField::Real **>,
                Kokkos::View<MeshField::Real *>,
                Kokkos::View<MeshField::Real *>) {
    count = fes.numMeshEnts;
  }
};

template <template <typename...> typename Controller>
void doRun(Omega_h::Mesh &mesh,
           MeshField::OmegahMeshField<ExecutionSpace, 2, Controller> &omf) {
  const auto ShapeOrder = 1;
  auto field = omf.getCoordField();
  const auto [shp, map] =
      MeshField::Omegah::getTriangleElement<ShapeOrder>(mesh);
  MeshField::FieldElement fes(mesh.nelems(), field, shp, map);

  CountIntegrator countInt(fes);
  countInt.process(fes);
  assert(mesh.nelems() == countInt.getCount());
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  auto mesh = createMeshTri18(lib);
#ifdef MESHFIELDS_ENABLE_CABANA
  {
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::CabanaController>
        omf(mesh);
    doRun<MeshField::CabanaController>(mesh, omf);
  }
#endif
  {
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::KokkosController>
        omf(mesh);
    doRun<MeshField::KokkosController>(mesh, omf);
  }
  return 0;
}
