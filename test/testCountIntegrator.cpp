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
template <size_t dim> Omega_h::Mesh createMesh(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  const auto numEnts3d = (dim == 3 ? 3 : 0);
  return Omega_h::build_box(world, family, len, len, len, 3, 3, numEnts3d);
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

template <template <typename...> typename Controller, size_t dim>
void doRun(Omega_h::Mesh &mesh,
           MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf) {
  const auto ShapeOrder = 1;
  auto field = omf.getCoordField();
  auto shapeSet = [&]() {
    if constexpr (dim == 3) {
      return MeshField::Omegah::getTetrahedronElement<ShapeOrder>(mesh);
    } else {
      return MeshField::Omegah::getTriangleElement<ShapeOrder>(mesh);
    }
  };
  const auto [shp, map] = shapeSet();
  MeshField::FieldElement fes(mesh.nelems(), field, shp, map);

  CountIntegrator countInt(fes);
  countInt.process(fes);
  assert(static_cast<size_t>(mesh.nelems()) == countInt.getCount());
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
#ifdef MESHFIELDS_ENABLE_CABANA
  {
    auto mesh2D = createMesh<2>(lib);
    auto mesh3D = createMesh<3>(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::CabanaController>
        omf2D(mesh2D);
    doRun<MeshField::CabanaController>(mesh2D, omf2D);
    MeshField::OmegahMeshField<ExecutionSpace, 3, MeshField::CabanaController>
        omf3D(mesh3D);
    doRun<MeshField::CabanaController>(mesh3D, omf3D);
  }
#endif
  {
    auto mesh2D = createMesh<2>(lib);
    auto mesh3D = createMesh<3>(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::KokkosController>
        omf2D(mesh2D);
    doRun<MeshField::KokkosController>(mesh2D, omf2D);
    MeshField::OmegahMeshField<ExecutionSpace, 3, MeshField::KokkosController>
        omf3D(mesh3D);
    doRun<MeshField::KokkosController>(mesh3D, omf3D);
  }
  Kokkos::finalize();
  return 0;
}
