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

template <typename FieldElement>
class testIntegrator : public MeshField::Integrator {
private:
  testIntegrator(){};

protected:
  MeshField::Real integral;
  FieldElement &fes;

public:
  MeshField::Real getIntegral() { return integral; }
  testIntegrator(FieldElement &fes_in, int order)
      : Integrator(order), integral(0), fes(fes_in){};
  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) {
    const auto globalCoords = MeshField::evaluate(fes, p, p.extent(0) / fes.numMeshEnts);
    Kokkos::parallel_reduce(
        "integrate", p.extent(0),
        KOKKOS_LAMBDA(const int &ent, MeshField::Real &integ) {
          const auto x = globalCoords(ent, 0);
          const auto y = globalCoords(ent, 1);
          const auto z = globalCoords.extent(1) == 3 ? globalCoords(ent, 2) : 1;
          integ += w(ent) * x * dV(ent);
        },
        integral);
  }
};

template <template <typename...> typename Controller, size_t ShapeOrder,
          size_t dim>
void doRun(Omega_h::Mesh &mesh,
           MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf) {
  auto field =
      omf.template CreateLagrangeField<MeshField::Real, ShapeOrder, dim>();
  auto coords = mesh.coords();
  Kokkos::parallel_for(
      mesh.nverts(), KOKKOS_LAMBDA(int vtx) {
        field(vtx, 0, 0, MeshField::Vertex) = coords[vtx * dim];
        field(vtx, 0, 1, MeshField::Vertex) = coords[vtx * dim + 1];
        if constexpr (dim == 3) {
          field(vtx, 0, 2, MeshField::Vertex) = coords[vtx * dim + 2];
        }
      });
  if (ShapeOrder == 2) {
    auto edge2vtx = mesh.ask_down(1, 0).ab2b;
    auto edgeMap = mesh.ask_down(dim, 1).ab2b;
    Kokkos::parallel_for(
        mesh.nedges(), KOKKOS_LAMBDA(int edge) {
          const auto left = edge2vtx[edge * 2];
          const auto right = edge2vtx[edge * 2 + 1];
          const auto x = (coords[left * dim] + coords[right * dim]) / 2.0;
          const auto y =
              (coords[left * dim + 1] + coords[right * dim + 1]) / 2.0;
          field(edge, 0, 0, MeshField::Edge) = x;
          field(edge, 0, 1, MeshField::Edge) = y;
          if constexpr (dim == 3) {
            const auto z =
                (coords[left * dim + 2] + coords[right * dim + 2]) / 2.0;
            field(edge, 0, 2, MeshField::Edge) = z;
          }
        });
  }

  auto shapeSet = [&]() {
    if constexpr (dim == 3) {
      return MeshField::Omegah::getTetrahedronElement<ShapeOrder>(mesh);
    } else {
      return MeshField::Omegah::getTriangleElement<ShapeOrder>(mesh);
    }
  };
  const auto [shp, map] = shapeSet();
  MeshField::FieldElement fes(mesh.nelems(), field, shp, map);
  testIntegrator testInt(fes, ShapeOrder);
  testInt.process(fes);
  std::cout << testInt.getIntegral() << std::endl;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
#ifdef MESHFIELDS_ENABLE_CABANA
  {
    Omega_h::Mesh mesh3D = Omega_h::build_box(world, family, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0);
    MeshField::OmegahMeshField<ExecutionSpace, 3, MeshField::CabanaController>
        omf3D(mesh3D);
    doRun<MeshField::CabanaController, 1>(mesh3D, omf3D);
    doRun<MeshField::CabanaController, 2>(mesh3D, omf3D);
  }
#endif
  {
  }
  Kokkos::finalize();
  return 0;
}
