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
#include <MeshField_GetHash.hpp>
#include <iostream>
#include <sstream>
#include <chrono>

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
  testIntegrator(FieldElement &fes_in, int order) : Integrator(order), integral(0), fes(fes_in) {};
  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) {
    std::cout << p.extent(0) << ":" << p.extent(1) << ":" << p.extent(2) << std::endl;
    const auto globalCoords = MeshField::evaluate(fes, p, p.extent(0) / fes.numMeshEnts);
    Kokkos::parallel_reduce("integrate", p.extent(0),
    KOKKOS_LAMBDA(const int &ent, MeshField::Real &integ) {
      const auto x = globalCoords(ent, 0);
      const auto y = globalCoords(ent, 1);
      const auto z = globalCoords.extent(1) == 3 ? globalCoords(ent, 2) : 1;
      integ += w(ent) * x * dV(ent);
    },
    integral);
  }

};

template <size_t dim, size_t ShapeOrder, template <typename ...> typename Controller>
void doRun(size_t size, Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  auto mesh = Omega_h::build_box(world, family, len, len, dim == 3 ? len : 0.0, size, size, dim == 3 ? size : 0);
  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto field = omf.template CreateLagrangeField<MeshField::Real, ShapeOrder, dim>();
  auto coords = mesh.coords();
  Kokkos::parallel_for(mesh.nverts(), KOKKOS_LAMBDA(int vtx) {
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
  auto start = std::chrono::high_resolution_clock::now();
  testInt.process(fes); 
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "RESULT: " << GitHash << "," << mesh.nelems() << "," << ShapeOrder << "," << testInt.getIntegral() << "," << std::chrono::duration<double>(end - start).count() << std::endl;
}


int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  bool useCab = true;
  size_t order = 2;
  size_t size = 2;
  size_t dim = 3;
  if (argc == 5) {
    useCab = atoi(argv[1]);
    order = atoi(argv[2]);
    size = atoi(argv[3]);
    dim = atoi(argv[4]);
  }
  // [dim][order][controller]
  void (* funcCall[2][2][2])(size_t, Omega_h::Library &) = {
  {{doRun<2, 1, MeshField::KokkosController>, doRun<2, 1, MeshField::CabanaController>}, 
  {doRun<2, 2, MeshField::KokkosController>, doRun<2, 2, MeshField::CabanaController>}}, 
  {{doRun<3, 1, MeshField::KokkosController>, doRun<3, 1, MeshField::CabanaController>}, 
  {doRun<3, 2, MeshField::KokkosController>, doRun<3, 2, MeshField::CabanaController>}}};
  funcCall[dim - 2][order - 1][useCab](size, lib);
  Kokkos::finalize();
  return 0;
}
