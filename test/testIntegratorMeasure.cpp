#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Integrate.hpp"
#include "Omega_h_build.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

// compute the measure (3D volume, 2D area, 1D length)
class MeasureIntegrator : public MeshField::Integrator {
public:
  MeasureIntegrator() : MeshField::Integrator(1) {}
  void atPoints(Kokkos::View<MeshField::Real **>,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) override {
    MeshField::Real sum = 0;
    Kokkos::parallel_reduce(
        "integrateMeasure", w.extent(0),
        KOKKOS_LAMBDA(const int i, MeshField::Real &lsum) {
          lsum += w(i) * dV(i);
        },
        sum);
    measure += sum;
  }
  MeshField::Real measure = 0;
};

template <size_t dim>
bool testMeasure(Omega_h::Mesh &mesh, MeshField::Real expected,
                 std::string_view name) {
  MeshField::OmegahMeshField<ExecutionSpace, dim> omf(mesh);
  auto coordField = omf.getCoordField().field;
  const auto [shp, map] = [&]() {
    if constexpr (dim == 3) {
      return MeshField::Omegah::getTetrahedronElement<1>(mesh);
    } else {
      return MeshField::Omegah::getTriangleElement<1>(mesh);
    }
  }();
  MeshField::FieldElement elm(mesh.nelems(), coordField, shp, map);
  MeasureIntegrator integrator;
  integrator.process(elm);
  const auto err = std::abs(integrator.measure - expected);
  const bool pass = err <= MeshField::Epsilon;
  std::cout << "testMeasure(" << name << "): " << integrator.measure
            << " expected " << expected << (pass ? " pass" : " FAIL") << "\n";
  return pass;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    auto lib = Omega_h::Library(&argc, &argv);
    auto tris = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0,
                                   0.0, 3, 3, 0);
    auto tets = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 2.0,
                                   3.0, 2, 2, 2);
    failed += !testMeasure<2>(tris, 1.0, "triangles");
    failed += !testMeasure<3>(tets, 6.0, "tetrahedra");
  }
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
