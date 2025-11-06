#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <MeshField_Integrate.hpp>
#include <iostream>
#include <cmath>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

Omega_h::Mesh createMesh(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  const auto len = 1.0;
  return Omega_h::build_box(world, family, len, len, 0.0, 2, 2, 0);
}

template <typename FieldElement>
class ReducedQuinticImplicitIntegrator : public MeshField::Integrator {
public:
  ReducedQuinticImplicitIntegrator(FieldElement &fes_in)
      : MeshField::Integrator(5), fes(fes_in), totalValue(0.0) {}

  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) override {
    const auto numPts = p.extent(0);
    double localSum = 0.0;

    Kokkos::parallel_reduce(
        "IntegrateReducedQuinticImplicit", numPts,
        KOKKOS_LAMBDA(const int i, double &sum) {
          const double weight = w(i);
          const double jac = dV(i);
          const double f = 1.0; // constant field
          sum += f * weight * jac;
        },
        localSum);

    totalValue += localSum;
  }

  void post() override {
    printf("[ReducedQuinticImplicit] Integrated Value = %.12e\n", totalValue);
  }

  double getResult() const { return totalValue; }

private:
  FieldElement &fes;
  double totalValue;
};

template <template <typename...> typename Controller>
void runReducedQuinticImplicit(Omega_h::Mesh &mesh,
                               MeshField::OmegahMeshField<ExecutionSpace, 2, Controller> &omf) {
  constexpr auto ShapeOrder = 5;

  auto field = omf.getCoordField();
  const auto [shp, map] = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  MeshField::FieldElement fes(mesh.nelems(), field, shp, map);

  ReducedQuinticImplicitIntegrator integrator(fes);
  integrator.process(fes);

  const double expected = 1.0;
  const double result = integrator.getResult();
  const double error = std::fabs(result - expected);

  std::cout << "\nExpected Integral = " << expected
            << "\nComputed Integral = " << result
            << "\nAbsolute Error   = " << error << "\n";

  if (error < 1e-6)
    std::cout << "[PASS] ReducedQuinticImplicit integration test succeeded\n";
  else
    std::cerr << "[FAIL] Integration mismatch\n", exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  Omega_h::Library lib(&argc, &argv);

  {
    auto mesh2D = createMesh(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::KokkosController> omf2D(mesh2D);
    runReducedQuinticImplicit<MeshField::KokkosController>(mesh2D, omf2D);
  }

  Kokkos::finalize();
  return 0;
}

