#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <MeshField_Integrate.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

template <typename FieldElementT>
class ReducedQuinticIntegrator : public MeshField::Integrator {
public:
  ReducedQuinticIntegrator(FieldElementT &fes_in)
      : MeshField::Integrator(
            MeshField::ReducedQuinticImplicitShape::Order),
        fes(fes_in),
        total(0.0) {}

  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) override {

    const int npts = w.extent(0);

    Kokkos::parallel_for("RQ_diag", npts, KOKKOS_LAMBDA(const int ip) {
      MeshField::Vector3 xi;
      xi[0] = p(ip,0);
      xi[1] = p(ip,1);
      xi[2] = p(ip,2);

      auto N  = fes.shapeFn.getValues(xi);
      auto dN = fes.shapeFn.getLocalGradients(xi);

      double sumN = 0, sumDx = 0, sumDy = 0;
      for (int a = 0;
           a < (int)MeshField::ReducedQuinticImplicitShape::numNodes;
           ++a) {
        sumN += N[a];
        sumDx += dN[a][0];
        sumDy += dN[a][1];
      }

      Kokkos::printf(
        "quad %2d: xi=(%.10f %.10f %.10f)  w=% .6e  dV=% .6e  "
        "sumN=% .6e  sumGrad=(%.6e %.6e)\n",
        ip, xi[0], xi[1], xi[2],
        double(w(ip)), double(dV(ip)),
        sumN, sumDx, sumDy
      );

      // print first few shape values
      for (int a=0; a<6; ++a) {
        Kokkos::printf("   N[%d]=%.6e  dN=% .6e % .6e\n",
                       a, N[a], dN[a][0], dN[a][1]);
      }
    });

    double local = 0.0;
    Kokkos::parallel_reduce("RQ_integrate", npts,
      KOKKOS_LAMBDA(const int i, double &acc) {
        acc += double(w(i));
      }, local);

    total += local;
  }

  void post() override {
    printf("[ReducedQuinticImplicit] Integrated Value = %.12e\n", total);
  }

  double getResult() const { return total; }

private:
  FieldElementT &fes;
  double total;
};

template <template <typename...> typename Controller>
void runTest(Omega_h::Mesh &mesh) {
  constexpr int dim = 2;

  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto coordsField = omf.getCoordField();

  const auto [shape, map] =
      MeshField::Omegah::getReducedQuinticImplicitElement(mesh);

  std::cout << "=== Reduced-Quintic Node Mapping ===\n";
  for (int a=0;
       a < (int)MeshField::ReducedQuinticImplicitShape::numNodes;
       ++a)
  {
    auto m = map(a, 0, 0, MeshField::Triangle);
    std::cout << "node " << a
              << " → (node=" << m.node
              << ", comp=" << m.component
              << ", entity=" << m.entity
              << ", topo=" << m.topo
              << ")\n";
  }
  std::cout << "====================================\n\n";

  MeshField::FieldElement FE(
      mesh.nelems(), coordsField, shape, map);

  ReducedQuinticIntegrator<decltype(FE)> integrator(FE);

  integrator.process(FE);

  double computed = integrator.getResult();
  double expected = 0.5;
  double err = std::fabs(computed - expected);

  std::cout << std::setprecision(12);
  std::cout << "Expected Integral = " << expected << "\n";
  std::cout << "Computed Integral = " << computed << "\n";
  std::cout << "Absolute Error    = " << err << "\n";

  if (err > 1e-10) {
    std::cerr << "[FAIL] ReducedQuinticImplicit integration mismatch\n";
    std::exit(EXIT_FAILURE);
  } else {
    std::cout << "[PASS] ReducedQuinticImplicit integration OK\n";
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    Omega_h::Library lib(&argc, &argv);

    Omega_h::Reals coords({
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0
    });

    Omega_h::LOs verts({0,1,2});

    Omega_h::Mesh mesh(&lib);
    Omega_h::build_from_elems_and_coords(
        &mesh, OMEGA_H_SIMPLEX, 2, verts, coords);

    runTest<MeshField::KokkosController>(mesh);
    }

  Kokkos::finalize();
  return 0;
}
