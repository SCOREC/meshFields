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

static KOKKOS_INLINE_FUNCTION
double f_analytic(double x, double y) {
  return x * x + y * y * y;
}

template <typename FieldElementT>
class AnalyticIntegral_BarycentricToPhysical final : public MeshField::Integrator {
public:
  AnalyticIntegral_BarycentricToPhysical(FieldElementT& fes_in,
                                         double x0, double y0,
                                         double x1, double y1,
                                         double x2, double y2)
    : MeshField::Integrator(
        MeshField::ReducedQuinticImplicitShape::Order)
    , fes(fes_in)
    , x0(x0), y0(y0)
    , x1(x1), y1(y1)
    , x2(x2), y2(y2)
  {}

  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *>  w,
                Kokkos::View<MeshField::Real *>  /*dV*/) override {
    const auto npts = w.extent(0);

    double sum_f = 0.0;
    double sum_1 = 0.0;

    Kokkos::parallel_reduce(
      "Integrate_f", npts,
      KOKKOS_LAMBDA(const int ip, double& acc) {

        const auto l0 = p(ip,0);
        const auto l1 = p(ip,1);
        const auto l2 = p(ip,2);

        const double x = l0 * x0 + l1 * x1 + l2 * x2;
        const double y = l0 * y0 + l1 * y1 + l2 * y2;

        acc += f_analytic(x,y) * w(ip);
      },
      sum_f
    );

    Kokkos::parallel_reduce(
      "Integrate_1", npts,
      KOKKOS_LAMBDA(const int ip, double& acc) {
        acc += w(ip);
      },
      sum_1
    );

    integral_f += sum_f;
    integral_1 += sum_1;
  }

  double integralOne() const { return integral_1; }
  double integralF()   const { return integral_f; }

private:
  FieldElementT& fes;

  double x0, y0, x1, y1, x2, y2;

  double integral_f = 0.0;
  double integral_1 = 0.0;
};

template <template <typename...> typename Controller>
bool runTest(Omega_h::Mesh& mesh) {
  constexpr double tolerance = 1e-10;
  constexpr int dim = 2;

  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto coordsField = omf.getCoordField();

  const auto [shape, map] = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  MeshField::FieldElement rqFE(mesh.nelems(), coordsField, shape, map);

  constexpr double x0 = 0.0, y0 = 0.0;
  constexpr double x1 = 1.0, y1 = 0.0;
  constexpr double x2 = 0.0, y2 = 1.0;

  AnalyticIntegral_BarycentricToPhysical integ(rqFE, x0, y0, x1, y1, x2, y2);

  integ.process(rqFE);

  constexpr double expected_1 = 0.5;
  constexpr double expected_f = 2.0 / 15.0;

  const double computed_1 = integ.integralOne();
  const double computed_f = integ.integralF();

  const double err_1 = std::fabs(computed_1 - expected_1);
  const double err_f = std::fabs(computed_f - expected_f);

  std::cout << "\n=== ReducedQuinticImplicit Analytic Integration Test ===\n";
  std::cout << std::setprecision(15);
  std::cout << "Integral(1): expected=" << expected_1
            << " computed=" << computed_1
            << " abs_err=" << err_1 << "\n";

  std::cout << "Integral(f): expected=" << expected_f
            << " computed=" << computed_f
            << " abs_err=" << err_f << "\n";

  if (err_1 > tolerance) {
    std::cerr << "[FAIL] Integral(1) mismatch\n";
    return false;
  }
  if (err_f > tolerance) {
    std::cerr << "[FAIL] Integral(f) too inaccurate\n";
    return false;
  }

  return true;
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  bool success = true;

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
        &mesh,
        OMEGA_H_SIMPLEX,
        2,
        verts,
        coords);

    success = runTest<MeshField::KokkosController>(mesh);
  }

  Kokkos::finalize();

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

