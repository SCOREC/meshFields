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

// Analytic function to integrate over the physical triangle.
static inline double f_analytic(double x, double y) {
  return x * x + y * y * y;
}

/**
 * Integrator that:
 *  - receives quadrature points `p` in *barycentric/area coordinates* on the reference triangle
 *  - maps those barycentric coordinates to *physical (x,y)* using the element's vertex coordinates
 *  - integrates f(x,y) and 1 using quadrature weights `w`
 *
 * Notes:
 *  - `p` is expected to be nonnegative and sum to 1 (barycentric).
 *  - `w` is assumed to already include the differential volume (dV), so we don't multiply by `dV`.
 */
template <typename FieldElementT>
class AnalyticIntegral_BarycentricToPhysical final : public MeshField::Integrator {
public:
  // Pass triangle vertex coordinates explicitly. These define the physical element geometry.
  AnalyticIntegral_BarycentricToPhysical(FieldElementT& fes_in,
                                         double x0, double y0,
                                         double x1, double y1,
                                         double x2, double y2)
    : MeshField::Integrator(MeshField::ReducedQuinticImplicitShape::Order) // quadrature order used by process()
    , fes(fes_in)
    , x0(x0), y0(y0)
    , x1(x1), y1(y1)
    , x2(x2), y2(y2)
    , integral_f(0.0)
    , integral_1(0.0)
  {}

  /**
   * Called once per process() invocation with:
   *  - p: local (barycentric) coordinates for each quadrature point
   *  - w: quadrature weights (including dV per the test assumption)
   *  - dV: provided by the framework, but unused here
   */
  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> /*dV*/) override
  {
    const int npts = static_cast<int>(w.extent(0));

    // Accumulators local to this element's quadrature loop.
    double sum_f = 0.0;
    double sum_1 = 0.0;

    // Integrate analytic f over the physical triangle.
    Kokkos::parallel_reduce(
      "Integrate_f_bary_to_phys", npts,
      KOKKOS_LAMBDA(const int ip, double& acc) {
        // Barycentric weights (l0,l1,l2) corresponding to the element's vertices (v0,v1,v2).
        const double l0 = static_cast<double>(p(ip, 0));
        const double l1 = static_cast<double>(p(ip, 1));
        const double l2 = static_cast<double>(p(ip, 2));

        // Map barycentric coordinates to physical (x,y) via affine combination of vertex coords.
        // For any triangle, (x,y) = l0*(x0,y0) + l1*(x1,y1) + l2*(x2,y2).
        const double x = l0 * x0 + l1 * x1 + l2 * x2;
        const double y = l0 * y0 + l1 * y1 + l2 * y2;

        // Accumulate f(x,y) * w. We assume w already includes dV.
        acc += f_analytic(x, y) * static_cast<double>(w(ip)); // w includes dV
      },
      sum_f
    );

    // Integrate constant 1 over the triangle: should equal its area.
    Kokkos::parallel_reduce(
      "Integrate_1_bary_to_phys", npts,
      KOKKOS_LAMBDA(const int ip, double& acc) {
        acc += static_cast<double>(w(ip));
      },
      sum_1
    );

    // Store totals (for this single-element test, these are the final integrals).
    integral_f += sum_f;
    integral_1 += sum_1;
  }

  double integralOne() const { return integral_1; }
  double integralF() const { return integral_f; }

private:
  // The FieldElement is passed to process(); this member is kept for conventional ownership/context,
  // even though this integrator doesn't directly query fes during atPoints().
  FieldElementT& fes; // kept to match Integrator::process signature usage; not otherwise needed

  // Physical vertex coordinates for the element geometry.
  double x0, y0, x1, y1, x2, y2;

  // Accumulated integrals.
  double integral_f, integral_1;
};

template <template <typename...> typename Controller>
void runTest(Omega_h::Mesh& mesh) {
  constexpr int dim = 2;

  // Wrap the Omega_h mesh in a MeshField interface and fetch the coordinate field (used to build FE).
  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  auto coordsField = omf.getCoordField();

  // Build the reduced quintic implicit element (shape + dof map) and then a FieldElement.
  // In this test, rqFE is used to drive the quadrature points/weights through Integrator::process().
  const auto [shape, map] = MeshField::Omegah::getReducedQuinticImplicitElement(mesh);
  MeshField::FieldElement rqFE(mesh.nelems(), coordsField, shape, map);

  // Physical triangle vertices: (0,0), (1,0), (0,1).
  // This is the unit right triangle with area 0.5.
  const double x0 = 0.0, y0 = 0.0;
  const double x1 = 1.0, y1 = 0.0;
  const double x2 = 0.0, y2 = 1.0;

  // Run the integration using barycentric->physical mapping.
  AnalyticIntegral_BarycentricToPhysical<decltype(rqFE)> integ(rqFE, x0, y0, x1, y1, x2, y2);
  integ.process(rqFE);

  const double expected_1 = 0.5;
  const double expected_f = 2.0 / 15.0;

  const double computed_1 = integ.integralOne();
  const double computed_f = integ.integralF();

  const double err_1 = std::fabs(computed_1 - expected_1);
  const double err_f = std::fabs(computed_f - expected_f);

  std::cout << "\n=== ReducedQuinticImplicit: Analytic Integration Test (barycentric->physical) ===\n";
  std::cout << std::setprecision(15);
  std::cout << "Integral(1): expected=" << expected_1 << " computed=" << computed_1 << " abs_err=" << err_1 << "\n";
  std::cout << "Integral(f): expected=" << expected_f << " computed=" << computed_f << " abs_err=" << err_f << "\n";

  if (err_1 > 1e-10) {
    std::cerr << "[FAIL] Integral(1) mismatch\n";
    std::exit(EXIT_FAILURE);
  }
  if (err_f > 1e-6) {
    std::cerr << "[FAIL] Integral(f) too inaccurate\n";
    std::exit(EXIT_FAILURE);
  }

  std::cout << "[PASS] ReducedQuinticImplicit analytic integration OK\n";
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Omega_h::Library lib(&argc, &argv);

    // Build a single-triangle mesh with vertices (0,0), (1,0), (0,1).
    Omega_h::Reals coords({
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0
    });
    Omega_h::LOs verts({0, 1, 2});

    Omega_h::Mesh mesh(&lib);
    Omega_h::build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, 2, verts, coords);

    runTest<MeshField::KokkosController>(mesh);
  }
  Kokkos::finalize();
  return 0;
}

