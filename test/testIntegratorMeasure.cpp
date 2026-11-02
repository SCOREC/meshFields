#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Integrate.hpp"
#include "MeshField_TestFunctions.hpp"
#include "Omega_h_build.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

// compute the measure (3D volume, 2D area, 1D length)
class MeasureIntegrator : public MeshField::Integrator {
public:
  MeasureIntegrator(int order) : MeshField::Integrator(order) {}
  void atPoints(Kokkos::View<MeshField::Real **>,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) override {
    MeshField::Real sum = 0;
    Kokkos::parallel_reduce(
        "integrateMeasure", dV.extent(0),
        KOKKOS_LAMBDA(const int i, MeshField::Real &lsum) {
          lsum += w(i % w.extent(0)) * dV(i);
        },
        sum);
    measure += sum;
  }
  MeshField::Real measure = 0;
};

/*
 * Integrate the measure of the mesh with a coordinate field of the given
 * order, created with coordMap.
 */
template <size_t dim, size_t order,
          typename CoordinateMap = MeshField::IdentityCoordinateMap>
bool testMeasure(Omega_h::Mesh &mesh, MeshField::Real expected,
                 std::string_view name,
                 CoordinateMap const &coordMap = CoordinateMap()) {
  MeshField::OmegahMeshField<ExecutionSpace, dim> omf(mesh);
  auto coordFieldWithCtrlr =
      omf.template CreateLagrangeCoordinateField<order>(coordMap);
  auto elm = omf.CreateLagrangeElement(coordFieldWithCtrlr.field);
  MeasureIntegrator integrator(order);
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
    failed += !testMeasure<2, 1>(tris, 1.0, "triangles");
    failed += !testMeasure<3, 1>(tets, 6.0, "tetrahedra");
    failed += !testMeasure<2, 2>(tris, 1.0, "quadratic triangles");
    failed += !testMeasure<3, 2>(tets, 6.0, "quadratic tetrahedra");
    failed += !testMeasure<3, 2>(tets, 6.0, "sheared quadratic tetrahedra",
                                 Shear<3>());
    // the unit square with its top bent into y = 1 + delta * x(1 - x).  The
    // top edges are parabolas in their edge parameter, so quadratic elements
    // reproduce the boundary, and the area, exactly.
    const MeshField::Real delta = 0.1;
    failed += !testMeasure<2, 2>(
        tris, 1.0 + delta / 6.0, "curved quadratic triangles",
        KOKKOS_LAMBDA(Vec<2> x) {
          x[1] *= 1 + delta * x[0] * (1 - x[0]);
          return x;
        });
  }
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
