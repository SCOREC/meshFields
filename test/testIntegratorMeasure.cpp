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
  MeasureIntegrator(int order) : MeshField::Integrator(order) {}
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

/*
 * Move the edge nodes of the edges on y = 1 of a unit square by delta in y,
 * so each of those edges becomes a parabola that adds 2/3 * length * delta
 * to the area: 2/3 * delta in all.
 */
template <typename CoordField>
void moveTopEdges(Omega_h::Mesh &mesh, CoordField coordField,
                  MeshField::Real delta) {
  const auto edgeVerts = mesh.ask_verts_of(Omega_h::EDGE);
  const auto coords = mesh.coords();
  Kokkos::parallel_for(
      "moveTopEdges", mesh.nedges(), KOKKOS_LAMBDA(const int edge) {
        const auto a = edgeVerts[edge * 2];
        const auto b = edgeVerts[edge * 2 + 1];
        if (Kokkos::fabs(coords[a * 2 + 1] - 1) < MeshField::Epsilon &&
            Kokkos::fabs(coords[b * 2 + 1] - 1) < MeshField::Epsilon)
          coordField(edge, 0, 1, MeshField::Edge) += delta;
      });
}

/*
 * Integrate the measure of the mesh with a coordinate field of the given
 * order.
 */
template <size_t dim, size_t order>
bool testMeasure(Omega_h::Mesh &mesh, MeshField::Real expected,
                 std::string_view name, MeshField::Real delta = 0) {
  MeshField::OmegahMeshField<ExecutionSpace, dim> omf(mesh);
  auto coordFieldWithCtrlr =
      omf.template CreateLagrangeCoordinateField<order>();
  if constexpr (dim == 2 && order == 2) {
    if (delta != 0)
      moveTopEdges(mesh, coordFieldWithCtrlr.field, delta);
  } else if (delta != 0) {
    std::cout << "testMeasure(" << name
              << "): moving edges needs a quadratic 2d field FAIL\n";
    return false;
  }
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
    const MeshField::Real delta = 0.1;
    failed += !testMeasure<2, 2>(tris, 1.0 + (2.0 / 3.0) * delta,
                                 "curved quadratic triangles", delta);
  }
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
