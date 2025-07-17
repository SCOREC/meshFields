#include <Kokkos_Core.hpp>
#include <MeshField.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_mesh.hpp>

// Uncommenting will reorder the parametric evaluation coordinates which causes
// the test to pass.
// #define REORDER_COORDS

int main(int argc, char **argv) {
  // setup
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  {
    auto world = lib.world();
    auto mesh =
        Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 0, false);
    MeshField::OmegahMeshField<Kokkos::DefaultExecutionSpace, 2> mesh_field(
        mesh);
    auto shape_field = mesh_field.template CreateLagrangeField<double, 1, 1>();

    // initialize vertices
    auto dim = mesh.dim();
    auto coords = mesh.coords();
    auto f = KOKKOS_LAMBDA(double x, double y)->double {
      return 2 * x + 3 * y;
    };
    // auto f = [](double x, double y) -> double { return 2 * x + 3 * y; };
    Kokkos::parallel_for(
        mesh.nverts(), KOKKOS_LAMBDA(int vtx) {
          auto x = coords[vtx * dim];
          auto y = coords[vtx * dim + 1];
          shape_field(vtx, 0, 0, MeshField::Mesh_Topology::Vertex) = f(x, y);
        });

    Kokkos::View<double **> local_coords("", mesh.nelems() * 3, 3);
    Kokkos::parallel_for(
        "set", mesh.nelems() * 3, KOKKOS_LAMBDA(const int &i) {
          local_coords(i, 0) = (i % 3 == 0);
          local_coords(i, 1) = (i % 3 == 1);
          local_coords(i, 2) = (i % 3 == 2);
        });

    auto eval_results =
        mesh_field.triangleLocalPointEval(local_coords, 3, shape_field);

    MeshField::Omegah::LinearTriangleToVertexField map(mesh);
    int errors = 0;
    Kokkos::parallel_reduce(
        "test", mesh.nelems(),
        KOKKOS_LAMBDA(const int &tri, int &errors) {
          for (int node = 0; node < 3; ++node) {
            auto mapping =
                map(node, 0, tri, MeshField::Mesh_Topology::Triangle);
            int vtx = mapping.entity;
            auto x = coords[vtx * dim];
            auto y = coords[vtx * dim + 1];
            auto expected = f(x, y);
            auto actual = eval_results(tri * 3, node);
            if (Kokkos::fabs(expected - actual) > MeshField::MachinePrecision) {
              ++errors;
              Kokkos::printf(
                  "expected: %lf, actual: %lf, element: %d, node: %d\n",
                  expected, actual, tri, node);
            }
          }
        },
        errors);
    if (errors > 0) {
      MeshField::fail("One or more mappings did not match\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
