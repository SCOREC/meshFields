#include <Kokkos_Core.hpp>
#include <MeshField.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_mesh.hpp>
template<size_t order>
void runTest(Omega_h::Mesh mesh) {
    MeshField::OmegahMeshField<Kokkos::DefaultExecutionSpace, 2> mesh_field(
        mesh);
    auto shape_field = mesh_field.template CreateLagrangeField<double, order, 1>();

    // initialize vertices
    auto dim = mesh.dim();
    auto coords = mesh.coords();
    auto f = KOKKOS_LAMBDA(double x, double y)->double {
      return 2 * x + 3 * y;
    };
    Kokkos::parallel_for(
        mesh.nverts(), KOKKOS_LAMBDA(int vtx) {
          auto x = coords[vtx * dim];
          auto y = coords[vtx * dim + 1];
          shape_field(vtx, 0, 0, MeshField::Mesh_Topology::Vertex) = f(x, y);
        });
    const auto numPtsPerElem = 3;
    Kokkos::View<double **> local_coords("", mesh.nelems() * numPtsPerElem, 3);
    Kokkos::parallel_for(
        "set", mesh.nelems() * 3, KOKKOS_LAMBDA(const int &i) {
          local_coords(i, 0) = (i % 3 == 0);
          local_coords(i, 1) = (i % 3 == 1);
          local_coords(i, 2) = (i % 3 == 2);
        });

    auto eval_results =
        mesh_field.triangleLocalPointEval(local_coords, numPtsPerElem, shape_field);

    int errors = 0;
    const auto triVerts = mesh.ask_elem_verts();
    Kokkos::parallel_reduce(
        "test", mesh.nelems(),
        KOKKOS_LAMBDA(const int &tri, int &errors) {
          for (int node = 0; node < numPtsPerElem; ++node) {
            const auto triDim = 2;
            const auto vtxDim = 0;
            const auto ignored = -1;
            const auto localVtxIdx = Omega_h::simplex_down_template(triDim, vtxDim, node, ignored);
            const auto triToVtxDegree = Omega_h::simplex_degree(triDim, vtxDim);
            int vtx = triVerts[(tri * triToVtxDegree) + localVtxIdx];
            auto x = coords[vtx * dim];
            auto y = coords[vtx * dim + 1];
            auto expected = f(x, y);
            auto actual = eval_results(tri * numPtsPerElem + node, 0);
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
int main(int argc, char **argv) {
  // setup
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  {
    auto world = lib.world();
    auto mesh =
        Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 0, false);
    runTest<1>(mesh);
    runTest<2>(mesh);
  }
  Kokkos::finalize();
  return 0;
}
