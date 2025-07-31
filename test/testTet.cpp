#include <Kokkos_Core.hpp>
#include <MeshField.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_mesh.hpp>

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  {
    auto world = lib.world();
    auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 10, false);
    MeshField::OmegahMeshField<Kokkos::DefaultExecutionSpace, 3> mesh_field(mesh);
    auto shape_field = 
        mesh_field.template CreateLagrangeField<double, 1, 1>();
    auto dim = mesh.dim();
    auto coords = mesh.coords();
    auto f = KOKKOS_LAMBDA(double x, double y, double z)->double {return 2 * x + 3 * y + 4 * z;};
    Kokkos::parallel_for(mesh.nverts(), KOKKOS_LAMBDA(int vtx) {
        auto x = coords[vtx * dim];
        auto y = coords[vtx * dim + 1];
        auto z = coords[vtx * dim + 2];
        shape_field(vtx, 0, 0, MeshField::Mesh_Topology::Vertex) = f(x, y, z);
    });
    const auto numNodesPerElem = 4;
    Kokkos::View<double **> local_coords("", mesh.nelems() * numNodesPerElem, 4);
    Kokkos::parallel_for("set", mesh.nelems() * numNodesPerElem, KOKKOS_LAMBDA(const int &i) {
      const auto val = i % numNodesPerElem;
      local_coords(i, 0) = (val == 0);
      local_coords(i, 1) = (val == 1);
      local_coords(i, 2) = (val == 2);
      local_coords(i, 3) = (val == 3);

    });
    auto eval_results = mesh_field.tetrahedronLocalPointEval(
        local_coords, numNodesPerElem, shape_field);

    int errors = 0;
    const auto tetVerts = mesh.ask_elem_verts();
    const auto coordField = mesh_field.getCoordField();
    Kokkos::parallel_reduce(
        "test", mesh.nelems(),
        KOKKOS_LAMBDA(const int &tet, int &errors) {
          for (int node = 0; node < 4; ++node) {
            const auto tetDim = 3;
            const auto vtxDim = 0;
            const auto ignored = -1;
            const auto localVtxIdx = Omega_h::simplex_down_template(tetDim, vtxDim, node, ignored);
            const auto tetToVtxDegree = Omega_h::simplex_degree(tetDim, vtxDim);
            int vtx = tetVerts[(tet * tetToVtxDegree) + localVtxIdx];
            auto x = coords[vtx * dim];
            auto y = coords[vtx * dim + 1];
            auto z = coords[vtx * dim + 2];
            auto expected = f(x, y, z);
            auto actual = eval_results(tet * numNodesPerElem + node, 0);
            if (Kokkos::fabs(expected - actual) > MeshField::MachinePrecision) {
              ++errors;
              
              Kokkos::printf(
                  "expected: %lf, actual: %lf, element: %d, node(vtx): %d\n",
                  expected, actual, tet, node);
            }
          }
        }, errors);
     if (errors > 0) {
       MeshField::fail("One or more mappings did not match\n");
     }
  }
  Kokkos::finalize();
  return 0;
}
