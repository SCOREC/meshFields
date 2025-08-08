#include <Kokkos_Core.hpp>
#include <MeshField.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_mesh.hpp>
template <size_t order, size_t dim> void runTest(Omega_h::Mesh mesh) {
  MeshField::OmegahMeshField<Kokkos::DefaultExecutionSpace, dim> mesh_field(
      mesh);
  const auto vertexNodes = (dim == 3 ? 4 : 3);
  auto shape_field =
      mesh_field.template CreateLagrangeField<double, order, 1>();
  // initialize vertices
  auto coords = mesh.coords();
  auto f = KOKKOS_LAMBDA(double x, double y, double z)->double {
    return 2 * x + 3 * y + 4 * z;
  };
  auto edge2vtx = mesh.ask_down(1, 0).ab2b;
  auto edgeMap = mesh.ask_down(dim, 1).ab2b;
  Kokkos::parallel_for(
      mesh.nverts(), KOKKOS_LAMBDA(int vtx) {
        auto x = coords[vtx * dim];
        auto y = coords[vtx * dim + 1];
        auto z = dim == 3 ? coords[vtx * dim + 2] : 0;
        shape_field(vtx, 0, 0, MeshField::Mesh_Topology::Vertex) = f(x, y, z);
      });
  if (order == 2) {
    Kokkos::parallel_for(
        mesh.nedges(), KOKKOS_LAMBDA(int edge) {
          const auto left = edge2vtx[edge * 2];
          const auto right = edge2vtx[edge * 2 + 1];
          const auto x = (coords[left * dim] + coords[right * dim]) / 2.0;
          const auto y =
              (coords[left * dim + 1] + coords[right * dim + 1]) / 2.0;
          const auto z =
              dim == 3
                  ? (coords[left * dim + 2] + coords[right * dim + 2]) / 2.0
                  : 0;
          shape_field(edge, 0, 0, MeshField::Mesh_Topology::Edge) = f(x, y, z);
        });
  }
  const auto numNodesPerElem = order == 2 ? (dim == 3 ? 10 : 6) : vertexNodes;
  Kokkos::View<double **> local_coords("", mesh.nelems() * numNodesPerElem,
                                       vertexNodes);
  Kokkos::parallel_for(
      "set", mesh.nelems() * numNodesPerElem, KOKKOS_LAMBDA(const int &i) {
        const auto val = i % numNodesPerElem;
        local_coords(i, 0) = (val == 0);
        local_coords(i, 1) = (val == 1);
        local_coords(i, 2) = (val == 2);
        if constexpr (dim == 3) {
          local_coords(i, 3) = (val == 3);
          if constexpr (order == 2) {
            if (val == 4) {
              local_coords(i, 0) = 1 / 2.0;
              local_coords(i, 1) = 1 / 2.0;
            } else if (val == 5) {
              local_coords(i, 1) = 1 / 2.0;
              local_coords(i, 2) = 1 / 2.0;
            } else if (val == 6) {
              local_coords(i, 0) = 1 / 2.0;
              local_coords(i, 2) = 1 / 2.0;
            } else if (val == 7) {
              local_coords(i, 0) = 1 / 2.0;
              local_coords(i, 3) = 1 / 2.0;
            } else if (val == 8) {
              local_coords(i, 1) = 1 / 2.0;
              local_coords(i, 3) = 1 / 2.0;
            } else if (val == 9) {
              local_coords(i, 2) = 1 / 2.0;
              local_coords(i, 3) = 1 / 2.0;
            }
          }

        } else if constexpr (order == 2) {
          if (val == 3) {
            local_coords(i, 0) = 1 / 2.0;
            local_coords(i, 1) = 1 / 2.0;
          } else if (val == 4) {
            local_coords(i, 1) = 1 / 2.0;
            local_coords(i, 2) = 1 / 2.0;
          } else if (val == 5) {
            local_coords(i, 0) = 1 / 2.0;
            local_coords(i, 2) = 1 / 2.0;
          }
        }
      });

  auto eval_results = dim == 3
                          ? mesh_field.tetrahedronLocalPointEval(
                                local_coords, numNodesPerElem, shape_field)
                          : mesh_field.triangleLocalPointEval(
                                local_coords, numNodesPerElem, shape_field);

  int errors = 0;
  const auto triVerts = mesh.ask_elem_verts();
  Kokkos::parallel_reduce(
      "test", mesh.nelems(),
      KOKKOS_LAMBDA(const int &tri, int &errors) {
        for (int node = 0; node < vertexNodes; ++node) {
          const auto elemDim = (dim == 3 ? 3 : 2);
          const auto vtxDim = 0;
          const auto ignored = -1;
          const auto localVtxIdx =
              Omega_h::simplex_down_template(elemDim, vtxDim, node, ignored);
          const auto triToVtxDegree = Omega_h::simplex_degree(elemDim, vtxDim);
          int vtx = triVerts[(tri * triToVtxDegree) + localVtxIdx];
          auto x = coords[vtx * dim];
          auto y = coords[vtx * dim + 1];
          auto z = dim == 3 ? coords[vtx * dim + 2] : 0;
          auto expected = f(x, y, z);
          auto actual = eval_results(tri * numNodesPerElem + node, 0);
          if (Kokkos::fabs(expected - actual) > MeshField::MachinePrecision) {
            ++errors;
            Kokkos::printf(
                "expected: %lf, actual: %lf, element: %d, node(vtx): %d\n",
                expected, actual, tri, node);
          }
        }
        for (int node = vertexNodes; node < numNodesPerElem; ++node) {
          const auto elemDim = (dim == 3 ? 3 : 2);
          const auto edgeDim = 1;
          const auto triToEdgeDegree =
              Omega_h::simplex_degree(elemDim, edgeDim);
          const MeshField::LO triNode2DofHolder[6] = {0, 1, 2, 3, 4, 5};
          int edge = edgeMap[(tri * triToEdgeDegree) +
                             triNode2DofHolder[node - (dim == 3 ? 4 : 3)]];
          auto left = edge2vtx[edge * 2];
          auto right = edge2vtx[edge * 2 + 1];
          auto x = (coords[left * dim] + coords[right * dim]) / 2.0;
          auto y = (coords[left * dim + 1] + coords[right * dim + 1]) / 2.0;
          auto z =
              dim == 3
                  ? (coords[left * dim + 2] + coords[right * dim + 2]) / 2.0
                  : 0;
          auto expected = f(x, y, z);
          auto actual = eval_results(tri * numNodesPerElem + node, 0);
          if (Kokkos::fabs(expected - actual) > MeshField::MachinePrecision) {
            ++errors;
            Kokkos::printf(
                "expected: %lf, actual: %lf, element: %d, node(edge): %d\n",
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
    runTest<1, 2>(mesh);
    runTest<2, 2>(mesh);
    mesh =
        Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 10, false);
    runTest<1, 3>(mesh);
    runTest<2, 3>(mesh);
  }
  Kokkos::finalize();
  return 0;
}
