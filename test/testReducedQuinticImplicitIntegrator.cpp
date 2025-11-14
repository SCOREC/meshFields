#include "MeshField.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Shape.hpp"
#include "KokkosController.hpp"

#include "Kokkos_Core.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_simplex.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

inline double analytic(double x, double y) { return 1.0 + x + y; }

Omega_h::Mesh create_mesh(Omega_h::Library& lib) {
    auto world = lib.world();
    return Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 1, 1, 0);
}

template <template <typename...> typename Controller, size_t dim>
void run_test(Omega_h::Mesh& mesh) {
    MeshField::MeshInfo info{};
    info.numVtx = mesh.nverts();
    info.numTri = (dim==2 ? mesh.nfaces() : 0);
    info.numTet = (dim==3 ? mesh.nregions() : 0);

    auto coords = mesh.coords();
    auto elemVerts = mesh.ask_elem_verts();

    int v0 = elemVerts[0];
    int v1 = elemVerts[1];
    int v2 = elemVerts[2];

    auto field = MeshField::CreateLagrangeField<ExecutionSpace, Controller, double, 1, dim, 1>(info);

    for (int i = 0; i < info.numVtx; ++i) {
        double x = coords[i*dim + 0];
        double y = coords[i*dim + 1];
        field(i, 0, 0, MeshField::Vertex) = analytic(x, y);
    }

    MeshField::ReducedQuinticImplicitShape shape;
    MeshField::Vector3 xi;
    xi[0] = 1.0/3.0; xi[1] = 1.0/3.0; xi[2] = 1.0/3.0;

    auto N = shape.getValues(xi);

    double approx = 0.0;
    approx += N[0] * field(v0, 0, 0, MeshField::Vertex);
    approx += N[1] * field(v1, 0, 0, MeshField::Vertex);
    approx += N[2] * field(v2, 0, 0, MeshField::Vertex);

    double xb = (coords[v0*dim+0] + coords[v1*dim+0] + coords[v2*dim+0]) / 3.0;
    double yb = (coords[v0*dim+1] + coords[v1*dim+1] + coords[v2*dim+1]) / 3.0;
    double exact = analytic(xb, yb);

    std::cout<<"approx: "<<approx<<"exact: "<<exact<<std::endl;
    assert(std::fabs(approx - exact) < 1e-6);
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    Omega_h::Library lib(&argc, &argv);

    auto mesh = create_mesh(lib);
    run_test<MeshField::KokkosController, 2>(mesh);

    Kokkos::finalize();
    return 0;
}

