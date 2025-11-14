#include "MeshField.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Integrate.hpp"
#include "MeshField_Shape.hpp"
#include "KokkosController.hpp"
#include "Kokkos_Core.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_simplex.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

inline double analyticFunction(double x, double y, double z = 0.0) {
    return 1.0 + x + y + z;
}

const MeshField::MeshInfo meshInfo{.numVtx = 5, .numTri = 3, .numTet = 1};

template <size_t dim>
Omega_h::Mesh createMesh(Omega_h::Library &lib) {
    auto world = lib.world();
    const auto family = OMEGA_H_SIMPLEX;
    auto len = 1.0;
    const auto numEnts3d = (dim == 3 ? 3 : 0);
    return Omega_h::build_box(world, family, len, len, len, 3, 3, numEnts3d);
}

template <template <typename...> typename Controller, size_t dim>
void doRun(Omega_h::Mesh &mesh) {
    Omega_h::Reals coords = mesh.coords();

    auto coordField = MeshField::CreateCoordinateField<ExecutionSpace, Controller, dim>(meshInfo);

    auto field = MeshField::CreateCoordinateField<ExecutionSpace, Controller, dim>(meshInfo);
    
    for (int i = 0; i < meshInfo.numVtx; ++i) {
        double x = coords[i*dim + 0];
        double y = coords[i*dim + 1];
        double z = (dim == 3 ? coords[i*dim + 2] : 0.0);
        field(i, 0, 0, MeshField::Vertex) = analyticFunction(x, y, z);
    }

    Kokkos::parallel_for(meshInfo.numVtx, KOKKOS_LAMBDA(const int &i){
        for (size_t d = 0; d < dim; ++d)
            coordField(i, 0, d, MeshField::Vertex) = coords[i*dim + d];
    });

    MeshField::ReducedQuinticImplicitShape integrator;

    auto elemShape = MeshField::Omegah::getTriangleElement<1>(mesh);
    auto &shp = elemShape.shp;
    auto &map = elemShape.map;
    MeshField::FieldElement fes(meshInfo.numTri, coordField, shp, map);

    MeshField::Vector3 xi;
    xi[0] = 1.0/3.0;
    xi[1] = 1.0/3.0;
    xi[2] = 1.0/3.0;

    auto vals = integrator.getValues(xi);
    double sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) sum += vals[i];

    double val = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        val += vals[i] / sum * field(i, 0, 0, MeshField::Vertex);
    }

    double expected = analyticFunction(xi[0], xi[1]);
    assert(std::fabs(val - expected) < 1e-6);
}

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    Omega_h::Library lib(&argc, &argv);

    {
        auto mesh2D = createMesh<2>(lib);
        doRun<MeshField::KokkosController,2>(mesh2D);
    }

    Kokkos::finalize();
    return 0;
}

