#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"    //remove?
#include "MeshField_Fail.hpp"       //remove?
#include "MeshField_For.hpp"        //remove?
#include "MeshField_ShapeField.hpp" //remove?
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <MeshField_Integrate.hpp>
#include <iostream>
#include <sstream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

template <typename FieldElement>
class testIntegrator : public MeshField::Integrator {
private:
  testIntegrator(){};

protected:
  MeshField::Real integral;
  FieldElement &fes;
  int m;
  int n;
  int l;

public:
  MeshField::Real getIntegral() { return integral; }
  testIntegrator(FieldElement &fes_in, int _m, int _n, int _l, int order)
      : Integrator(order), integral(0), fes(fes_in), m(_m), n(_n), l(_l){};
  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) {
    const auto globalCoords = MeshField::evaluate(fes, p, p.extent(0));
    const int _m = m;
    const int _n = n;
    const int _l = l;
    Kokkos::parallel_reduce(
        "integrate", p.extent(0),
        KOKKOS_LAMBDA(const int &ent, MeshField::Real &integ) {
          const auto x = globalCoords(ent, 0);
          const auto y = globalCoords(ent, 1);
          const auto z = globalCoords.extent(1) == 3 ? globalCoords(ent, 2) : 1;
          integ += w(ent) * pow(x, _m) * pow(y, _n) * pow(z, _l) * dV(ent);
        },
        integral);
  }
};

template <template <typename...> typename Controller, size_t ShapeOrder,
          size_t dim>
void doRun(Omega_h::Mesh &mesh,
           MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf) {
  auto field =
      omf.template CreateLagrangeField<MeshField::Real, ShapeOrder, dim>();
  auto coords = mesh.coords();
  Kokkos::parallel_for(
      mesh.nverts(), KOKKOS_LAMBDA(int vtx) {
        field(vtx, 0, 0, MeshField::Vertex) = coords[vtx * dim];
        field(vtx, 0, 1, MeshField::Vertex) = coords[vtx * dim + 1];
        if constexpr (dim == 3) {
          field(vtx, 0, 2, MeshField::Vertex) = coords[vtx * dim + 2];
        }
      });
  if (ShapeOrder == 2) {
    auto edge2vtx = mesh.ask_down(1, 0).ab2b;
    auto edgeMap = mesh.ask_down(dim, 1).ab2b;
    Kokkos::parallel_for(
        mesh.nedges(), KOKKOS_LAMBDA(int edge) {
          const auto left = edge2vtx[edge * 2];
          const auto right = edge2vtx[edge * 2 + 1];
          const auto x = (coords[left * dim] + coords[right * dim]) / 2.0;
          const auto y =
              (coords[left * dim + 1] + coords[right * dim + 1]) / 2.0;
          field(edge, 0, 0, MeshField::Edge) = x;
          field(edge, 0, 1, MeshField::Edge) = y;
          if constexpr (dim == 3) {
            const auto z =
                (coords[left * dim + 2] + coords[right * dim + 2]) / 2.0;
            field(edge, 0, 2, MeshField::Edge) = z;
          }
        });
  }

  auto shapeSet = [&]() {
    if constexpr (dim == 3) {
      return MeshField::Omegah::getTetrahedronElement<ShapeOrder>(mesh);
    } else {
      return MeshField::Omegah::getTriangleElement<ShapeOrder>(mesh);
    }
  };
  const auto [shp, map] = shapeSet();
  MeshField::FieldElement fes(1, field, shp, map);
  const int maxn = 32;
  int binom[maxn + 1][maxn + 1];
  for (int n = 0; n <= maxn; n++) {
    binom[n][0] = binom[n][n] = 1;
    for (int k = 1; k < n; k++) {
      binom[n][k] = binom[n - 1][k] + binom[n - 1][k - 1];
    }
  }
  if (dim == 2) {
    for (int p = 1; p <= ShapeOrder; ++p) {
      for (int m = p; m >= 0; --m) {
        int n = p - m;
        testIntegrator testInt(fes, m, n, 0, p);
        testInt.process(fes);
        double exact = 1.0 / binom[p][m] / (p + 1) / (p + 2);
        if (Kokkos::fabs(testInt.getIntegral() - exact) >
            MeshField::MachinePrecision) {
          std::stringstream ss;
          ss << "Integration over "
             << ((dim == 3) ? "Tetrahedron " : "Triangle ") << "with order "
             << ShapeOrder << " failed\n";
          MeshField::fail(ss.str());
        }
      }
    }
  } else {
    for (int p = 1; p <= ShapeOrder; ++p) {
      for (int l = p; l >= 0; --l) {
        for (int m = p - l; m >= 0; --m) {
          int n = p - l - m;
          testIntegrator testInt(fes, l, m, n, p);
          testInt.process(fes);
          double exact = 1.0 / binom[p][l + m] / binom[l + m][l] / (p + 1) /
                         (p + 2) / (p + 3);
          if (Kokkos::fabs(testInt.getIntegral() - exact) >
              MeshField::MachinePrecision) {
            std::stringstream ss;
            ss << "Integration over "
               << ((dim == 3) ? "Tetrahedron " : "Triangle ") << "with order "
               << ShapeOrder << " failed\n";
            MeshField::fail(ss.str());
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
#ifdef MESHFIELDS_ENABLE_CABANA
  {
    Omega_h::Reals coords2D({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});
    Omega_h::Reals coords3D(
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

    Omega_h::LOs tris_to_verts({0, 1, 2});
    Omega_h::LOs tets_to_verts({3, 0, 1, 2});
    Omega_h::Mesh mesh2D(&lib);
    Omega_h::Mesh mesh3D(&lib);
    Omega_h::build_from_elems_and_coords(&mesh2D, OMEGA_H_SIMPLEX, 2,
                                         tris_to_verts, coords2D);
    Omega_h::build_from_elems_and_coords(&mesh3D, OMEGA_H_SIMPLEX, 3,
                                         tets_to_verts, coords3D);

    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::CabanaController>
        omf2D(mesh2D);
    doRun<MeshField::CabanaController, 1>(mesh2D, omf2D);
    doRun<MeshField::CabanaController, 2>(mesh2D, omf2D);
    MeshField::OmegahMeshField<ExecutionSpace, 3, MeshField::CabanaController>
        omf3D(mesh3D);
    doRun<MeshField::CabanaController, 1>(mesh3D, omf3D);
    doRun<MeshField::CabanaController, 2>(mesh3D, omf3D);
  }
#endif
  {
    Omega_h::Reals coords2D({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});
    Omega_h::Reals coords3D(
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

    Omega_h::LOs tris_to_verts({0, 1, 2});
    Omega_h::LOs tets_to_verts({3, 0, 1, 2});
    Omega_h::Mesh mesh2D(&lib);
    Omega_h::Mesh mesh3D(&lib);
    Omega_h::build_from_elems_and_coords(&mesh2D, OMEGA_H_SIMPLEX, 2,
                                         tris_to_verts, coords2D);
    Omega_h::build_from_elems_and_coords(&mesh3D, OMEGA_H_SIMPLEX, 3,
                                         tets_to_verts, coords3D);

    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::KokkosController>
        omf2D(mesh2D);
    doRun<MeshField::KokkosController, 1>(mesh2D, omf2D);
    doRun<MeshField::KokkosController, 2>(mesh2D, omf2D);
    MeshField::OmegahMeshField<ExecutionSpace, 3, MeshField::KokkosController>
        omf3D(mesh3D);
    doRun<MeshField::KokkosController, 1>(mesh3D, omf3D);
    doRun<MeshField::KokkosController, 2>(mesh3D, omf3D);
  }
  Kokkos::finalize();
  return 0;
}
