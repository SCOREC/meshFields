#include "KokkosController.hpp"
#include "MeshField.hpp"
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif
#include "Omega_h_build.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;

template <size_t dim, size_t order> auto getElement(Omega_h::Mesh &mesh) {
  if constexpr (dim == 3) {
    return MeshField::Omegah::getTetrahedronElement<order>(mesh);
  } else {
    return MeshField::Omegah::getTriangleElement<order>(mesh);
  }
}

template <size_t dim>
KOKKOS_INLINE_FUNCTION Real quadratic(Kokkos::Array<Real, dim> const &x) {
  Real f = 1;
  for (size_t d = 0; d < dim; ++d)
    f += (d + 1) * x[d] + x[d] * x[d];
  return f;
}

template <size_t dim>
KOKKOS_INLINE_FUNCTION Real linear(Kokkos::Array<Real, dim> const &x) {
  Real f = 1;
  for (size_t d = 0; d < dim; ++d)
    f += (d + 1) * x[d];
  return f;
}

/*
 * Interpolate an analytic function, with numComp components
 */
template <size_t dim, size_t order, size_t numComp,
          template <typename...> typename Controller>
bool testAnalytic(
    Omega_h::Mesh &mesh,
    MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf) {
  const auto [coordShp, coordMap] = getElement<dim, 1>(mesh);
  MeshField::FieldElement coordElm(mesh.nelems(), omf.getCoordField().field,
                                   coordShp, coordMap);
  auto fieldWithCtrlr =
      omf.template CreateLagrangeField<Real, order, numComp>();
  const auto [shp, map] = getElement<dim, order>(mesh);
  MeshField::FieldElement elm(mesh.nelems(), fieldWithCtrlr.field, shp, map);

  const auto f = KOKKOS_LAMBDA(Kokkos::Array<Real, dim> const &x) {
    return order == 1 ? linear<dim>(x) : quadratic<dim>(x);
  };
  MeshField::interpolate(
      elm, KOKKOS_LAMBDA(const int ent, Kokkos::Array<Real, dim> const &xi) {
        const auto x = coordElm.getValue(ent, xi);
        Kokkos::Array<Real, numComp> val;
        for (size_t c = 0; c < numComp; ++c)
          val[c] = (c + 1) * f(x);
        return val;
      });

  // points away from the nodes: the centroid and an off-center point
  constexpr size_t numPts = 2;
  Kokkos::View<Real **> lc("localCoords", mesh.nelems() * numPts, dim);
  Kokkos::parallel_for(
      "setLocalCoords", mesh.nelems(), KOKKOS_LAMBDA(const int ent) {
        for (size_t d = 0; d < dim; ++d) {
          lc(ent * numPts, d) = 1.0 / (dim + 1);
          lc(ent * numPts + 1, d) = 0.1 * (d + 1);
        }
      });
  const auto values = MeshField::evaluate(elm, lc, numPts);
  const auto x = MeshField::evaluate(coordElm, lc, numPts);
  MeshField::LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkAnalytic", values.extent(0),
      KOKKOS_LAMBDA(const int pt, MeshField::LO &lerrors) {
        Kokkos::Array<Real, dim> xp;
        for (size_t d = 0; d < dim; ++d)
          xp[d] = x(pt, d);
        for (size_t c = 0; c < numComp; ++c)
          if (Kokkos::fabs(values(pt, c) - (c + 1) * f(xp)) >
              MeshField::Epsilon)
            ++lerrors;
      },
      numErrors);
  const bool pass = numErrors == 0;
  std::cout << "testAnalytic(dim " << dim << ", order " << order << ", "
            << numComp << " components): "
            << (pass ? "pass" : std::to_string(numErrors) + " errors FAIL")
            << "\n";
  return pass;
}

/*
 * Write source(ent, xi) = ent. If ownership is working, we should have the
 * neighboring element with the smallest id listed on every dof holder thats
 * is on a shared entity
 */
template <size_t dim, size_t order, template <typename...> typename Controller>
bool testOwnership(
    Omega_h::Mesh &mesh,
    MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> &omf) {
  auto fieldWithCtrlr = omf.template CreateLagrangeField<Real, order, 1>();
  auto field = fieldWithCtrlr.field;
  const auto [shp, map] = getElement<dim, order>(mesh);
  MeshField::FieldElement elm(mesh.nelems(), field, shp, map);
  MeshField::interpolate(
      elm, KOKKOS_LAMBDA(const int ent, Kokkos::Array<Real, dim> const &) {
        return Kokkos::Array<Real, 1>{static_cast<Real>(ent)};
      });

  MeshField::LO numErrors = 0;
  // the dof holders are the vertices and, for order 2, the edges
  for (int holderDim = 0; holderDim < order; ++holderDim) {
    const auto topo = holderDim == 0 ? MeshField::Vertex : MeshField::Edge;
    const auto up = mesh.ask_up(holderDim, dim);
    const auto a2ab = up.a2ab;
    const auto ab2b = up.ab2b;
    MeshField::LO lerrorsDim = 0;
    Kokkos::parallel_reduce(
        "checkOwnership", mesh.nents(holderDim),
        KOKKOS_LAMBDA(const int e, MeshField::LO &lerrors) {
          auto lowest = ab2b[a2ab[e]];
          for (auto i = a2ab[e] + 1; i < a2ab[e + 1]; ++i)
            lowest = Kokkos::min(lowest, ab2b[i]);
          if (field(e, 0, 0, topo) != lowest)
            ++lerrors;
        },
        lerrorsDim);
    numErrors += lerrorsDim;
  }
  const bool pass = numErrors == 0;
  std::cout << "testOwnership(dim " << dim << ", order " << order << "): "
            << (pass ? "pass" : std::to_string(numErrors) + " errors FAIL")
            << "\n";
  return pass;
}

template <size_t dim, template <typename...> typename Controller>
int runTests(Omega_h::Mesh &mesh) {
  MeshField::OmegahMeshField<ExecutionSpace, dim, Controller> omf(mesh);
  int failed = 0;
  failed += !testOwnership<dim, 1>(mesh, omf);
  failed += !testOwnership<dim, 2>(mesh, omf);
  failed += !testAnalytic<dim, 1, 1>(mesh, omf);
  failed += !testAnalytic<dim, 2, 1>(mesh, omf);
  failed += !testAnalytic<dim, 2, 3>(mesh, omf);
  return failed;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    auto lib = Omega_h::Library(&argc, &argv);
    auto tris = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0,
                                   3, 3, 0);
    auto tets = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 2.0, 3.0,
                                   2, 2, 2);
    failed += runTests<2, MeshField::KokkosController>(tris);
    failed += runTests<3, MeshField::KokkosController>(tets);
#ifdef MESHFIELDS_ENABLE_CABANA
    failed += runTests<2, MeshField::CabanaController>(tris);
    failed += runTests<3, MeshField::CabanaController>(tets);
#endif
  }
  Kokkos::finalize();
  if (failed) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
