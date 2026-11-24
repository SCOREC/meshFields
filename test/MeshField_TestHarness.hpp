#ifndef MESHFIELD_TESTHARNESS_HPP
#define MESHFIELD_TESTHARNESS_HPP

/**
 * @file MeshField_TestHarness.hpp
 * @brief Catch2 test configuration, mesh construction, and device helpers
 *
 * Kokkos and Omega_h are initialized once in catchMain.cpp. Controller
 * policies and mesh configurations are ordinary types, shared through Catch2
 * type lists. Each configuration is independently registered with Catch2.
 *
 * nvcc does not allow a KOKKOS_LAMBDA in a Catch2 test case body (a function
 * in an anonymous namespace), so device code goes in named test functions or
 * functors.
 */

#include "MeshField_TestCommon.hpp"
#include "Omega_h_build.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <tuple>

/** @brief the Omega_h library created in main */
Omega_h::Library &library();

/** @brief the unit square of triangles or the 1x2x3 box of tetrahedra */
template <size_t dim> Omega_h::Mesh buildMesh() {
  static_assert(dim == 2 || dim == 3);
  if constexpr (dim == 2)
    return Omega_h::build_box(library().world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0,
                              3, 3, 0);
  else
    return Omega_h::build_box(library().world(), OMEGA_H_SIMPLEX, 1.0, 2.0, 3.0,
                              2, 2, 2);
}

// Preserve debug validation without leaking a test's setting into later cases.
struct ScopedMeshFieldDebug {
  bool previous = MeshField::Debug;
  ScopedMeshFieldDebug() { MeshField::Debug = true; }
  ~ScopedMeshFieldDebug() { MeshField::Debug = previous; }
};

// Controller policies avoid passing template-template parameters through
// Catch2's signature macros (and needing an unused execution-space argument).
struct KokkosBackend {
  template <typename... Ts>
  using Controller = MeshField::KokkosController<Ts...>;
};
struct NoParameters {};
template <typename Backend, size_t D, typename Parameters = NoParameters>
struct MeshCase : Backend, Parameters {
  static constexpr size_t dim = D;
};
template <typename Parameters = NoParameters>
using Kokkos2D = MeshCase<KokkosBackend, 2, Parameters>;
template <typename Parameters = NoParameters>
using Kokkos3D = MeshCase<KokkosBackend, 3, Parameters>;
#ifdef MESHFIELDS_ENABLE_CABANA
struct CabanaBackend {
  template <typename... Ts>
  using Controller = MeshField::CabanaController<Ts...>;
};
template <typename Parameters = NoParameters>
using Cabana2D = MeshCase<CabanaBackend, 2, Parameters>;
template <typename Parameters = NoParameters>
using Cabana3D = MeshCase<CabanaBackend, 3, Parameters>;
using ControllerCases = std::tuple<KokkosBackend, CabanaBackend>;
using MeshCases = std::tuple<Kokkos2D<>, Kokkos3D<>, Cabana2D<>, Cabana3D<>>;
#define MESHFIELD_MESH_CASE_TEMPLATES Kokkos2D, Kokkos3D, Cabana2D, Cabana3D
#else
using ControllerCases = std::tuple<KokkosBackend>;
using MeshCases = std::tuple<Kokkos2D<>, Kokkos3D<>>;
#define MESHFIELD_MESH_CASE_TEMPLATES Kokkos2D, Kokkos3D
#endif

#endif
