#ifndef MESHFIELD_TESTHARNESS_HPP
#define MESHFIELD_TESTHARNESS_HPP

/**
 * @file MeshField_TestHarness.hpp
 * @brief a test runner, reporting, and integrators shared by the tests
 *
 * A test's main creates a TestRunner and loops over controllers, meshes, and
 * types (e.g., element factories) with its forEach functions.  The loop
 * bodies are template lambdas:
 *
 * @code
 * TestRunner t(argc, argv);
 * t.forEachController([&]<template <typename...> typename Controller>() {
 *   t.forEachMesh([&]<size_t dim>(Omega_h::Mesh &mesh) {
 *     testSomething<dim, Controller>(mesh);
 *   });
 * });
 * return t.exitCode();
 * @endcode
 *
 * Tests report each check with report(), which prefixes the controller and
 * mesh being run and counts the failures for exitCode().
 *
 * nvcc does not allow a KOKKOS_LAMBDA inside a template lambda, so device code
 * goes in named test functions or functors, not in the loop bodies.
 */

#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Integrate.hpp"
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif
#include "Omega_h_build.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <vector>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

namespace detail {
/** the controller, mesh, etc. being run, printed by report() */
inline std::vector<std::string> context;
inline int numFailed = 0;

/** adds a label to the context for its lifetime */
struct ContextScope {
  ContextScope(std::string const &label) { context.push_back(label); }
  ~ContextScope() { context.pop_back(); }
};
} // namespace detail

/**
 * @brief print the result of a check, prefixed by the context, and count it
 * if it failed
 * @return true if the check passed
 */
inline bool report(std::string const &name, MeshField::LO numErrors) {
  const bool pass = numErrors == 0;
  if (!detail::context.empty()) {
    std::cout << "[";
    for (size_t i = 0; i < detail::context.size(); ++i)
      std::cout << (i ? ", " : "") << detail::context[i];
    std::cout << "] ";
  }
  std::cout << name << ": "
            << (pass ? "pass" : std::to_string(numErrors) + " errors FAIL")
            << "\n";
  if (!pass)
    ++detail::numFailed;
  return pass;
}

/** @brief call f.template operator()<T>() for each T */
template <typename... Ts, typename F> void forEachType(F &&f) {
  (f.template operator()<Ts>(), ...);
}

/**
 * @brief initializes Kokkos and Omega_h for its lifetime, and runs tests with
 * each controller and mesh
 */
class TestRunner {
public:
  TestRunner(int &argc, char **&argv)
      : kokkos(argc, argv), lib(&argc, &argv) {}

  Omega_h::Library &library() { return lib; }

  /**
   * @brief call f.template operator()<Controller>() with the Kokkos
   * controller and, if enabled, the Cabana controller
   */
  template <typename F> void forEachController(F &&f) {
    {
      detail::ContextScope scope("Kokkos");
      f.template operator()<MeshField::KokkosController>();
    }
#ifdef MESHFIELDS_ENABLE_CABANA
    {
      detail::ContextScope scope("Cabana");
      f.template operator()<MeshField::CabanaController>();
    }
#endif
  }

  /**
   * @brief call f.template operator()<dim>(mesh) with the unit square of
   * triangles and the 1x2x3 box of tetrahedra
   */
  template <typename F> void forEachMesh(F &&f) {
    {
      detail::ContextScope scope("2d tris");
      auto tris = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0,
                                     0.0, 3, 3, 0);
      f.template operator()<2>(tris);
    }
    {
      detail::ContextScope scope("3d tets");
      auto tets = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 2.0,
                                     3.0, 2, 2, 2);
      f.template operator()<3>(tets);
    }
  }

  /** @brief print a summary; the exit code for main */
  int exitCode() const {
    if (detail::numFailed) {
      std::cerr << detail::numFailed << " check(s) failed\n";
      return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
  }

private:
  Kokkos::ScopeGuard kokkos;
  Omega_h::Library lib;
};

/** @brief compute the measure (3D volume, 2D area, 1D length) */
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

#endif
