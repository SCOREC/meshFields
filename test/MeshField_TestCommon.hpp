#ifndef MESHFIELD_TESTCOMMON_HPP
#define MESHFIELD_TESTCOMMON_HPP

/**
 * @file MeshField_TestCommon.hpp
 * @brief includes, execution space, and integrators shared by the tests
 */

#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Integrate.hpp"
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif
#include <Kokkos_Core.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

/** @brief compute the measure (3D volume, 2D area, 1D length) */
class MeasureIntegrator : public MeshField::Integrator {
public:
  MeasureIntegrator(int order) : MeshField::Integrator(order) {}
  void atPoints(Kokkos::View<MeshField::Real **>,
                Kokkos::View<MeshField::Real *> w,
                Kokkos::View<MeshField::Real *> dV) override {
    MeshField::Real sum = 0;
    Kokkos::parallel_reduce(
        "integrateMeasure", dV.extent(0),
        KOKKOS_LAMBDA(const int i, MeshField::Real &lsum) {
          lsum += w(i % w.extent(0)) * dV(i);
        },
        sum);
    measure += sum;
  }
  MeshField::Real measure = 0;
};

#endif
