#include <KokkosBatched_ApplyQ_Decl.hpp> //KokkosBlas::ApplyQ
#include <KokkosBatched_QR_Decl.hpp>     //KokkosBlas::QR
#include <KokkosBatched_Trsv_Decl.hpp>   //KokkosBlas::Trsv
#include <KokkosBatched_Util.hpp>        //KokkosBlas::Algo, KokkosBatched::Diag
#include <KokkosBlas2_gemv.hpp>          //KokkosBlas::gemv
#include <Kokkos_Core.hpp>

/* here is a test case run with Octave */
static double const a_data[16][10] = {
    {1.0000000e+00, -4.9112749e-02, -1.5629814e+00, -8.2662407e-02,
     7.6762316e-02, 1.2919981e-01, 4.0597781e-03, 2.4120621e-03, 2.4429110e+00,
     6.8330735e-03},
    {1.0000000e+00, -7.4718700e-01, 1.1447982e+00, -6.1608208e-01,
     -8.5537834e-01, -7.0528966e-01, 4.6032852e-01, 5.5828842e-01,
     1.3105629e+00, 3.7955713e-01},
    {1.0000000e+00, -4.8564839e-01, -7.2143765e-01, -5.3574860e-02,
     3.5036503e-01, 3.8650921e-02, 2.6018545e-02, 2.3585436e-01, 5.2047228e-01,
     2.8702657e-03},
    {1.0000000e+00, -8.1013248e-01, -1.0234062e+00, 3.4345012e-01,
     8.2909460e-01, -3.5148898e-01, -2.7824010e-01, 6.5631463e-01,
     1.0473602e+00, 1.1795799e-01},
    {1.0000000e+00, -1.0508609e+00, -8.6973926e-01, 1.4570417e+00,
     9.1397495e-01, -1.2672464e+00, -1.5311481e+00, 1.1043086e+00,
     7.5644638e-01, 2.1229706e+00},
    {1.0000000e+00, -1.7802012e-01, 1.8697947e+00, 8.3576559e-01,
     -3.3286107e-01, 1.5627100e+00, -1.4878309e-01, 3.1691163e-02,
     3.4961320e+00, 6.9850413e-01},
    {1.0000000e+00, -6.4594368e-01, -5.5999878e-01, 2.5848452e+00,
     3.6172768e-01, -1.4475102e+00, -1.6696645e+00, 4.1724324e-01,
     3.1359864e-01, 6.6814250e+00},
    {1.0000000e+00, -3.0007589e-01, -5.6481843e-01, -3.0670467e-01,
     1.6948839e-01, 1.7323245e-01, 9.2034678e-02, 9.0045542e-02, 3.1901985e-01,
     9.4067754e-02},
    {1.0000000e+00, 1.8896909e-01, 2.7899549e-01, -2.5907897e-01, 5.2721524e-02,
     -7.2281865e-02, -4.8957918e-02, 3.5709317e-02, 7.7838484e-02,
     6.7121914e-02},
    {1.0000000e+00, 1.8586762e+00, -4.5760801e-01, -9.5892373e-02,
     -8.5054510e-01, 4.3881118e-02, -1.7823287e-01, 3.4546770e+00,
     2.0940510e-01, 9.1953472e-03},
    {1.0000000e+00, -1.0915266e-01, 1.9050743e+00, 3.2218623e-01,
     -2.0794394e-01, 6.1378871e-01, -3.5167484e-02, 1.1914304e-02,
     3.6293082e+00, 1.0380396e-01},
    {1.0000000e+00, -3.2178312e-01, -6.9392454e-01, -6.1112393e-01,
     2.2329320e-01, 4.2407389e-01, 1.9664936e-01, 1.0354437e-01, 4.8153127e-01,
     3.7347245e-01},
    {1.0000000e+00, 1.4229431e+00, 3.8000845e-01, -1.5989849e-01, 5.4073041e-01,
     -6.0762776e-02, -2.2752645e-01, 2.0247671e+00, 1.4440643e-01,
     2.5567526e-02},
    {1.0000000e+00, 9.0145077e-01, -9.6991898e-01, 7.9086551e-01,
     -8.7433421e-01, -7.6707547e-01, 7.1292632e-01, 8.1261349e-01,
     9.4074284e-01, 6.2546825e-01},
    {1.0000000e+00, -7.2533533e-01, 1.4790603e-01, 7.7147564e-01,
     -1.0728147e-01, 1.1410590e-01, -5.5957854e-01, 5.2611134e-01,
     2.1876193e-02, 5.9517466e-01},
    {1.0000000e+00, 5.3974943e-01, -7.7853625e-01, -1.2196455e+00,
     -4.2021450e-01, 9.4953820e-01, -6.5830294e-01, 2.9132945e-01,
     6.0611869e-01, 1.4875350e+00},
};

static double const x_data[10] = {
    4.4283983e-01,  -2.9416715e-01, 2.5654724e-01,  3.3493879e-01,
    -6.3165447e-01, -5.1360652e-01, -2.4971659e-01, -2.5776427e-01,
    -8.8432424e-02, -5.2756825e-01};

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using DeviceType = ExecutionSpace::device_type;

// based on pumi (github.com/SCOREC/core) test/qr.cc
static void testSolveQR() {
  typedef Kokkos::View<double[16][10], Kokkos::LayoutLeft, DeviceType>
      MatrixViewType;
  typedef Kokkos::View<double[16], Kokkos::LayoutLeft, DeviceType>
      RowVectorViewType;
  typedef Kokkos::View<double[10], Kokkos::LayoutLeft, DeviceType>
      ColVectorViewType;
  typedef Kokkos::View<double[10], Kokkos::LayoutRight, DeviceType>
      ColWorkViewType;
  const auto one = 1.0;
  const auto zero = 0.0;

  MatrixViewType A("A");
  typename MatrixViewType::HostMirror A_host = Kokkos::create_mirror_view(A);
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 10; j++)
      A_host(i, j) = a_data[i][j];
  Kokkos::deep_copy(A, A_host);

  ColVectorViewType kx("kx");
  typename ColVectorViewType::HostMirror kx_host =
      Kokkos::create_mirror_view(kx);
  for (int j = 0; j < 10; j++)
    kx_host(j) = x_data[j];
  Kokkos::deep_copy(kx, kx_host);

  // b = A*kx
  RowVectorViewType b("b");
  KokkosBlas::gemv("N", one, A, kx, zero, b);

  // x = b
  RowVectorViewType x("x");
  Kokkos::deep_copy(x, b);

  // t: tau (see SerialQR call below)
  ColVectorViewType t("t");

  // w: working space for SerialQR
  ColWorkViewType w("w");

  // roughly following
  // kokkos-kernels/batched/dense/unit_test/Test_Batched_TeamVectorQR.hpp
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  Kokkos::parallel_for(
      "solveQR", 1, KOKKOS_LAMBDA(int) {
        // compute the QR factorization of A and store the results in A and t
        // (tau) - see the lapack dgeqp3(...) documentation:
        // www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga1b0500f49e03d2771b797c6e88adabbb.html
        KokkosBatched::SerialQR<AlgoTagType>::invoke(A, t, w);
        // x = Q^{T}x
        KokkosBatched::SerialApplyQ<
            KokkosBatched::Side::Left, KokkosBatched::Trans::Transpose,
            KokkosBlas::Algo::ApplyQ::Unblocked>::invoke(A, t, x, w);
        // x = R^{-1}x
        KokkosBatched::SerialTrsv<
            KokkosBatched::Uplo::Upper, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Diag::NonUnit,
            KokkosBlas::Algo::Trsv::Unblocked>::invoke(one, A, x);
      });
  Kokkos::parallel_for(
      "compare", kx.size(), KOKKOS_LAMBDA(int i) {
        auto wrong = Kokkos::fabs(kx(i) - x(i)) > 1e-15;
        if (wrong)
          Kokkos::printf("kx(%d) %f != x(%d) %f\n", i, kx(i), i, x(i));
        KOKKOS_ASSERT(!wrong);
      });
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard scope_gaurd(argc, argv);
  testSolveQR();
}
