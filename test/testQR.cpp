#include <KokkosBatched_QR_Decl.hpp>     //KokkosBlas::QR
#include <KokkosBatched_Util.hpp>        //KokkosBlas::Algo
#include <Kokkos_Core.hpp>

void testQR() {
  typedef Kokkos::View<double[16][10]> MatrixViewType;
  typedef Kokkos::View<double[10]> ColVectorViewType;
  typedef Kokkos::View<double[10]> ColWorkViewType;

  MatrixViewType A("A");
  ColVectorViewType t("t");
  ColWorkViewType w("w");

  // roughly following
  // kokkos-kernels/batched/dense/unit_test/Test_Batched_TeamVectorQR.hpp
  typedef KokkosBlas::Algo::QR::Unblocked AlgoTagType;
  Kokkos::parallel_for("serialQR", 1, KOKKOS_LAMBDA(int) {
        // compute the QR factorization of A and store the results in A and t
        // (tau) - see the lapack dgeqp3(...) documentation:
        // www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga1b0500f49e03d2771b797c6e88adabbb.html
        KokkosBatched::SerialQR<AlgoTagType>::invoke(A, t, w);
      });
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard scope_gaurd(argc, argv);
  testQR();
}
