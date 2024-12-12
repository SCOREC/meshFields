#include <Cabana_Core.hpp>
#include "MeshField_Utility.hpp"

namespace {
  template <class Fn, typename ExecutionSpace, size_t VecLen>
    typename std::enable_if<1 == MeshFieldUtil::function_traits<Fn>::arity>::type
    simd_for(Fn &kernel, const Kokkos::Array<int64_t, 1> &start,
        const Kokkos::Array<int64_t, 1> &end, std::string tag) {
      Cabana::SimdPolicy<VecLen, ExecutionSpace> policy(start[0], end[0]);
      Cabana::simd_parallel_for(
          policy,
          KOKKOS_LAMBDA(const int &s, const int &a) {
            const std::size_t i = Cabana::Impl::Index<VecLen>::i(s, a);
            kernel(i);
          }, tag);
    }

  template <class Fn, typename ExecutionSpace, size_t VecLen>
    typename std::enable_if<2 == MeshFieldUtil::function_traits<Fn>::arity>::type
    simd_for(Fn &kernel, const Kokkos::Array<int64_t, 2> &start,
        const Kokkos::Array<int64_t, 2> &end, std::string tag) {
      Cabana::SimdPolicy<VecLen, ExecutionSpace> policy(start[0], end[0]);
      const int64_t s1 = start[1];
      const int64_t e1 = end[1];
      Cabana::simd_parallel_for(
          policy,
          KOKKOS_LAMBDA(const int &s, const int &a) {
          const std::size_t i = Cabana::Impl::Index<VecLen>::i(s, a);
          for (int j = s1; j < e1; j++)
            kernel(i, j);
          }, tag);
    }

  template <class Fn, typename ExecutionSpace, size_t VecLen>
    typename std::enable_if<3 == MeshFieldUtil::function_traits<Fn>::arity>::type
    simd_for(Fn &kernel, const Kokkos::Array<int64_t, 3> &start,
        const Kokkos::Array<int64_t, 3> &end, std::string tag) {
      Cabana::SimdPolicy<VecLen, ExecutionSpace> policy(start[0], end[0]);
      const int s1 = start[1], s2 = start[2];
      const int e1 = end[1], e2 = end[2];
      Cabana::simd_parallel_for(
          policy,
          KOKKOS_LAMBDA(const int &s, const int &a) {
            const std::size_t i = Cabana::Impl::Index<VecLen>::i(s, a);
            for (int j = s1; j < e1; j++) {
              for (int k = s2; k < e2; k++) {
              kernel(i, j, k);
            }
          }
          }, tag);
    }

  template <class Fn, typename ExecutionSpace, size_t VecLen>
    typename std::enable_if<4 == MeshFieldUtil::function_traits<Fn>::arity>::type
    simd_for(Fn &kernel, const Kokkos::Array<int64_t, 4> &start,
        const Kokkos::Array<int64_t, 4> &end, std::string tag) {
      Cabana::SimdPolicy<VecLen, ExecutionSpace> policy(start[0], end[0]);
      const int s1 = start[1], s2 = start[2], s3 = start[3];
      const int e1 = end[1], e2 = end[2], e3 = end[3];
      Cabana::simd_parallel_for(
          policy,
          KOKKOS_LAMBDA(const int &s, const int &a) {
            const std::size_t i = Cabana::Impl::Index<VecLen>::i(s, a);
            for (int j = s1; j < e1; j++) {
              for (int k = s2; k < e2; k++) {
                for (int l = s3; l < e3; l++) {
                kernel(i, j, k, l);
                }
              }
            }
          }, tag);
    }
}

namespace MeshField {
  template <typename CabController, typename FunctorType, class IS, class IE>
    void simd_parallel_for(CabController, const std::initializer_list<IS> &start_init,
        const std::initializer_list<IE> &end_init,
        FunctorType &vectorKernel, std::string tag) {
      static_assert(std::is_integral<IS>::value, "Integral required\n");
      static_assert(std::is_integral<IE>::value, "Integral required\n");
      constexpr std::size_t RANK =
        MeshFieldUtil::function_traits<FunctorType>::arity;
      assert(start_init.size() >= RANK);
      assert(end_init.size() >= RANK);
      Kokkos::Array<int64_t, RANK> a_start =
        MeshFieldUtil::to_kokkos_array<RANK>(start_init);
      Kokkos::Array<int64_t, RANK> a_end =
        MeshFieldUtil::to_kokkos_array<RANK>(end_init);
      using VecLen = CabController::vecLen;
      using ExecutionSpace = CabController::exe;
      simd_for<FunctorType, ExecutionSpace, VecLen>(vectorKernel, a_start, a_end, tag);
    }
}

