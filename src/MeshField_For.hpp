#ifndef MESHFIELD_FOR_HPP
#define MESHFIELD_FOR_HPP

#include "MeshField_Utility.hpp"
#include <Kokkos_Core.hpp>
#include "KokkosController.hpp"
#ifdef MESHFIELDS_ENABLE_CABANA
#include "MeshField_SimdFor.hpp"
#endif
namespace MeshField {
template <typename T, template <typename...> class Y> struct checkController {
  static constexpr bool value = false;
};

template <template <typename...> class T, typename... innerArgs>
struct checkController<T<innerArgs...>, T> {
  static constexpr bool value = true;
};
template <typename Controller, typename FunctorType, class IS, class IE>
void parallel_for(const std::initializer_list<IS> &start,
                  const std::initializer_list<IE> &end,
                  FunctorType &vectorKernel, std::string tag) {
  using ExecutionSpace = Controller::ExecutionSpace;
  if constexpr (checkController<Controller, KokkosController>::value) {
    constexpr auto funcRank = MeshFieldUtil::function_traits<FunctorType>::arity;
    assert(funcRank >= 1);
    Kokkos::Array<int64_t, funcRank> a_start =
	    MeshFieldUtil::to_kokkos_array<funcRank>(start);
    Kokkos::Array<int64_t, funcRank> a_end =
	    MeshFieldUtil::to_kokkos_array<funcRank>(end);
    if constexpr (funcRank == 1) {
	    Kokkos::RangePolicy<ExecutionSpace> p(a_start[0], a_end[0]);
	    Kokkos::parallel_for(tag, p, vectorKernel);
    } else {
	    Kokkos::MDRangePolicy<Kokkos::Rank<funcRank>, ExecutionSpace> policy(
			    a_start, a_end);
	    Kokkos::parallel_for(tag, policy, vectorKernel);
    }
  }
  else {
    simd_parallel_for<Controller>(start, end, vectorKernel, tag);
  }
  
}
} // namespace MeshField

#endif
