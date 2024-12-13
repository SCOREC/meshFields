#include "MeshField_Utility.hpp"
#include <Kokkos_Core.hpp>
namespace MeshField {
template <typename ExecutionSpace, typename FunctorType, class IS, class IE>
void parallel_for(ExecutionSpace, const std::initializer_list<IS> &start,
                  const std::initializer_list<IE> &end,
                  FunctorType &vectorKernel, std::string tag) {
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
} // namespace MeshField
