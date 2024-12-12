#include <Kokkos_Core.hpp>
#include "MeshField_Utility.hpp"
namespace MeshField {
template <typename ExecutionSpace, typename FunctorType, class IS, class IE, class ReducerType>
void parallel_reduce(ExecutionSpace, std::string tag, const std::initializer_list<IS> &start,
    const std::initializer_list<IE> &end,
    FunctorType &reductionKernel, ReducerType &reducer) {
  /* TODO: infinite reducers */
  /* Number of arguements to lambda should be equal to number of ranks +
   * number of reducers
   * -> adjust 'Rank' accordingly */
  constexpr std::size_t reducer_count = 1;
  constexpr auto funcRank =
    MeshFieldUtil::function_traits<FunctorType>::arity - reducer_count;

  assert(start.size() == end.size());
  if constexpr (funcRank <= 1) {
    Kokkos::RangePolicy<ExecutionSpace> policy((*start.begin()), (*end.begin()));
    Kokkos::parallel_reduce(tag, policy, reductionKernel, reducer);
  } else {
    auto a_start = MeshFieldUtil::to_kokkos_array<funcRank>(start);
    auto a_end = MeshFieldUtil::to_kokkos_array<funcRank>(end);
    Kokkos::MDRangePolicy<Kokkos::Rank<funcRank>, ExecutionSpace> policy(a_start,
        a_end);
    Kokkos::parallel_reduce(tag, policy, reductionKernel, reducer);
  }
}
}

