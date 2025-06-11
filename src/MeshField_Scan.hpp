#ifndef MESHFIELD_SCAN_HPP
#define MESHFIELD_SCAN_HPP

#include "MeshField_Utility.hpp"
#include <Kokkos_Core.hpp>
namespace MeshField {
template <typename ExecutionSpace, typename KernelType, typename resultant>
void parallel_scan(ExecutionSpace, std::string tag, int64_t start_index,
                   int64_t end_index, KernelType &scanKernel,
                   resultant &result) {
  static_assert(std::is_standard_layout<resultant>::value &&
                std::is_trivial<resultant>::value);
  Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<int64_t>> p(start_index,
                                                                    end_index);
  Kokkos::parallel_scan(tag, p, scanKernel, result);
}
} // namespace MeshField

#endif
