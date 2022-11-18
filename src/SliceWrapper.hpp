#ifndef slicewrapper_hpp
#define slicewrapper_hpp

#include <Cabana_Core.hpp>

namespace SliceWrapper {

template< class SliceType, class T >
struct SliceWrapper {

  SliceType st_; //store the underlying instance

  SliceWrapper(SliceType st) : st_(st)  {}

  SliceWrapper() {}
  
  KOKKOS_INLINE_FUNCTION
  T& access(int s, int a) const {
    return st_.access(s,a);
  }
  KOKKOS_INLINE_FUNCTION
  auto& access(int s, int a, int i) const {
    return st_.access(s,a,i);
  }
  KOKKOS_INLINE_FUNCTION
  auto& access(int s, int a, int i, int j) const {
    return st_.access(s,a,i,j);
  }
  KOKKOS_INLINE_FUNCTION
  auto& access(int s, int a, int i, int j, int k) const {
    return st_.access(s,a,i,j,k);
  }  
};

using namespace Cabana;

template <class ExecutionSpace, class MemorySpace, class... Ts>
class CabSliceController {
  // type definitions
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;
public:
  static constexpr int vecLen = Cabana::AoSoA<DataTypes, DeviceType>::vector_length;
private:
  using soa_t = SoA<DataTypes, vecLen>;

  template <std::size_t index>
  using member_data_t = typename Cabana::MemberTypeAtIndex<index, DataTypes>::type;

  template <std::size_t index>
      using member_value_t =
    typename std::remove_all_extents<member_data_t<index>>::type;
  
  template <class T, int stride>
  using member_slice_t = 
    Cabana::Slice<T, DeviceType, 
		  Cabana::DefaultAccessMemory, 
		  vecLen, stride>;

  template <class T, int stride>
  using wrapper_slice_t = SliceWrapper<member_slice_t<T, stride>, T>;

  // member vaiables
  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa; 
  const int num_tuples;
  
public:
  int size() {
    return num_tuples;
  }

  KOKKOS_INLINE_FUNCTION
  void indexToSA(const int i, int& s, int& a) {
    s = i / vecLen;
    a = i % vecLen;
  }
  
  template <typename FunctorType, typename ReductionType>
  void parallel_reduce(FunctorType& reduceKernel, ReductionType& reductionType, std::string tag) {
    Kokkos::RangePolicy<ExecutionSpace> policy(0, num_tuples);
    Kokkos::parallel_reduce(tag, policy, reduceKernel, reductionType);
  }
  
  template<typename FunctorType>
  void parallel_for(int lower_bound, int upper_bound, FunctorType& vectorKernel, std::string tag) {
    Cabana::SimdPolicy<vecLen, ExecutionSpace> simd_policy(lower_bound, upper_bound);
    Cabana::simd_parallel_for(simd_policy, vectorKernel, tag);
  }
  
  template <std::size_t index>
  auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    const int stride = sizeof(soa_t) / sizeof(member_value_t<index>);
    auto slice = Cabana::slice<index>(aosoa);
    return wrapper_slice_t< type, stride >(std::move(slice));
  }

  CabSliceController() {}
  
  CabSliceController(int n) : aosoa("sliceAoSoA", n), num_tuples(n) {
    if (sizeof...(Ts) == 0) {
      throw std::invalid_argument("Must provide at least one member type in template definition");
    }
  }
};

}

#endif

