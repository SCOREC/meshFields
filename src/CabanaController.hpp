#ifndef cabanaslicewrapper_hpp
#define cabanaslicewrapper_hpp

#include <Cabana_Core.hpp>

namespace Controller {

template<class SliceType, class T>
struct CabanaSliceWrapper {
  
  SliceType slice;
  typedef T Type;

  CabanaSliceWrapper( SliceType slice_in ) : slice(slice_in) {}
  CabanaSliceWrapper( ) {}
  
  /* 1D access */
  KOKKOS_INLINE_FUNCTION
  T &operator()(int s) const { return slice(s); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a) const { return slice(s,a); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i) const { return slice(s,a,i); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j) 
    const { return slice(s,a,i,j); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j, int k) 
    const { return slice(s,a,i,j,k); }

  /* 2D access */
  KOKKOS_INLINE_FUNCTION
  T &access( int s ) const { return slice.access(s); }

  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a ) const { return slice.access(s,a); }

  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a, int i ) const { return slice.access(s,a,i); }

  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a, int i, int j ) 
    const { return slice.access(s,a,i,j); }

  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a, int i, int j, int k ) 
    const { return slice.access(s,a,i,j,k); }

};


using namespace Cabana;

template <class ExecutionSpace, class MemorySpace, class... Ts>
class CabanaController {

  // type definitions
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;

public:
  static constexpr int vecLen =
      Cabana::AoSoA<DataTypes, DeviceType>::vector_length;

private:
  // all the type defenitions that are needed us to get the type of the slice
  // returned by the underlying AoSoA
  using soa_t = SoA<DataTypes, vecLen>;

  template <std::size_t index>
  using member_data_t =
      typename Cabana::MemberTypeAtIndex<index, DataTypes>::type;

  template <std::size_t index>
  using member_value_t =
      typename std::remove_all_extents<member_data_t<index>>::type;

  template <class T, int stride>
  using member_slice_t =
      Cabana::Slice<T, DeviceType, Cabana::DefaultAccessMemory, vecLen, stride>;

  template <class T, int stride>
  using wrapper_slice_t = CabanaSliceWrapper<member_slice_t<T, stride>, T>;

  // member vaiables
  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa;
  const int num_tuples;

public:

  CabanaController() : num_tuples(0) {}

  CabanaController(int n)
      : aosoa("sliceAoSoA", n), num_tuples(n) {}

  int size() const { return num_tuples; }


  template <std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    const int stride = sizeof(soa_t) / sizeof(member_value_t<index>);
    auto slice = Cabana::slice<index>(aosoa);
    return wrapper_slice_t<type, stride>(std::move(slice));
  }
  /*
  template <typename FunctorType, typename ReductionType>
  void parallel_reduce(FunctorType &reductionKernel,
                       ReductionType &reductionType, std::string tag) {
    Kokkos::RangePolicy<ExecutionSpace> policy(0, num_tuples);
    Kokkos::parallel_reduce(tag, policy, reductionKernel, reductionType);
  }

  template <typename FunctorType>
  void parallel_for(int lowerBound, int upperBound, FunctorType &vectorKernel,
                    std::string tag) {
    Cabana::SimdPolicy<vecLen, ExecutionSpace> simdPolicy(lowerBound,
                                                          upperBound);
    Cabana::simd_parallel_for(simdPolicy, vectorKernel, tag);
  }
  */
};
} // namespace SliceWrapper

#endif
