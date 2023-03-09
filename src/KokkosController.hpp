#ifndef kokkosslicewrapper_hpp
#define kokkosslicewrapper_hpp

#include <tuple>

namespace Controller {

template <class SliceType, class T>
struct KokkosSliceWrapper{
  
  SliceType slice;
  typedef T Type;

  KokkosSliceWrapper(SliceType slice_in) : slice(slice_in) {}
  KokkosSliceWrapper() {}
  

  /* 1D Access */
  KOKKOS_INLINE_FUNCTION
  T &operator()( int s ) const { return slice(s); }
  
  KOKKOS_INLINE_FUNCTION
  auto &operator()( int s, int a ) const { return slice(s,a); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()( int s, int a, int i ) const { return slice(s,a,i); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()( int s, int a, int i, int j ) 
    const { return slice(s,a,i,j); }
  
  KOKKOS_INLINE_FUNCTION
  auto &operator()( int s, int a, int i, int j, int k ) 
    const { return slice(s,a,i,j,k); }
  
  /* TODO 2D access /w vectorization */

  KOKKOS_INLINE_FUNCTION
  T &access( int s ) const { return NULL };
  
  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a ) const { return NULL };
  
  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a, int i ) const { return NULL };

  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a, int i, int j ) const { return NULL };

  KOKKOS_INLINE_FUNCTION
  auto &access( int s, int a, int i, int j, int k ) const { return NULL };

};


template <class ExecutionSpace, class MemorySpace, class... Ts>
class KokkosController {

  // type definitions
  using TypeTuple = std::tuple<Ts...>;

private:
  // all the type defenitions that are needed us to get the type of the slice
  // returned by the underlying AoSoA
  template<std::size_t index> 
  using member_data_t = 
      typename std::tuple_element<index, std::tuple<Ts...>>::type;

  template <std::size_t index>
  using member_value_t =
      typename std::remove_all_extents<member_data_t<index>>::type;

  template <class T>
  using member_slice_t =
      Kokkos::View<T, ExecutionSpace, MemorySpace>;

  template <class T>
  using wrapper_slice_t = KokkosSliceWrapper<member_slice_t<T>, T>;

  template< typename... Tx>
  auto construct( int size_) {
    return std::make_tuple( Kokkos::View<Tx,ExecutionSpace,MemorySpace>(size_)... );
  }

  // member vaiables
  const int num_tuples;
  std::tuple<Kokkos::View<Ts,MemorySpace>...> values_;

public:

  KokkosController() : values_(construct<Ts...>(1)) {}

  KokkosController(int n)
      : num_tuples(n), values_(construct<Ts...>(n)) {}
  
  int size() const { return num_tuples; }
  
  template<std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    return wrapper_slice_t<type>(std::move(std::get<index>(values_)));
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
    Kokkos::RangePolicy<ExecutionSpace> p(lowerBound, upperBound);
    Kokkos::parallel_for(tag,p,vectorKernel);
  }
  */
};
} // namespace SliceWrapper

#endif
