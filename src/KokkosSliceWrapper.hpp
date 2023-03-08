#ifndef kokkosslicewrapper_hpp
#define kokkosslicewrapper_hpp

#include <tuple>

namespace SliceWrapper {

template <class SliceType, class T> struct KokkosSliceWrapper {

  SliceType slice;

  typedef T Type;
  
  int vectorLength = 1; // TODO REP EXPOSURE
  
  KokkosSliceWrapper(SliceType slice_in, int vecLen) 
    : slice(slice_in),
      vectorLength(vecLen) {}

  KokkosSliceWrapper(SliceType slice_in) 
    : slice(slice_in){}

  KokkosSliceWrapper() {}
  
  // TODO define proper accessing parameters...
  // Stride? VectorLength?
  // -> Need size of dimensions for slice-type to get
  // stride (and possibly vector length)
  // 
  // in header <type_traits>
  //  -> std::remove_extent()
    
  KOKKOS_INLINE_FUNCTION
  T &access(int s) const { return slice(s); }

  KOKKOS_INLINE_FUNCTION
  T &access(int s, int a) const { return slice(s,a); } 

  KOKKOS_INLINE_FUNCTION
  auto &access(int s, int a, int i) const { return slice(s, a, i); }

  KOKKOS_INLINE_FUNCTION
  auto &access(int s, int a, int i, int j) const {
    return slice(s, a, i, j);
  }

  KOKKOS_INLINE_FUNCTION
  auto &access(int s, int a, int i, int j, int k) const {
    return slice(s, a, i, j, k);
  }
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
      typename std::tuple_element<N, std::tuple<Ts...>>::type;

  template <std::size_t index>
  using member_value_t =
      typename std::remove_all_extents<member_data_t<index>>::type;

  template <class T>
  using member_slice_t =
      Kokkos::View<T, ExecutionSpace, MemorySpace>;

  template <class T>
  using wrapper_slice_t = KokkosSliceWrapper<member_slice_t<T>, T>;

  template< typename... Tx>
  auto construct( std::size_t size_) {
    return std::make_tuple( Kokkos::View<Tx,ExeuctionSpace,MemorySpace>(size_)... );
  }

  // member vaiables
  const int num_tuples;
  std::tuple<Kokkos::View<Ts,ExecutionSpace,MemorySpace>...> values_;

public:

  struct IndexToSA {
    IndexToSA( int vecLen_in ) : vecLen(vecLen_in) {}
    int vecLen;
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, int &s, int&a) const {
      s = i / vecLen;
      a = i % vecLen;
    }
  }

  IndexToSA indexToSA;


  // constructors (default constructor necessary)
  KokkosController() : values_(construct<Ts...>(1)), indexToSA(1) {}

  KokkosController(int n)
      : num_tuples(n), values_(construct<Ts...>(n), indexToSA(1)) {}
  
  KokkosController(int n, int vector_length)
      : num_tuples(n), values_(construct<Ts...>(n), indexToSA(vector_length)) {}

  // size function to get the number of tuples
  const int size() const { return num_tuples; }
  
  template<std::size_t index auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    return wrapper_slice_t<type>(std::get<index>(values_));
  }

  /* Parallel functions

     These functions run the given kernel in parallel on the GPU
     (unless the execution space is serial)
     parallel_for - 
       a for loop that will iterate from lowerBound to upperbound
     parallel_reduce - 
       a way for a user to pass in a reduction kernel and a
       reducer to make their own reductions 
  */

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
};
} // namespace SliceWrapper

#endif
