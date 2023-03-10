#ifndef kokkosslicewrapper_hpp
#define kokkosslicewrapper_hpp

#define MeshFields_BUILD1(x) x[0]
#define MeshFields_BUILD2(x) MeshFields_BUILD1, x[1]
#define MeshFields_BUILD3(x) MeshFields_BUILD2, x[2]
#define MeshFields_BUILD4(x) MeshFields_BUILD3, x[3]
#define MeshFields_BUILD5(x) MeshFields_BUILD4, x[4]
#define MeshFields_BUILD(i,x) MeshFields_BUILD##i(x)

#include <tuple>
#include <vector>

namespace Controller {

template <class SliceType, class T>
struct KokkosSliceWrapper{
  
  SliceType slice;
  typedef T Type;

  KokkosSliceWrapper(SliceType slice_in) : slice(slice_in) {}
  KokkosSliceWrapper() {}
  

  /* 1D Access */
  KOKKOS_INLINE_FUNCTION
  auto &operator()( int s ) const { return slice(s); }
  
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
  
};


template <class MemorySpace, class... Ts>
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
      Kokkos::View<T,MemorySpace>;

  template <class T>
  using wrapper_slice_t = KokkosSliceWrapper<member_slice_t<T>, T>;
  
  /*
  template< typename... Tx>
  auto construct( int size_ ) {
    return std::make_tuple( Kokkos::View<Tx,MemorySpace>("view",size_)... );
  }*/
  template< typename... Tx>
  auto construct( std::vector<int> runtime_ds ) {
    return std::make_tuple( Kokkos::View<Tx,MemorySpace>("MeshFieldView",
          ( 
           /*  TODO:
               HERE, need indices to
               the vector for runtime dimensions...
               maybe add a helper function
               to the compiler definitions at
               the top? 
            */
            
            ( MeshFields_BUILD(std::rank<Tx>{},runtime_ds) )  
          )
          )... );
  }             
  // x = double***[extent]
  // rank = 4
  // extent<x,3> = extent
  // extent<x,2> = 0
  // extent<x,1> = 0
  // extent<x,0> = 0
  // KokkosController<MemorySpace,Tx...> x( int... y );
  
  //MeshFields_BUILD5(x) -> [0], [1], [2], ...

  int construct_helper() {
    return 0; 
  }
    
  

  // member vaiables
  const int* num_tuples;

  std::tuple<Kokkos::View<Ts,MemorySpace>...> values_;

public:

  KokkosController() : values_(construct<Ts...>()) {}
  
  /* TODO: accept runtime dimensions and place into array...
   *        - make sure that they're used in the construct
   *          function when creating views.
   *
   *    KokkosController(int... indices)?
   *    
   *    A way to determine runtime dimensions.
   *    for example:
   *      double***,doule**,double[extent]*
   *
   *
   *    int header <type_traits>
   *    std::rank<class T>{} -> How many dimensions
   *    std::extent<class T, unsigned N>::value
   *    
   *    idea:
   *      Calculate number of dimensions per Ts, then
   *      loop through dimensions until we get returned 
   *      0 which means that it is undefined (hence runime)
   *      and then we fill it with that number of dimensions
   *      from the array that is passed to the function.
   *
   */
  
  KokkosController(int* n)
      : num_tuples(std::move(n)), values_(construct<Ts...>(n)) {}
  //KokkosController(int n) : num_tuples(n), values_(construct<Ts...>(n)) {}
/*
  1 to 1
  double**
  kernal ( int rank1, int rank2 ) {
    ...
    field0(rank1,rank2);
    ...
  }
  */
  int size() const { return num_tuples; }
  
  template<std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    return wrapper_slice_t<type>(std::get<index>(values_));
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
