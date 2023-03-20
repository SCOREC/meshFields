#ifndef kokkosslicewrapper_hpp
#define kokkosslicewrapper_hpp

#define MeshFields_BUILD1(x) x[0]
#define MeshFields_BUILD2(x) MeshFields_BUILD1(x), x[1]
#define MeshFields_BUILD3(x) MeshFields_BUILD2(x), x[2]
#define MeshFields_BUILD4(x) MeshFields_BUILD3(x), x[3]
#define MeshFields_BUILD5(x) MeshFields_BUILD4(x), x[4]
#define MeshFields_BUILD(i,x) MeshFields_BUILD##i(x)

#include <tuple>
#include <vector>
#include <type_traits>
#include <cassert>
#include <initializer_list>

#include <Kokkos_Core.hpp>
#include "MeshField_Utility.hpp"
#include "MeshField_Macros.hpp"

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


template <class MemorySpace, class ExecutionSpace, class... Ts>
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

  template<typename... Tx>
  auto construct( std::vector<int> dims ) {
    return std::make_tuple( create_view<Tx>("view", dims)... );
  }

  template<typename Tx>
  Kokkos::View<Tx,MemorySpace>
  create_view( std::string tag, std::vector<int> &dims ) {
    int rank = Kokkos::View<Tx>::rank;
    int dynamic = Kokkos::View<Tx>::rank_dynamic;
    assert( rank <= 5 );

    if( rank == 0 ) return Kokkos::View<Tx,MemorySpace>(tag);
    
    Kokkos::View<Tx,MemorySpace> rt;
    switch( dynamic ) {
      case 1:
        rt = Kokkos::View<Tx,MemorySpace>(tag, MeshFields_BUILD(1,dims) );
        break;
      case 2:
        rt = Kokkos::View<Tx,MemorySpace>(tag, MeshFields_BUILD(2,dims) );
        break;
      case 3:
        rt = Kokkos::View<Tx,MemorySpace>(tag, MeshFields_BUILD(3,dims) );
        break;
      case 4:
        rt = Kokkos::View<Tx,MemorySpace>(tag, MeshFields_BUILD(4,dims) );
        break;
      case 5:
        rt = Kokkos::View<Tx,MemorySpace>(tag, MeshFields_BUILD(5,dims) );
        break;
      default:
        rt = Kokkos::View<Tx,MemorySpace>(tag);
    }
    
    dims.erase( dims.begin(), dims.begin()+dynamic );
    return rt;
  }
  
  // member vaiables
  const int* num_tuples;
  std::tuple<Kokkos::View<Ts,MemorySpace>...> values_;

public:

  KokkosController()  {
    std::vector<int> obj;
    values_ = construct<Ts...>(obj);
  }
    
  KokkosController(std::vector<int> items)
      : values_(construct<Ts...>(items)) {}

  int size() const { return num_tuples; }
  
  template<std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    return wrapper_slice_t<type>(std::get<index>(values_));
  }
  
  template <typename FunctorType, typename ReductionType>
  void parallel_reduce(FunctorType &reductionKernel,
                       ReductionType &reductionType, std::string tag) {
    /*
    Kokkos::RangePolicy<ExecutionSpace> policy(0, num_tuples);
    Kokkos::parallel_reduce(tag, policy, reductionKernel, reductionType);
    */
  }
  
  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS>& start,
                    const std::initializer_list<IE>& end,
                    FunctorType &vectorKernel,
                    std::string tag ) {
  
    constexpr auto RANK = MeshFieldUtil::function_traits<FunctorType>::arity;
    Kokkos::Array<int64_t,RANK> a_start = MeshFieldUtil::to_kokkos_array<RANK>( start );
    Kokkos::Array<int64_t,RANK> a_end = MeshFieldUtil::to_kokkos_array<RANK>( end );
    if constexpr ( RANK <= 1 ) {
      Kokkos::RangePolicy<> p(a_start[0], a_end[0]);
      Kokkos::parallel_for(tag,p,vectorKernel);
    } else {
      Kokkos::MDRangePolicy<Kokkos::Rank<RANK>> policy(a_start,a_end);
      Kokkos::parallel_for( tag, policy, vectorKernel );
    }
  }
  
};
} // namespace SliceWrapper

#endif
