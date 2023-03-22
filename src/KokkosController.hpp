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
#include <cassert>
#include <algorithm> // std::reverse
#include <type_traits>
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
    //itr = extent_sizes.begin();
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
    /*
    // Places all of the dyanmic ranks into the extent_sizes
    // -> pointed to by 'itr'
    for( int i = 0; i < dynamic; i++ ) {
      (*itr)[i] = x[i]; // extent_sizes[Tx][0,...,dynamic_rank-1]
    }
    
    itr++;
    */
    dims.erase( dims.begin(), dims.begin()+dynamic );
    return rt;
  }
  /*
  // *[1][2][3][4] -> static_rank = 4 dynamic_rank = 1 -> [0,1,2,3,4]
  // **[5] -> static_rank = 1, dynamic_rank = 2 -> [0,0,5]
  template< typename T1, typename... Tx>
  void construct_sizes() {
    std::vector<int> sizes;
    constexpr auto static_rank = Kokkos::View<T1>::rank;
    constexpr auto dynamic_rank = Kokkos::View<T1>::rank_dynamic;
    assert( static_rank + dynamic_rank <= 5 );
    assert( static_rank + dynamic_rank >= 0 );
    switch( static_rank ) {
      case 5:
        constexpr auto a = 4;
        sizes.push_back( (int)std::extent<T1,dynamic_rank + a>::value );
      case 4:
        constexpr auto b = 3;
        sizes.push_back( (int)std::extent<T1,dynamic_rank + b>::value );
      case 3:
        constexpr auto c = 2;
        sizes.push_back( (int)std::extent<T1,dynamic_rank + c>::value );
      case 2:
        constexpr auto d = 1;
        sizes.push_back( (int)std::extent<T1,dynamic_rank + d>::value );
      case 1:
        sizes.push_back( (int)std::extent<T1,dynamic_rank>::value );
      case 0:
        break;
    }
    for( int i = 0; i < dynamic_rank; i++ ) {
      sizes.push_back( 0 );
    }
    std::reverse( sizes.begin(), sizes.end() );
    extent_sizes.push_back(sizes);
    if( sizeof...(Tx) != 0 ) {
      construct_sizes_helper<Tx...>();
    }
  }
  */
  
  // member vaiables
  //std::vector<std::vector<int>>::iterator itr;
  std::vector<std::vector<int>> extent_sizes;
  std::tuple<Kokkos::View<Ts,MemorySpace>...> values_;

public:

  typedef ExecutionSpace exe;

  KokkosController()  {
    //construct_sizes<Ts...>();
    std::vector<int> obj;
    values_ = construct<Ts...>(obj);
  }
    
  KokkosController(const std::initializer_list<int> items) {
    std::vector<int> obj(items);
    //construct_sizes<Ts...>();
    values_ = construct<Ts...>(obj);
  }

  
  // TODO update relative to datastructure
  int size() const { return 0; }
  
  template<std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    return wrapper_slice_t<type>(std::get<index>(values_));
  }
  
  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS>& start,
                    const std::initializer_list<IE>& end,
                    FunctorType &vectorKernel,
                    std::string tag ) {
    constexpr auto RANK = MeshFieldUtil::function_traits<FunctorType>::arity;
    assert( RANK >= 1 );
    Kokkos::Array<int64_t,RANK> a_start = MeshFieldUtil::to_kokkos_array<RANK>( start );
    Kokkos::Array<int64_t,RANK> a_end = MeshFieldUtil::to_kokkos_array<RANK>( end );
    if constexpr ( RANK == 1 ) {
      Kokkos::RangePolicy<ExecutionSpace> p(a_start[0], a_end[0]);
      Kokkos::parallel_for(tag,p,vectorKernel);
    } else {
      Kokkos::MDRangePolicy<Kokkos::Rank<RANK>, ExecutionSpace> policy(a_start,a_end);
      Kokkos::parallel_for( tag, policy, vectorKernel );
    }
  }
};
} // namespace SliceWrapper

#endif
