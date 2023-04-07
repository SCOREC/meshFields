#ifndef kokkosslicewrapper_hpp
#define kokkosslicewrapper_hpp

#include <tuple>
#include <vector>
#include <cassert>
#include <iterator>
#include <typeinfo> // typeid
#include <type_traits>
#include <initializer_list>

#include <Kokkos_Core.hpp>
#include "MeshField_Utility.hpp"
#include "MeshField_Macros.hpp"

namespace Controller {

template <class SliceType, class T>
struct KokkosSliceWrapper{
  
  SliceType slice;
  int dimensions[5];
  typedef T Type;
  static const int MAX_RANK = 5;
  static const std::size_t RANK = Kokkos::View<T>::rank;

  KokkosSliceWrapper(SliceType slice_in, int* sizes) : slice(slice_in) {
    for( int i = 0; i < MAX_RANK; i++ ) dimensions[i] = sizes[i];
  }
  KokkosSliceWrapper() {}
  
  KOKKOS_INLINE_FUNCTION
  auto size(int i) const { 
    assert( i >= 0 );
    assert( i < MAX_RANK );
    return dimensions[i]; 
  }

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


template <class MemorySpace, class ExecutionSpace, typename... Ts>
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
  auto construct( std::vector<int> &dims ) {
    // Note: the '...' expansion will traverse <Tx...> backwards!
    return std::make_tuple( create_view<Tx>("view", dims)... );
  }
  
  template<typename Tx>
  Kokkos::View<Tx,MemorySpace> 
  create_view( std::string tag, std::vector<int> &dims ) { 
    int rank = Kokkos::View<Tx>::rank;
    int dynamic = Kokkos::View<Tx>::rank_dynamic;
    assert( rank <= 5 );
    assert( rank >= 0 );

    if( rank == 0 ) return Kokkos::View<Tx,MemorySpace>(tag);
    Kokkos::View<Tx,MemorySpace> rt;
    switch( dynamic ) {
      case 1:
        rt = Kokkos::View<Tx,MemorySpace>(tag, dims[dims.size()-1] );
        break;
      case 2:
        rt = Kokkos::View<Tx,MemorySpace>(tag, dims[dims.size()-2],
                                               dims[dims.size()-1] );
        break;
      case 3:
        rt = Kokkos::View<Tx,MemorySpace>(tag, dims[dims.size()-3],
                                               dims[dims.size()-2],
                                               dims[dims.size()-1]);
        break;
      case 4:
        rt = Kokkos::View<Tx,MemorySpace>(tag, dims[dims.size()-4],
                                               dims[dims.size()-3],
                                               dims[dims.size()-2],
                                               dims[dims.size()-1]);
        break;
      case 5:
        rt = Kokkos::View<Tx,MemorySpace>(tag, dims[dims.size()-5],
                                               dims[dims.size()-4],
                                               dims[dims.size()-3],
                                               dims[dims.size()-2],
                                               dims[dims.size()-1]);
        break;
      default:
        rt = Kokkos::View<Tx,MemorySpace>(tag);
        break;
    }
    
    // Places all of the dyanmic ranks into the extent_sizes
    for( int i = 0; i < dynamic; i++ ) {
      this->extent_sizes[theta][i] = dims[dims.size()-dynamic+i];
    }
    this->theta-=1;
    dims.erase( dims.end()-dynamic, dims.end());
    return rt;
  }
  
  template< typename T1, typename... Tx>
  void construct_sizes() {
    int total_rank = Kokkos::View<T1>::rank;
    int dynamic_rank = Kokkos::View<T1>::rank_dynamic;
    const int static_rank = total_rank - dynamic_rank;
    assert( total_rank <= 5 );
    assert( total_rank >= 0 );

    switch( static_rank ) {
      case 1: 
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        break;
      case 2:
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][dynamic_rank+1] = (int)std::extent<T1,1>::value;
        break;
      case 3:
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][dynamic_rank+1] = (int)std::extent<T1,1>::value;
        extent_sizes[delta][dynamic_rank+2] = (int)std::extent<T1,2>::value;
        break;
      case 4:
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][dynamic_rank+1] = (int)std::extent<T1,1>::value;
        extent_sizes[delta][dynamic_rank+2] = (int)std::extent<T1,2>::value;
        extent_sizes[delta][dynamic_rank+3] = (int)std::extent<T1,3>::value;
        break;
      case 5:
        extent_sizes[delta][0] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][1] = (int)std::extent<T1,1>::value;
        extent_sizes[delta][2] = (int)std::extent<T1,2>::value;
        extent_sizes[delta][3] = (int)std::extent<T1,3>::value;
        extent_sizes[delta][4] = (int)std::extent<T1,4>::value;
        break;
      case 0:
        break;
      default:
        break;
    }
    this->delta+=1;
    if constexpr ( sizeof...(Tx) > 1 ) construct_sizes<Tx...>();
    else                    construct_final_size<Tx...>();
  }

  template< typename T1 >
  void construct_final_size() {
    int total_rank = Kokkos::View<T1>::rank;
    int dynamic_rank = Kokkos::View<T1>::rank_dynamic;
    const int static_rank = total_rank - dynamic_rank;
    assert( total_rank <= 5 );
    assert( total_rank >= 0 );
    switch( static_rank ) {
      case 1: 
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        break;
      case 2:
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][dynamic_rank+1] = (int)std::extent<T1,1>::value;
        break;
      case 3:
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][dynamic_rank+1] = (int)std::extent<T1,1>::value;
        extent_sizes[delta][dynamic_rank+2] = (int)std::extent<T1,2>::value;
        break;
      case 4:
        extent_sizes[delta][dynamic_rank] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][dynamic_rank+1] = (int)std::extent<T1,1>::value;
        extent_sizes[delta][dynamic_rank+2] = (int)std::extent<T1,2>::value;
        extent_sizes[delta][dynamic_rank+3] = (int)std::extent<T1,3>::value;
        break;
      case 5:
        extent_sizes[delta][0] = (int)std::extent<T1,0>::value;
        extent_sizes[delta][1] = (int)std::extent<T1,1>::value;
        extent_sizes[delta][2] = (int)std::extent<T1,2>::value;
        extent_sizes[delta][3] = (int)std::extent<T1,3>::value;
        extent_sizes[delta][4] = (int)std::extent<T1,4>::value;
        break;
      default:
        break;
    }
    for( int i = 0; i < dynamic_rank; i++ ) {
      extent_sizes[delta][i] = 0;
    }
    this->delta+=1;
  }
  
  // member vaiables
  const int num_types = sizeof...(Ts);
  unsigned short delta = 0;
  unsigned short theta = num_types-1; 
  int extent_sizes[sizeof...(Ts)][5];
  std::tuple<Kokkos::View<Ts,MemorySpace>...> values_;

public:
  
  typedef ExecutionSpace exe;
  static const int MAX_RANK = 5;
  
  KokkosController() {
    if constexpr ( sizeof...(Ts) > 1 ) { construct_sizes<Ts...>(); }
    else if constexpr ( sizeof...(Ts) == 1 ) { construct_final_size<Ts...>(); }
    std::vector<int> obj;
    values_ = construct<Ts...>(obj);
  }

  KokkosController(const std::initializer_list<int> items) {
    if constexpr ( sizeof...(Ts) > 1 ) { construct_sizes<Ts...>(); }
    else if constexpr ( sizeof...(Ts) == 1 ) { construct_final_size<Ts...>(); }
    std::vector<int> obj(items);
    values_ = construct<Ts...>(obj);
  }
  
  ~KokkosController() = default;
  
  int size( int type_index, int dimension_index ) const { 
    assert(type_index >= 0);
    assert(type_index < num_types);
    assert(dimension_index >= 0);
    assert(dimension_index < 5);
    return extent_sizes[type_index][dimension_index]; 
  }
  
  template<std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    int slice_dims[5];
    for( int i = 0; i < 5; i++ ) slice_dims[i] = this->extent_sizes[index][i];
    return wrapper_slice_t<type>(std::get<index>(values_),slice_dims);
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
