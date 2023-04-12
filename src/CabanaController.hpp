#ifndef cabanaslicewrapper_hpp
#define cabanaslicewrapper_hpp

#include <type_traits>
#include <vector>
#include <Cabana_Core.hpp>

namespace Controller {

template<class SliceType, class T>
struct CabanaSliceWrapper {
    
  static const int MAX_RANK = 4;
  static const std::size_t RANK = Kokkos::View<T>::rank + 1;
  int dimensions[MAX_RANK];
  SliceType slice;
  typedef T Type;

  CabanaSliceWrapper( SliceType slice_in, int* sizes ) : slice(slice_in) {
    for( int i = 0; i < MAX_RANK; i++ ) {
      dimensions[i] = sizes[i];
    }
  }
  CabanaSliceWrapper( ) {}
  
  KOKKOS_INLINE_FUNCTION
  auto size( int i ) const { 
    assert( i >= 0 );
    assert( i <= MAX_RANK );
    return dimensions[i];  
  }

  KOKKOS_INLINE_FUNCTION
  auto rank() const { return std::rank<Type>{} + 1; }

  /* 1D access */
  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s) const { return slice(s); }

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
  
};

using namespace Cabana;

template <class ExecutionSpace, class MemorySpace, class... Ts>
class CabanaController {

  // type definitions
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;
  
public:
  typedef ExecutionSpace exe;
  static const int MAX_RANK = 4; // Including num_tuples -> so 3 additional extents
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

  template< typename T1, typename... Tx >
  void construct_sizes() {
    // There are no dynamic ranks w/ cabana controller.
    // So we only need to know extent.
    // Cabana SoA only supports up to 3 additional ranks...
    std::size_t rank = std::rank<T1>{};
    assert( rank < MAX_RANK );
    switch( rank ) {
      case 3:
        extent_sizes[theta][3] = (int)std::extent<T1,2>::value;
      case 2:
        extent_sizes[theta][2] = (int)std::extent<T1,1>::value;
      case 1:
        extent_sizes[theta][1] = (int)std::extent<T1,0>::value;
      default:
        for( int i = MAX_RANK -1; i > rank; i-- ) {
          extent_sizes[theta][i] = 0;
        }
        break;
    }
    extent_sizes[theta][0] = num_tuples;
    this->theta+=1;
    if constexpr (sizeof...(Tx) != 0 ) construct_sizes<Tx...>();
  }

  // member vaiables
  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa;
  const int num_tuples;
  int theta = 0;
  int extent_sizes[sizeof...(Ts)][MAX_RANK];

public:

  CabanaController() : num_tuples(0) {
    static_assert( sizeof...(Ts) != 0 );
    construct_sizes<Ts...>();
  }

  CabanaController(int n) 
    : aosoa("sliceAoSoA", n), num_tuples(n) {
    static_assert( sizeof...(Ts) != 0 );
    construct_sizes<Ts...>();
  }

  int size(int i, int j) const { 
    assert( j >= 0 );
    assert( j < 5 );
    if( j == 0 ) return this->tuples();
    else return extent_sizes[i][j];
  }
  
  int tuples() const { return num_tuples; }

  template <std::size_t index> auto makeSlice() {
    using type = std::tuple_element_t<index, TypeTuple>;
    const int stride = sizeof(soa_t) / sizeof(member_value_t<index>);
    auto slice = Cabana::slice<index>(aosoa);
    int sizes[MAX_RANK];
    for( int i = 0; i < MAX_RANK; i++ ) sizes[i] = this->size(index,i);
    return wrapper_slice_t<type, stride>(std::move(slice),sizes);
  }
  
  static constexpr int saToItemIndex( const int& s, const int& a ) {
    return s*vecLen + a;
  }
  
  template <typename FunctorType, class IS, class IE, int vectorLength>
  void parallel_for_helper(const std::initializer_list<IS>& start_init,
                           const std::initializer_list<IE>& end_init,
                           FunctorType &vectorKernel,
                           std::string tag) {
    static_assert( std::is_integral<IS>::value, "Integral required\n" );
    static_assert( std::is_integral<IE>::value, "Integral required\n" );
    std::vector<IS> start_v(start_init); // -> Needs to be a view
    std::vector<IE> end_v(end_init); // -> Needs to be view
    Kokkos::View<int*> start("1",(int)start_v.size());
    Kokkos::View<int*> end("2",(int)end_v.size()); 

    IS* start_p = start_v.data();
    IE* end_p = end_v.data();
    Kokkos::RangePolicy<ExecutionSpace> linear_policy(0,(int)start.size());
    Kokkos::parallel_for(linear_policy, KOKKOS_LAMBDA(const int& i) {
      start(i) = (int)(*(start_p+i));
      end(i) = (int)(*(end_p+i));
    });
    Cabana::SimdPolicy<vectorLength,ExecutionSpace> policy(start[0],end[0]);

    constexpr std::size_t FunctorRank = MeshFieldUtil::function_traits<FunctorType>::arity;
    if constexpr (FunctorRank == 1) {
      Cabana::simd_parallel_for(policy,KOKKOS_LAMBDA(const int& s, const int& a) {
        const int i = s*vectorLength+a;
        vectorKernel(i);
      },"Controller::CabanaController::parallel_for -> Rank 1\n");
    } else
    if constexpr (FunctorRank == 2) {
      Cabana::simd_parallel_for(policy,KOKKOS_LAMBDA(const int& s, const int& a) {
        const int i = s*vectorLength+a;
        for( int j = start(1); j < end(1); j++) {
          vectorKernel(i,j);
        }
      },"Controller::CabanaController::parallel_for -> Rank 2\n");
    } // TODO: else if ...
  
  }

  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS>& start_init,
                    const std::initializer_list<IE>& end_init,
                    FunctorType &vectorKernel,
                    std::string tag) {
    this->parallel_for_helper<FunctorType,IS,IE,vecLen>
      (start_init,end_init,vectorKernel,tag);
    /*
    static_assert( std::is_integral<IS>::value, "Integral required\n" );
    static_assert( std::is_integral<IE>::value, "Integral required\n" );
    std::vector<IS> start(start_init);
    std::vector<IE> end(end_init);
    Cabana::SimdPolicy<vecLen,ExecutionSpace> policy(start[0],end[0]);
    // need to have vector length in lambda expression, but 
    // references to host classes/functions is not allowed...
    // cant pass it into the LAMBDA expression because cabana
    // is expecting a specific number of arguements.
    // -> template lambdas only work in c++20 and above.
    constexpr std::size_t FunctorRank = MeshFieldUtil::function_traits<FunctorType>::arity;
    if constexpr (FunctorRank == 1) {
      auto kernel = KOKKOS_LAMBDA(const int& s, const int& a) {
        const int i = s*vecLen+a;
        vectorKernel(i);
      };
      Cabana::simd_parallel_for(policy,kernel,"Controller::CabanaController::parallel_for -> Rank 1\n");
    } else
    if constexpr (FunctorRank == 2) {
       auto kernel = KOKKOS_LAMBDA(const int& s, const int& a) {
        const int i = s*vecLen+a;
        for( int j = start[1]; j < end[1]; j++) {
          vectorKernel(i,j);
        }
      };
      Cabana::simd_parallel_for(policy,kernel,"Controller::CabanaController::parallel_for -> Rank 2\n");
    }*/
    /*
    Cabana::simd_parallel_for(policy, KOKKOS_LAMBDA( const int s, const int a ) {
      constexpr int i = Controller::CabanaController<ExecutionSpace,MemorySpace,bool>::saToItemIndex(s,a);
      constexpr std::size_t FunctorRank = MeshFieldUtil::function_traits<FunctorType>::arity;
      if constexpr (FunctorRank == 1) {
          vectorKernel(i);
      } else 
      if constexpr ( FunctorRank == 2 ) {
        for( int j = start[1]; j < end[1]; j++) {
          vectorKernel(i,j);
        } 
      } else 
      if constexpr ( FunctorRank == 3 ) {
        for( int j = start[2]; j < end[2]; j++) {
          for( int k = start[3]; k < end[3]; k++ ) {
            vectorKernel(i,j,k);
          }
        }
      } else
      if constexpr ( FunctorRank == 4 ) {
        for( int j = start[2]; j < end[2]; j++) {
          for( int k = start[3]; k < end[3]; k++ ) {
            for( int l = start[4]; l < end[4]; l++ ) {
              vectorKernel(i,j,k,l);
            }
          }
        }
      } else { fprintf(stderr,"Invalid Lambda Rank must be [1,4]\n"); }
    },"yesyes");
    */
  }
  
};
} // namespace Controller

#endif
