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
  auto &operator()(const int i) const {
    auto s = SliceType::index_type::s(i);
    auto a = SliceType::index_type::a(i);
    return slice.access(s,a); 
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int i, int j) const {
    auto s = SliceType::index_type::s(i);
    auto a = SliceType::index_type::a(i);
    return slice.access(s,a,j);
  }
  //TODO: copy above return of .access for following methods.
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
  
  /*
    can use SliceType::index_type::i(s,a)
  static constexpr int saToItemIndex( const int& s, const int& a ) {
    return s*vecLen + a;
  }
  
    can use SliceType::index_type::s(i)
    can use SliceType::index_type::a(i)
  static constexpr void itemIndexToSA(const int& i, int& s, int& a) {
    s = i / vecLen;
    a = i % vecLen;
  }

  */
  template <typename FunctorType, class IS, class IE, int vectorLength>
  void parallel_for_helper(const std::initializer_list<IS>& start_init,
                           const std::initializer_list<IE>& end_init,
                           FunctorType &vectorKernel,
                           std::string tag) {
    // requirement start_init and end_init must be of equal size.
    static_assert( std::is_integral<IS>::value, "Integral required\n" );
    static_assert( std::is_integral<IE>::value, "Integral required\n" );

    auto start = start_init.begin();
    auto end = end_init.begin();

    //assert( start.size() == end.size() );
    
    assert(cudaSuccess == cudaDeviceSynchronize());

    Cabana::SimdPolicy<vectorLength,ExecutionSpace> policy(*start,*end);


    constexpr std::size_t FunctorRank = MeshFieldUtil::function_traits<FunctorType>::arity;
    static_assert( FunctorRank >= 1 && FunctorRank <= CabanaController::MAX_RANK );
    if constexpr (FunctorRank == 1) {
      Cabana::simd_parallel_for(policy,KOKKOS_LAMBDA(const int& s, const int& a) {
        const int i = s*vectorLength+a; // TODO use impel_index
        vectorKernel(i);
      },"Controller::CabanaController::parallel_for -> Rank 1\n");
      assert(cudaSuccess == cudaDeviceSynchronize());
    } else
    if constexpr (FunctorRank == 2) {
      //TODO runtime failure ; work in progress
      Cabana::SimdPolicy<vectorLength,ExecutionSpace> policy_1(*start,*end);
      //const int s1 = 0;
      //const int e1 = 1;
      fprintf(stderr,"Here lolasdlghhwrlnkg\n");
      Cabana::simd_parallel_for(policy_1,KOKKOS_LAMBDA(const int& s, const int& a) {
        /*
        const int i = s*vectorLength+a;
        for( int j = s1; j < e1; j++) {
          //vectorKernel(i,j);
        }
        */
      },"Controller::CabanaController::parallel_for -> Rank 2\n");
    } else
    if constexpr (FunctorRank == 3) {
      Cabana::simd_parallel_for(policy,KOKKOS_LAMBDA(const int& s, const int& a) {
        const int i = s*vectorLength+a;
        const int s1=start(1),s2=start(2);
        const int e1=end(1),e2=end(2);
        for( int j = s1; j < e1; j++) {
          for( int k = s2; k < e2; k++ ) {
            vectorKernel(i,j,k);
          }
        }
      }, "Controller::CabanaController::parallel_for -> Rank 3\n");
    } // TODO: Rank 4
  
  }

  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS>& start_init,
                    const std::initializer_list<IE>& end_init,
                    FunctorType &vectorKernel,
                    std::string tag) {
    this->parallel_for_helper<FunctorType,IS,IE,vecLen>
      (start_init,end_init,vectorKernel,tag);
  }
  
};
} // namespace Controller

#endif
