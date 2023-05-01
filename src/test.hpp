#ifndef _test_hpp_
#define _test_hpp_
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <array>
#include <type_traits>
#include <Cabana_Core.hpp>
#include "MeshField_Utility.hpp"
#include "MeshField.hpp"
#include "CabanaController.hpp"

template <typename T>
struct function_traits
  : public function_traits<decltype(&T::operator())>
{};

template< typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const> {
  static constexpr std::size_t arity = sizeof...(Args);
  typedef ReturnType result_type;
  template<std::size_t i> struct arg {
    typedef typename std::tuple_element<i,std::tuple<Args...>>::type type;
  };
};

template<std::size_t RANK, typename Type>
struct impl{
  impl(){}
  virtual bool x( std::vector<int>& start,
                  std::vector<int>& end) {
    return false;
  }
};

template<typename FunctorType>
struct impl<1,FunctorType> {
  impl( FunctorType& t) : z(t) {}
  bool x( std::vector<int>& start,
          std::vector<int>& end ) {
    assert( start.size() >= 1 && end.size() >= 1 );
    for( int i = start[0]; i < end[0]; i++ ) z(i);
    return true;
  }

  FunctorType z;
};

template<typename FunctorType>
struct impl<2,FunctorType> {
  impl( FunctorType& t) : z(t) {}
  bool x( std::vector<int>& start,
          std::vector<int>& end) {
    assert( start.size() >= 2 && end.size() >= 2);
    for( int i = start[0]; i < end[0]; i++ ) {
      for( int j =(start[1]); j < (end[1]); j++ ) {
        z(i,j);
      }
    }
    return true;
  }

  FunctorType z;
};

template <class FunctorType>
typename std::enable_if< 2 == function_traits<FunctorType>::arity>::type
parallel_for(FunctorType& kernel, const std::initializer_list<int>& start, const std::initializer_list<int>& end) {
  Kokkos::Array<int64_t,2> a_start = MeshFieldUtil::to_kokkos_array<2>(start);
  Kokkos::Array<int64_t,2> a_end = MeshFieldUtil::to_kokkos_array<2>(end);
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> p(a_start,a_end);
  Kokkos::parallel_for( "rank_2", p, kernel );
}
template <class FunctorType>
typename std::enable_if< 1 == function_traits<FunctorType>::arity>::type
parallel_for(FunctorType& kernel, const std::initializer_list<int>& start, const std::initializer_list<int>& end) {
  Kokkos::RangePolicy p(*start.begin(),*end.begin());
  Kokkos::parallel_for( "rank_1", p, kernel );
}

template<class Fn, std::size_t vectorLength=32>
typename 
std::enable_if<1 == MeshFieldUtil::function_traits<Fn>::arity>::type
simd_for( Fn& kernel, const std::initializer_list<int>& start,
                      const std::initializer_list<int>& end ) {
  assert( *start.begin() >= 0 )
  assert( *end.begin() >= 0 && *end.begin() > *start.begin());
  Cabana::SimdPolicy<vectorLength> policy( *start.begin(), *start.end() );
  Cabana::simd_parallel_for( policy, KOKKOS_LAMBDA( const int& s, const int& a ) {
    const int i = s*vectorLength+a; // TODO: use impel_index
    kernel(i);
  }, "simd_for (rank 1)");
}
template<class Fn, std::size_t vectorLength=32>
typename 
std::enable_if<2==MeshFieldUtil::function_traits<Fn>::arity>::type
simd_for( Fn& kernel, const std::initializer_list<int>& start,
                      const std::initializer_list<int>& end ) {
  assert( *start.begin() >= 0 )
  assert( *end.begin() >= 0 && *end.begin() > *start.begin());
  Cabana::SimdPolicy<vectorLength> policy( *start.begin(), *start.end() );
  const int s1 = *( start.begin() + 1);
  const int e1 = *( end.begin() + 1);
  Cabana::simd_parallel_for( policy, KOKKOS_LAMBDA( const int& s, const int& a ) {
    const int i = s*vectorLength+a; // TODO: use impel_index
    for( int j = s1; j < e1; j++ ) kernel(i,j);
  }, "simd_for (rank 2)");
}

template< class Fn >
void launch( Fn& kernel, const std::initializer_list<int>& start, 
    const std::initializer_list<int>& end ) {
  const std::size_t v_length = 32;
  simd_for<Fn, v_length>(kernel, start, end);
}

template< typename FunctorType>
void parallel_for2(FunctorType &kernel) {
  const int n = 5;
  constexpr std::size_t RANK = function_traits<FunctorType>::arity;
  impl<RANK,FunctorType> p(kernel);
  std::vector<int> start, end;
  for( int i = 0; i < RANK; i++ ) { start.push_back(0); end.push_back(n); }
  assert( p.x( start, end ) );
}

#endif
