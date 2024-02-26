#ifndef _test_hpp_
#define _test_hpp_
#include "CabanaController.hpp"
#include "MeshField.hpp"
#include "MeshField_Utility.hpp"
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cassert>
#include <cstdio>
#include <stdio.h>
#include <type_traits>

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const> {
  static constexpr std::size_t arity = sizeof...(Args);
  typedef ReturnType result_type;
  template <std::size_t i> struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

template <std::size_t RANK, typename Type> struct impl {
  impl() {}
  virtual bool x(std::vector<int> &start, std::vector<int> &end) {
    return false;
  }
};

template <typename FunctorType> struct impl<1, FunctorType> {
  impl(FunctorType &t) : z(t) {}
  bool x(std::vector<int> &start, std::vector<int> &end) {
    assert(start.size() >= 1 && end.size() >= 1);
    for (int i = start[0]; i < end[0]; i++)
      z(i);
    return true;
  }

  FunctorType z;
};

template <typename FunctorType> struct impl<2, FunctorType> {
  impl(FunctorType &t) : z(t) {}
  bool x(std::vector<int> &start, std::vector<int> &end) {
    assert(start.size() >= 2 && end.size() >= 2);
    for (int i = start[0]; i < end[0]; i++) {
      for (int j = (start[1]); j < (end[1]); j++) {
        z(i, j);
      }
    }
    return true;
  }

  FunctorType z;
};
/*
template <class FunctorType>
typename std::enable_if< 2 == function_traits<FunctorType>::arity>::type
parallel_for(FunctorType& kernel, const std::initializer_list<int>& start, const
std::initializer_list<int>& end) { Kokkos::Array<int64_t,2> a_start =
MeshFieldUtil::to_kokkos_array<2>(start); Kokkos::Array<int64_t,2> a_end =
MeshFieldUtil::to_kokkos_array<2>(end); Kokkos::MDRangePolicy<Kokkos::Rank<2>>
p(a_start,a_end); Kokkos::parallel_for( "rank_2", p, kernel );
}
*/
/*
template <class FunctorType>
typename std::enable_if< 1 == function_traits<FunctorType>::arity>::type
parallel_for(FunctorType& kernel, const std::initializer_list<int>& start, const
std::initializer_list<int>& end) { Kokkos::RangePolicy
p(*start.begin(),*end.begin()); Kokkos::parallel_for( "rank_1", p, kernel );
}
*/

template <std::size_t vectorLen, class Fn>
typename std::enable_if<1 == function_traits<Fn>::arity>::type
simd_for(Fn kernel, const std::initializer_list<int> &start,
         const std::initializer_list<int> &end) {
  printf("rank1\n");
  Kokkos::Array<int64_t, 1> a_start = MeshFieldUtil::to_kokkos_array<1>(start);
  Kokkos::Array<int64_t, 1> a_end = MeshFieldUtil::to_kokkos_array<1>(end);
  assert(a_start[0] >= 0);
  assert(a_end[0] >= 0 && a_end[0] > a_start[0]);
  /*
  Cabana::SimdPolicy<vectorLen> policy( *start.begin(), *start.end() );
  Cabana::simd_parallel_for( policy, KOKKOS_LAMBDA( const int& s, const int& a )
  { const int i = s*vectorLen+a; // TODO: use impel_index kernel(i);
  }, "simd_for (rank 1)");
  */
}
template <std::size_t vectorLen, class Fn>
typename std::enable_if<2 == function_traits<Fn>::arity>::type
simd_for(Fn kernel, const std::initializer_list<int> &start,
         const std::initializer_list<int> &end) {
  Kokkos::Array<int64_t, 2> a_start = MeshFieldUtil::to_kokkos_array<2>(start);
  Kokkos::Array<int64_t, 2> a_end = MeshFieldUtil::to_kokkos_array<2>(end);
  assert(a_start.size() >= 2 && a_end.size() >= 2);
  assert(a_start[0] >= 0);
  assert(a_end[0] >= 0 && a_end[0] > a_start[0]);

  Cabana::SimdPolicy<vectorLen> policy(*start.begin(), *start.end());
  const int s1 = a_start[1];
  const int e1 = a_end[1];
  Cabana::simd_parallel_for(
      policy,
      KOKKOS_LAMBDA(const int &s, const int &a) {
        const int i = s * vectorLen + a; // TODO: use impel_index
        for (int j = s1; j < e1; j++)
          kernel(i, j);
      },
      "simd_for (rank 2)");
}
/*
template< typename FunctorType>
void parallel_for2(FunctorType &kernel) {
  const int n = 5;
  constexpr std::size_t RANK = function_traits<FunctorType>::arity;
  impl<RANK,FunctorType> p(kernel);
  std::vector<int> start, end;
  for( int i = 0; i < RANK; i++ ) { start.push_back(0); end.push_back(n); }
  assert( p.x( start, end ) );
}
*/
#endif
