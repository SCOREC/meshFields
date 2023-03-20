#ifndef MESHFIELD_UTILITY_HPP
#define MESHFIELD_UTILITY_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <tuple>
#include <initializer_list>
/*
stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda
*/
namespace MeshFieldUtil
{

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

template< std::size_t RANK, class T >
Kokkos::Array<int64_t, RANK> 
to_kokkos_array( const std::initializer_list<T>& item) {
  assert( std::is_integral<T>::value );
  Kokkos::Array<int64_t, RANK> rt{};
  auto x = item.begin();
  for( std::size_t i = 0; i < RANK; i++ ) {
    rt[i] = (*x);
    x++;
  }
  return rt;
}



} // END NAMESPACE MeshField



#endif // #ifndef MESHFIELD_UTILITY_HPP
