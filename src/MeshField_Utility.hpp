#ifndef MESHFIELD_UTILITY_HPP
#define MESHFIELD_UTILITY_HPP

#include <initializer_list>
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <tuple>
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
  // arity is used to find the number of arguements
  // used in a lambda function. Refer to stackoverflow link above.
  static constexpr std::size_t arity = sizeof...(Args);
  typedef ReturnType result_type;
  template<std::size_t i> struct arg {
    typedef typename std::tuple_element<i,std::tuple<Args...>>::type type;
  };
};

template< std::size_t RANK, class T >
Kokkos::Array<int64_t, RANK> 
to_kokkos_array( const std::initializer_list<T>& item) {
  // Convert initializer_list to a Kokkos::Array<RANK>
  assert( std::is_integral<T>::value );
  Kokkos::Array<int64_t, RANK> rt{};
  auto x = item.begin();
  for( std::size_t i = 0; i < RANK; i++ ) {
    rt[i] = (*x);
    x++;
  }
  return rt;
}

template<typename T>
struct identity {
  using type = T;
};
template<typename T>
struct remove_all_pointers : std::conditional_t<
    std::is_pointer_v<T>,
    remove_all_pointers<
        std::remove_pointer_t<T>
    >,
    identity<T>
>
{};

} // END NAMESPACE MeshFieldUtil


#endif // #ifndef MESHFIELD_UTILITY_HPP
