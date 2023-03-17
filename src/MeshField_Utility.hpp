#ifndef MESHFIELD_UTILITY_HPP
#define MESHFIELD_UTILITY_HPP

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
  static constexpr std::size_t arity = sizeof...(Args);
  typedef ReturnType result_type;
  template<std::size_t i> struct arg {
    typedef typename std::tuple_element<i,std::tuple<Args...>>::type type;
  };
};





} // END NAMESPACE MeshField



#endif // #ifndef MESHFIELD_UTILITY_HPP
