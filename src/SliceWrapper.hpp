#ifndef slicewrapper_hpp
#define slicewrapper_hpp

#include <Cabana_Core.hpp>

template< class SliceType, class T >
struct SliceWrapper {

  SliceType st_; //store the underlying instance

  SliceWrapper(SliceType st) : st_(st)  {}
  
  KOKKOS_INLINE_FUNCTION
  T& access(const int s, const int a) const {
    return st_.access(s,a);
  }
  int arraySize(int s) {
    return st_.arraySize(s);
  }
  int numSoA() {
    return st_.numSoA();
  }
};

using namespace Cabana;

template <class ExecutionSpace, class MemorySpace, int vecLen, class... Ts>
class CabSliceFactory {
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;

  template <class T>
  using member_slice_t = 
    Cabana::Slice<T, DeviceType, 
		  Cabana::DefaultAccessMemory, 
		  vecLen, vecLen>;

  template <class T>
  using wrapper_slice_t = SliceWrapper<member_slice_t<T>, T>;

  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa; 
  
public:
  template <std::size_t index>
  auto makeSliceCab() {
    auto slice = Cabana::slice<index>(aosoa);
    using type = typename std::tuple_element<index, TypeTuple>;
    return wrapper_slice_t<type>(std::move(slice));
  }
  
  CabSliceFactory(int n) : aosoa("sliceAoSoA", n) {}
};


#endif

