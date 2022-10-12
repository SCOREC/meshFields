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

template <class ExecutionSpace, class MemorySpace, class... Ts>
class CabSliceFactory {
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;
public:
  static constexpr int vecLen = Cabana::AoSoA<DataTypes, DeviceType>::vector_length;
private:
  using soa_t = SoA<DataTypes, vecLen>;
  
  template <class T, int stride>
  using member_slice_t = 
    Cabana::Slice<T, DeviceType, 
		  Cabana::DefaultAccessMemory, 
		  vecLen, stride>;

  template <class T, int stride>
  using wrapper_slice_t = SliceWrapper<member_slice_t<T, stride>, T>;

  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa; 
  
public:
  template <std::size_t index>
  auto makeSliceCab() {
    using type = std::tuple_element_t<index, TypeTuple>;
    const int stride = sizeof(soa_t) / sizeof(type);
    auto slice = Cabana::slice<index>(aosoa);
    return wrapper_slice_t< type, stride >(std::move(slice));
  }
  
  CabSliceFactory(int n) : aosoa("sliceAoSoA", n) {}
};


#endif

