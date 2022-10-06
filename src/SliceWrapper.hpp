#ifndef slicewrapper_hpp
#define slicewrapper_hpp

#include <Cabana_Core.hpp>

template< class SliceType, class T >
struct SliceWrapper {

  SliceType st_; //store the underlying instance

  SliceWrapper(SliceType st) : st_(st)  {}
  
  KOKKOS_INLINE_FUNCTION
  T& access(const int s, const int a, int i) const {
    return st_.access(s,a,i);
  }
  int arraySize(int s) {
    return st_.arraySize(s);
  }
  int numSoA() {
    return st_.numSoA();
  }
};

using namespace Cabana;

template <class ExecutionSpace, class MemorySpace, class T, int width, int vecLen>
class CabSliceFactory {
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<T[width]>;
  using member_slice_t = 
    Cabana::Slice<T[width], DeviceType, 
		  Cabana::DefaultAccessMemory, 
		  vecLen, width*vecLen>;
  using wrapper_slice_t = SliceWrapper<member_slice_t, T>;

  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa; 
  
public:
  wrapper_slice_t makeSliceCab() {
    auto slice0 = Cabana::slice<0>(aosoa);
    return wrapper_slice_t(std::move(slice0));
  }
  CabSliceFactory(int n) : aosoa("sliceAoSoA", n) {}
};


#endif

