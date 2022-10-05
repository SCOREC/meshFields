#include <Cabana_Core.hpp>

using member_type = double;
using MemorySpace = Kokkos::CudaSpace;
using ExecutionSpace = Kokkos::Cuda;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

template< class SliceType >
struct SliceWrapper {

  SliceType st_; //store the underlying instance

  SliceWrapper(SliceType st) : st_(st)  {}
  
  KOKKOS_INLINE_FUNCTION
  member_type& access(const int s, const int a, int i) const {
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

template <class T, int width, int vecLen>
class CabSliceFactory {
  using DataTypes = Cabana::MemberTypes<T[width]>;
  using member_slice_t = 
    Cabana::Slice<T[width], DeviceType, 
		  Cabana::DefaultAccessMemory, 
		  vecLen, width*vecLen>;
  using wrapper_slice_t = SliceWrapper<member_slice_t>;

  Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa; 
  
public:
  wrapper_slice_t makeSliceCab() {
    auto slice0 = Cabana::slice<0>(aosoa);
    return wrapper_slice_t(std::move(slice0));
  }
  CabSliceFactory(int n) : aosoa("sliceAoSoA", n) {}
};


int main(int argc, char* argv[]) {
  // AoSoA parameters
  const int vecLen = 4;
  const int width = 1;
  int num_tuples = 10;
  
  Kokkos::ScopeGuard scope_guard(argc, argv);

  using DataTypes = Cabana::MemberTypes<member_type[width]>;
  
  // Cabana::AoSoA<DataTypes, DeviceType, vecLen> aosoa("unwrapped_aosoa", num_tuples);
  //auto slice = Cabana::slice<0>(aosoa);
  
  
  
  // Slice Wrapper Factory
  CabSliceFactory<member_type, width, vecLen> cabSliceFactory(num_tuples);
  auto slice_wrapper = cabSliceFactory.makeSliceCab();
  
  // simd_parallel_for setup
  Cabana::SimdPolicy<vecLen, ExecutionSpace> simd_policy(0, num_tuples);
  
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      printf("s: %d, a: %d, i: %d\n", s,a,i);
      double x = 42/(s+a+1.3);
      slice_wrapper.access(s,a,i) = x;
      printf("value: %lf\n", slice_wrapper.access(s,a,i));
    }
  };

  Cabana::simd_parallel_for(simd_policy, vector_kernel, "parallel_for_cabSliceFactory");

  assert(cudaSuccess == cudaDeviceSynchronize());
  printf("done\n");
  

  return 0;
}
