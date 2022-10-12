#include "SliceWrapper.hpp"

int main(int argc, char* argv[]) {
  // AoSoA parameters
  int num_tuples = 10;
  
  Kokkos::ScopeGuard scope_guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  
  // Slice Wrapper Factory
  CabSliceFactory<ExecutionSpace, MemorySpace,
		  double, int, float, char> cabSliceFactory(num_tuples);
  
  auto slice_wrapper0 = cabSliceFactory.makeSliceCab<0>();
  auto slice_wrapper1 = cabSliceFactory.makeSliceCab<1>();
  auto slice_wrapper2 = cabSliceFactory.makeSliceCab<2>();
  auto slice_wrapper3 = cabSliceFactory.makeSliceCab<3>();
  
  // simd_parallel_for setup
  Cabana::SimdPolicy<cabSliceFactory.vecLen, ExecutionSpace> simd_policy(0, num_tuples);

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    printf("s: %d, a: %d\n", s,a);
    double x = 42/(s+a+1.3);
    slice_wrapper0.access(s,a) = x;
    slice_wrapper1.access(s,a) = s+a;
    slice_wrapper2.access(s,a) = float(x);
    slice_wrapper3.access(s,a) = 'a'+s+a;
    printf("SW0 value: %lf\n", slice_wrapper0.access(s,a));
    printf("SW1 value: %d\n", slice_wrapper1.access(s,a));
    printf("SW2 value: %f\n", slice_wrapper2.access(s,a));
    printf("SW3 value: %c\n", slice_wrapper3.access(s,a));
  };

  Cabana::simd_parallel_for(simd_policy, vector_kernel, "parallel_for_cabSliceFactory");

  assert(cudaSuccess == cudaDeviceSynchronize());
  printf("done\n");

  return 0;
}
