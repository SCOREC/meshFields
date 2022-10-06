#include "SliceWrapper.hpp"

int main(int argc, char* argv[]) {
  // AoSoA parameters
  const int vecLen = 4;
  const int width = 1;
  int num_tuples = 10;
  
  Kokkos::ScopeGuard scope_guard(argc, argv);

  using DataTypes = Cabana::MemberTypes<member_type[width]>;
  
  // Slice Wrapper Factory
  CabSliceFactory<member_type, width, vecLen> cabSliceFactory(num_tuples);
  auto slice_wrapper = cabSliceFactory.makeSliceCab();
  
  // simd_parallel_for setup
  Cabana::SimdPolicy<vecLen, ExecutionSpace> simd_policy(0, num_tuples);

  // kernel that reads and writes
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
