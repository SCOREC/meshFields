#include "SliceWrapper.hpp"
#include <stdio.h>

int array_type_test(int num_tuples) {
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  const int width = 3;
  
  // Slice Wrapper Factory
  CabSliceFactory<ExecutionSpace, MemorySpace,
		  double[width]> cabSliceFactory(num_tuples);
  
  auto slice_wrapper0 = cabSliceFactory.makeSliceCab<0>();
  
  // simd_parallel_for setup
  Cabana::SimdPolicy<cabSliceFactory.vecLen, ExecutionSpace> simd_policy(0, num_tuples);

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      double x = 42/(s+a+i+1.3);
      slice_wrapper0.access(s,a,i) = x;
      assert(slice_wrapper0.access(s,a,i) == x);
      printf("SW0 value: %lf\n", slice_wrapper0.access(s,a,i));
    }
  };

  Cabana::simd_parallel_for(simd_policy, vector_kernel, "parallel_for_array_type_test");
  return 0;
  
}

int single_type_test(int num_tuples) {
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  
  // Slice Wrapper Factory
  CabSliceFactory<ExecutionSpace, MemorySpace,
		  double> cabSliceFactory(num_tuples);
  
  auto slice_wrapper0 = cabSliceFactory.makeSliceCab<0>();
  
  // simd_parallel_for setup
  Cabana::SimdPolicy<cabSliceFactory.vecLen, ExecutionSpace> simd_policy(0, num_tuples);

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double x = 42/(s+a+1.3);
    slice_wrapper0.access(s,a) = x;
    
    printf("SW0 value: %lf\n", slice_wrapper0.access(s,a));
  };

  Cabana::simd_parallel_for(simd_policy, vector_kernel, "parallel_for_single_type_test");
  return 0;
}

int multi_type_test(int num_tuples) {
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

  Cabana::simd_parallel_for(simd_policy, vector_kernel, "parallel_for_multi_type_test");
  return 0;
}

int many_type_test(int num_tuples) {
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  
  // Slice Wrapper Factory
  CabSliceFactory<ExecutionSpace, MemorySpace,
		  double, short int, float, char, int,
		  char, double, long unsigned int,
		  long double> cabSliceFactory(num_tuples);
  
  auto slice_wrapper0 = cabSliceFactory.makeSliceCab<0>();
  auto slice_wrapper1 = cabSliceFactory.makeSliceCab<1>();
  auto slice_wrapper2 = cabSliceFactory.makeSliceCab<2>();
  auto slice_wrapper3 = cabSliceFactory.makeSliceCab<3>();
  auto slice_wrapper4 = cabSliceFactory.makeSliceCab<4>();
  auto slice_wrapper5 = cabSliceFactory.makeSliceCab<5>();
  auto slice_wrapper6 = cabSliceFactory.makeSliceCab<6>();
  auto slice_wrapper7 = cabSliceFactory.makeSliceCab<7>();
  auto slice_wrapper8 = cabSliceFactory.makeSliceCab<8>();
  
  // simd_parallel_for setup
  Cabana::SimdPolicy<cabSliceFactory.vecLen, ExecutionSpace> simd_policy(0, num_tuples);

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double x = 42/(s+a+1.3);
    slice_wrapper0.access(s,a) = x;
    slice_wrapper1.access(s,a) = s+a;
    slice_wrapper2.access(s,a) = float(x);
    slice_wrapper3.access(s,a) = 'a'+s+a;
    slice_wrapper4.access(s,a) = int(s+a/x);
    slice_wrapper5.access(s,a) = 'a'+((s*a+a) % 26);
    slice_wrapper6.access(s,a) = (s+a+a+s*s)*x;
    slice_wrapper7.access(s,a) = (s+a)*num_tuples/(s+2);
    slice_wrapper8.access(s,a) = (x+s+a)/(x*x);
    printf("SW0 value: %lf\n", slice_wrapper0.access(s,a));
    printf("SW1 value: %d\n", slice_wrapper1.access(s,a));
    printf("SW2 value: %f\n", slice_wrapper2.access(s,a));
    printf("SW3 value: %c\n", slice_wrapper3.access(s,a));
    printf("SW4 value: %d\n", slice_wrapper4.access(s,a));
    printf("SW5 value: %c\n", slice_wrapper5.access(s,a));
    printf("SW6 value: %lf\n", slice_wrapper6.access(s,a));
    printf("SW7 value: %lu\n", slice_wrapper7.access(s,a));
    printf("SW8 value: %Lf\n", slice_wrapper8.access(s,a));
  };

  Cabana::simd_parallel_for(simd_policy, vector_kernel, "parallel_for_many_type_test");
  return 0;
}


int main(int argc, char* argv[]) {
  // AoSoA parameters
  int num_tuples = 50;
  
  Kokkos::ScopeGuard scope_guard(argc, argv);

  no_type_test(num_tuples);
  single_type_test(num_tuples);
  multi_type_test(num_tuples);
  many_type_test(num_tuples);
  1d_array_test(num_tuples);
  2d_array_test(num_tuples);
  3d_array_test(num_tuples);
  mix_arrays_test(num_tuples);
  
  
  assert(cudaSuccess == cudaDeviceSynchronize());
  printf("done\n");

  return 0;
}
