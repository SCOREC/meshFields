#include "CabanaSliceWrapper.hpp"
#include <stdio.h>

#define TOLERANCE 1e-10;

KOKKOS_INLINE_FUNCTION
bool doubleCompare(double d1, double d2) {
  double diff = fabs(d1 - d2);
  return diff < TOLERANCE;
}

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

int rank1_array_test(int num_tuples) {

  const int width = 3;

  // Slice Wrapper Controller
  SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace, double[width]>
      cabSliceController(num_tuples);

  auto slice_wrapper0 = cabSliceController.makeSlice<0>();

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      double x = 42 / (s + a + i + 1.3);
      slice_wrapper0.access(s, a, i) = x;
      assert(doubleCompare(slice_wrapper0.access(s, a, i), x));
    }
  };
  cabSliceController.parallel_for(0, num_tuples, vector_kernel,
                                  "parallel_for_rank1_array");
  return 0;
}

int rank2_array_test(int num_tuples) {

  const int width = 3;
  const int height = 4;

  // Slice Wrapper Controller
  SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace,
                                   double[width][height]>
      cabSliceController(num_tuples);

  auto slice_wrapper0 = cabSliceController.makeSlice<0>();

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        double x = 42 / (s + a + i + j + 1.3);
        slice_wrapper0.access(s, a, i, j) = x;
        assert(doubleCompare(slice_wrapper0.access(s, a, i, j), x));
      }
    }
  };
  cabSliceController.parallel_for(0, num_tuples, vector_kernel,
                                  "parallel_for_rank2_array");
  return 0;
}

int rank3_array_test(int num_tuples) {

  const int width = 3;
  const int height = 4;
  const int depth = 2;

  // Slice Wrapper Controller
  SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace,
                                   double[width][height][depth]>
      cabSliceController(num_tuples);

  auto slice_wrapper0 = cabSliceController.makeSlice<0>();
  
  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        for (int k = 0; k < depth; k++) {
          double x = 42 / (s + a + i + j + k + 1.3);
          slice_wrapper0.access(s, a, i, j, k) = x;
          assert(doubleCompare(slice_wrapper0.access(s, a, i, j, k), x));
        }
      }
    }
  };
  cabSliceController.parallel_for(0, num_tuples, vector_kernel,
                                  "parallel_for_rank3_array");
  return 0;
}

int mix_arrays_test(int num_tuples) {

  const int width = 3;
  const int height = 4;
  const int depth = 2;

  // Slice Wrapper Controller
  SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace, double[width],
                                   char, double[width][height][depth],
                                   float[width][height]>
      cabSliceController(num_tuples);

  auto slice_wrapper0 = cabSliceController.makeSlice<0>();
  auto slice_wrapper1 = cabSliceController.makeSlice<1>();
  auto slice_wrapper2 = cabSliceController.makeSlice<2>();
  auto slice_wrapper3 = cabSliceController.makeSlice<3>();

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    char x0 = 'a' + (s + a);
    slice_wrapper1.access(s, a) = x0;
    assert(slice_wrapper1.access(s, a) == x0);

    for (int i = 0; i < width; i++) {
      double x1 = 42 / (s + a + i + 1.3);
      slice_wrapper0.access(s, a, i) = x1;
      assert(doubleCompare(slice_wrapper0.access(s, a, i), x1));

      for (int j = 0; j < height; j++) {
        double x2 = float(x1 / (j + 1.2));
        slice_wrapper3.access(s, a, i, j) = x2;
        assert(doubleCompare(slice_wrapper3.access(s, a, i, j), x2));

        for (int k = 0; k < depth; k++) {
          double x3 = x2 * x1 + k;
          slice_wrapper2.access(s, a, i, j, k) = x3;
          assert(doubleCompare(slice_wrapper2.access(s, a, i, j, k), x3));
        }
      }
    }
  };
  cabSliceController.parallel_for(0, num_tuples, vector_kernel,
                                  "parallel_for_mix_array");
  return 0;
}

int single_type_test(int num_tuples) {

  // Slice Wrapper Controller
  SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace, double>
      cabSliceController(num_tuples);

  auto slice_wrapper0 = cabSliceController.makeSlice<0>();

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double x = 42 / (s + a + 1.3);
    slice_wrapper0.access(s, a) = x;
    assert(doubleCompare(slice_wrapper0.access(s, a), x));
  };
  cabSliceController.parallel_for(0, num_tuples, vector_kernel,
                                  "parallel_for_single_type");

  return 0;
}

int multi_type_test(int num_tuples) {

  // Slice Wrapper Controller
  SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace, double, int,
                                   float, char>
      cabSliceController(num_tuples);

  auto slice_wrapper0 = cabSliceController.makeSlice<0>();
  auto slice_wrapper1 = cabSliceController.makeSlice<1>();
  auto slice_wrapper2 = cabSliceController.makeSlice<2>();
  auto slice_wrapper3 = cabSliceController.makeSlice<3>();

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double x = 42 / (s + a + 1.3);
    char c = 'a' + s + a;
    slice_wrapper0.access(s, a) = x;
    slice_wrapper1.access(s, a) = s + a;
    slice_wrapper2.access(s, a) = float(x);
    slice_wrapper3.access(s, a) = c;

    assert(doubleCompare(slice_wrapper0.access(s, a), x));
    assert(slice_wrapper1.access(s, a) == s + a);
    assert(doubleCompare(slice_wrapper2.access(s, a), float(x)));
    assert(slice_wrapper3.access(s, a) == c);
  };
  cabSliceController.parallel_for(0, num_tuples, vector_kernel,
                                  "parallel_for_multi_type");

  return 0;
}

int many_type_test(int num_tuples) {

  // Slice Wrapper Controller
  SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace, long double,
                                   double, double, long unsigned int, float,
                                   int, short int, char, char>
      cabSliceController(num_tuples);

  auto slice_wrapper0 = cabSliceController.makeSlice<0>();
  auto slice_wrapper1 = cabSliceController.makeSlice<1>();
  auto slice_wrapper2 = cabSliceController.makeSlice<2>();
  auto slice_wrapper3 = cabSliceController.makeSlice<3>();
  auto slice_wrapper4 = cabSliceController.makeSlice<4>();
  auto slice_wrapper5 = cabSliceController.makeSlice<5>();
  auto slice_wrapper6 = cabSliceController.makeSlice<6>();
  auto slice_wrapper7 = cabSliceController.makeSlice<7>();
  auto slice_wrapper8 = cabSliceController.makeSlice<8>();

  // kernel that reads and writes
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double d0 = 42 / (s + a + 1.3);
    double d1 = d0 * d0;
    double d2 = d1 / 123.456751;
    float f0 = 0;
    char c0 = 'a' + s + a;
    char c1 = 'a' + ((s * a + a) % 26);
    int i0 = s + a;
    int i1 = s + a / int(d0);
    int i2 = i0 + i1;
    slice_wrapper0.access(s, a) = d0;
    slice_wrapper1.access(s, a) = d1;
    slice_wrapper2.access(s, a) = d2;
    slice_wrapper3.access(s, a) = i0;
    slice_wrapper4.access(s, a) = f0;
    slice_wrapper5.access(s, a) = i1;
    slice_wrapper6.access(s, a) = i2;
    slice_wrapper7.access(s, a) = c0;
    slice_wrapper8.access(s, a) = c1;

    assert(doubleCompare(slice_wrapper0.access(s, a), d0));
    assert(doubleCompare(slice_wrapper1.access(s, a), d1));
    assert(doubleCompare(slice_wrapper2.access(s, a), d2));
    assert(slice_wrapper3.access(s, a) == i0);
    assert(doubleCompare(slice_wrapper4.access(s, a), f0));
    assert(slice_wrapper5.access(s, a) == i1);
    assert(slice_wrapper6.access(s, a) == i2);
    assert(slice_wrapper7.access(s, a) == c0);
    assert(slice_wrapper8.access(s, a) == c1);
  };
  cabSliceController.parallel_for(0, num_tuples, vector_kernel,
                                  "parallel_for_many_type");
  return 0;
}

int main(int argc, char *argv[]) {
  // AoSoA parameters
  int num_tuples = atoi(argv[1]);

  Kokkos::ScopeGuard scope_guard(argc, argv);

  single_type_test(num_tuples);
  multi_type_test(num_tuples);
  many_type_test(num_tuples);

  rank1_array_test(num_tuples);
  rank2_array_test(num_tuples);
  rank3_array_test(num_tuples);
  mix_arrays_test(num_tuples);

  return 0;
}
