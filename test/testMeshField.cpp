#include "MeshField.hpp"
#include "SliceWrapper.hpp"

#include <Cabana_Core.hpp>

#define TOLERANCE 1e-10;

// helper testing functions

// compare doubles within a tolerance
KOKKOS_INLINE_FUNCTION
bool doubleCompare(double d1, double d2) {
  double diff = fabs(d1 - d2);
  return diff < TOLERANCE;
}

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

// sum of all values from 0 to n-1
KOKKOS_INLINE_FUNCTION
int simpleSum(int n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    sum += i;
  }
  return sum;
}

void test_scan(int num_tuples) {
  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace, int, int>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  // create fields
  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();

  // create views to initialize our fields with
  Kokkos::View<int *> initView0("InitView0", num_tuples);
  Kokkos::View<int *> initView1("InitView1", num_tuples);
  Kokkos::parallel_for(
      "InitViewLoop", num_tuples, KOKKOS_LAMBDA(const int &i) {
        initView0(i) = i;
        initView1(i) = i;
      });

  // initialize fields with views
  cabMeshField.setField(field0, initView0);
  cabMeshField.setField(field1, initView1);

  // create result view
  Kokkos::View<int *> resultView0("ScanView0", num_tuples + 1);
  Kokkos::View<int *> resultView1("ScanView1", num_tuples);

  // define exclusive scan kernel
  auto indexToSA = c.indexToSA;
  auto binOp0 = KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
    int s, a;
    indexToSA(i, s, a);
    if (is_final) {
      resultView0(i) = partial_sum;
    }
    partial_sum += field0(s, a);
  };

  // define inclusive scan kernel
  auto binOp1 = KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
    int s, a;
    indexToSA(i, s, a);
    partial_sum += field0(s, a);
    if (is_final) {
      resultView1(i) = partial_sum;
    }
  };

  // do scans
  cabMeshField.parallel_scan(binOp0, "parallel_scan0");
  cabMeshField.parallel_scan(binOp1, "parallel_scan1");

  // create test data
  std::vector<int> testData(num_tuples);
  std::iota(testData.begin(), testData.end(), 0);
  std::vector<int> scanResult0(num_tuples);
  std::vector<int> scanResult1(num_tuples);

  // run std scans to compare results
  std::exclusive_scan(testData.begin(), testData.end(), scanResult0.begin(), 0);
  std::inclusive_scan(testData.begin(), testData.end(), scanResult1.begin());

  // add final sum manually because std::exclusive_scan doesn't
  scanResult0.push_back(simpleSum(num_tuples));

  // convert results of std scans into host views
  Kokkos::View<int *, Kokkos::HostSpace> h_expectedView0(
      std::string("host_expectedView0"), num_tuples + 1);
  Kokkos::View<int *, Kokkos::HostSpace> h_expectedView1(
      std::string("host_expectedView1"), num_tuples);

  for (int i = 0; i < num_tuples + 1; i++) {
    h_expectedView0(i) = scanResult0[i];
  }

  for (int i = 0; i < num_tuples + 1; i++) {
    h_expectedView1(i) = scanResult1[i];
  }

  // create device views copied from host views to be used in the final
  auto expectedView0 =
      Kokkos::create_mirror_view_and_copy(ExecutionSpace(), h_expectedView0);
  auto expectedView1 =
      Kokkos::create_mirror_view_and_copy(ExecutionSpace(), h_expectedView1);

  // final validity check
  assert(Kokkos::Experimental::equal(Kokkos::DefaultExecutionSpace(),
                                     expectedView0, resultView0));
  assert(Kokkos::Experimental::equal(Kokkos::DefaultExecutionSpace(),
                                     expectedView1, resultView1));
}

void test_reductions(int num_tuples) {
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace,
                                                      MemorySpace, double, int>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();

  Kokkos::View<double *> initView0("InitView0", num_tuples);
  Kokkos::View<int *> initView1("InitView1", num_tuples);
  Kokkos::parallel_for(
      "InitViewLoop", num_tuples, KOKKOS_LAMBDA(const int &i) {
        initView0(i) = i;
        initView1(i) = i;
      });

  cabMeshField.setField(field0, initView0);
  cabMeshField.setField(field1, initView1);

  // double reductions
  {
    double sum = cabMeshField.sum(field0);
    double expected_sum = static_cast<double>(simpleSum(num_tuples));
    assert(doubleCompare(sum, expected_sum));

    double mean = cabMeshField.mean(field0);
    double expected_mean = expected_sum / num_tuples;
    assert(doubleCompare(mean, expected_mean));

    double min = cabMeshField.min(field0);
    double expected_min = 0;
    assert(doubleCompare(min, expected_min));

    double max = cabMeshField.max(field0);
    double expected_max = num_tuples - 1;
    assert(doubleCompare(max, expected_max));
  }

  // int reductions
  {
    int sum = cabMeshField.sum(field1);
    int expected_sum = simpleSum(num_tuples);
    assert(sum == expected_sum);

    double mean = cabMeshField.mean(field1);
    double expected_mean = static_cast<double>(expected_sum) / num_tuples;
    assert(doubleCompare(mean, expected_mean));

    int min = cabMeshField.min(field1);
    int expected_min = 0;
    assert(min == expected_min);

    int max = cabMeshField.max(field1);
    int expected_max = num_tuples - 1;
    assert(max == expected_max);
  }

  // user defined reductions
  {
    double result;
    auto indexToSA = c.indexToSA;
    auto reduce_kernel = KOKKOS_LAMBDA(const int &i, double &lresult) {
      int s, a;
      indexToSA(i, s, a);
      lresult += field0(s, a) * 10;
    };
    cabMeshField.parallel_reduce(reduce_kernel, result, "user_def_reduce");
    int expected_result = 10 * simpleSum(num_tuples);
    assert(doubleCompare(result, expected_result));
  }
}

void single_type(int num_tuples) {
  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace, double>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double d0 = 10;
    field0(s, a) = d0;
    assert(doubleCompare(field0(s, a), d0));
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "single_type_pfor");
}

void multi_type(int num_tuples) {
  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace, double,
                                       double, float, int, char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();
  auto field4 = cabMeshField.makeField<4>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double d0 = 10.456;
    field0(s, a) = d0;
    double d1 = 43.973234567;
    field1(s, a) = d1;
    float f0 = 123.45;
    field2(s, a) = f0;
    int i0 = 22;
    field3(s, a) = i0;
    char c0 = 'a';
    field4(s, a) = c0;

    assert(doubleCompare(field0(s, a), d0));
    assert(doubleCompare(field1(s, a), d1));
    assert(doubleCompare(field2(s, a), f0));
    assert(field3(s, a) == i0);
    assert(field4(s, a) == c0);
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "multi_type_pfor");
}

void many_type(int num_tuples) {

  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace, double,
                                       double, float, float, int, short int,
                                       char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();
  auto field4 = cabMeshField.makeField<4>();
  auto field5 = cabMeshField.makeField<5>();
  auto field6 = cabMeshField.makeField<6>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double d0 = 10.456;
    field0(s, a) = d0;
    double d1 = 43.973234567;
    field1(s, a) = d1;
    float f0 = 123.45;
    field2(s, a) = f0;
    float f1 = 543.21;
    field3(s, a) = f1;
    int i0 = 222;
    field4(s, a) = i0;
    short int i1 = 50;
    field5(s, a) = i1;
    char c0 = 'h';
    field6(s, a) = c0;

    assert(doubleCompare(field0(s, a), d0));
    assert(doubleCompare(field1(s, a), d1));
    assert(doubleCompare(field2(s, a), f0));
    assert(doubleCompare(field3(s, a), f1));
    assert(field4(s, a) == i0);
    assert(field5(s, a) == i1);
    assert(field6(s, a) == c0);
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "many_type_pfor");
}

void rank1_arr(int num_tuples) {
  const int width = 3;
  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
                                       double[width]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      double d0 = 10 + i;
      field0(s, a, i) = d0;
      assert(doubleCompare(field0(s, a, i), d0));
    }
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "rank1_arr_pfor");
}

void rank2_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
                                       double[width][height]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        double d0 = (10 + i) / (j + 1);
        field0(s, a, i, j) = d0;
        assert(doubleCompare(field0(s, a, i, j), d0));
      }
    }
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "rank2_arr_pfor");
}

void rank3_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  const int depth = 2;
  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
                                       double[width][height][depth]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        for (int k = 0; k < depth; k++) {
          double d0 = ((10 + i) * (k + 1)) / (j + 1);
          field0(s, a, i, j, k) = d0;
          assert(doubleCompare(field0(s, a, i, j, k), d0));
        }
      }
    }
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "rank3_arr_pfor");
}

void mix_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  const int depth = 2;
  using Controller =
      SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
                                       double[width][height][depth],
                                       float[width][height], int[width], char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    float f0;
    int i0;
    char c0 = 's';
    field3(s, a) = c0;

    for (int i = 0; i < width; i++) {
      i0 = i + s + a;
      field2(s, a, i) = i0;
      for (int j = 0; j < height; j++) {
        f0 = i0 / (i + j + 1.123);
        field1(s, a, i, j) = f0;
        for (int k = 0; k < depth; k++) {
          double d0 = ((10 + i) * (k + 1)) / (j + 1);
          field0(s, a, i, j, k) = d0;
          assert(doubleCompare(field0(s, a, i, j, k), d0));
        }
        assert(doubleCompare(field1(s, a, i, j), f0));
      }
      assert(field2(s, a, i) == i0);
    }
    assert(field3(s, a) == c0);
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "mix_arr_pfor");
}

int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);

  single_type(num_tuples);
  multi_type(num_tuples);
  many_type(num_tuples);
  rank1_arr(num_tuples);
  rank2_arr(num_tuples);
  rank3_arr(num_tuples);
  mix_arr(num_tuples);

  test_reductions(num_tuples);
  test_scan(num_tuples);

  return 0;
}
