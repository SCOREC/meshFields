#include "MeshField_TestHarness.hpp"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <string>

using MemorySpace = ExecutionSpace::memory_space;
struct KokkosBenchmark {
  static constexpr char const *name = "Kokkos";
  using Storage =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, double *[3]>;
};
#ifdef MESHFIELDS_ENABLE_CABANA
struct CabanaBenchmark {
  static constexpr char const *name = "Cabana";
  using Storage =
      MeshField::CabanaController<ExecutionSpace, MemorySpace, double[3]>;
};
using BenchmarkCases = std::tuple<KokkosBenchmark, CabanaBenchmark>;
#else
using BenchmarkCases = std::tuple<KokkosBenchmark>;
#endif

// Keep device lambdas in named functions for nvcc. Fence inside every timed
// iteration: the measurement includes launch and completion, not just enqueue.
template <typename Field> void writeCoordinates(Field field, int n) {
  Kokkos::parallel_for(
      "benchmarkWrite", n, KOKKOS_LAMBDA(const int i) {
        for (int d = 0; d < 3; ++d)
          field(i, d) = i + d;
      });
  ExecutionSpace().fence();
}

template <typename Field> int coordinateErrors(Field field, int n) {
  int errors = 0;
  Kokkos::parallel_reduce(
      "benchmarkCheck", n,
      KOKKOS_LAMBDA(const int i, int &sum) {
        for (int d = 0; d < 3; ++d)
          sum += field(i, d) != i + d;
      },
      errors);
  return errors;
}

TEMPLATE_LIST_TEST_CASE("controller coordinate writes",
                        "[!benchmark][controller]", BenchmarkCases) {
  constexpr int n = 65536;
  using Storage = typename TestType::Storage;
  Storage storage({n});
  auto field = MeshField::makeField<Storage, 0>(storage);
  REQUIRE(field.size(0) == n);
  REQUIRE(field.size(1) == 3);
  INFO("execution space: " << ExecutionSpace::name());
  INFO("entries: " << n
                   << ", components: 3, bytes: " << n * 3 * sizeof(double));
  // Allocation, warmup, and verification are outside the measured region.
  writeCoordinates(field, n);
  REQUIRE(coordinateErrors(field, n) == 0);
  BENCHMARK(std::string(TestType::name) + " / " + ExecutionSpace::name() +
            " / write 65536 coordinates and fence") {
    writeCoordinates(field, n);
  };
  CHECK(coordinateErrors(field, n) == 0);
}
