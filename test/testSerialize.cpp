#include "MeshField_TestFields.hpp"
#include "MeshField_TestHarness.hpp"

using MeshField::LO;

/*
 * serialize a field of rank R, check the row-major order of the result, and
 * deserialize it back
 */
template <int R, typename Field> void testSerialize(Field field, Extents n) {
  INFO("rank " << R);
  fill<R>(field, n);
  const auto serial = field.serialize();

  LO expectedSize = 1;
  for (int d = 0; d < R; ++d)
    expectedSize *= n[d];
  REQUIRE(serial.extent(0) == expectedSize);

  // entry p of the result is the entry at the row-major index p
  LO orderErrors = 0;
  Kokkos::parallel_reduce(
      "checkSerialOrder", serial.extent(0),
      KOKKOS_LAMBDA(const int p, LO &lerr) {
        Extents idx = {0, 0, 0, 0, 0};
        int rest = p;
        for (int d = R - 1; d >= 0; --d) {
          idx[d] = rest % n[d];
          rest /= n[d];
        }
        lerr += serial(p) != indexValue(idx[0], idx[1], idx[2], idx[3], idx[4]);
      },
      orderErrors);
  CHECK(orderErrors == 0);

  fill<R>(field, n, 1); // so that a deserialize that does nothing fails
  field.deserialize(serial);
  CHECK(mismatches<R>(field, n) == 0);
}

// Field::serialize supports only data types without static extents, so only
// Kokkos controllers with runtime extents are tested
TEMPLATE_TEST_CASE_SIG("serialize runtime extents", "[serialize]",
                       ((typename T, int R), T, R), (int, 1), (int, 2),
                       (int, 3), (int, 4), (int, 5), (double, 1), (double, 2),
                       (double, 3), (double, 4), (double, 5)) {
  using Ctrlr = MeshField::KokkosController<MemorySpace, ExecutionSpace, T *,
                                            T **, T ***, T ****, T *****>;
  Ctrlr c({rank1[0], /**/ rank2[0], Y, /**/ rank3[0], Y, Z,
           /**/ rank4[0], Y, Z, A, /**/ rank5[0], Y, Z, A, B});
  constexpr Extents extents[] = {rank1, rank2, rank3, rank4, rank5};
  testSerialize<R>(MeshField::makeField<Ctrlr, R - 1>(c), extents[R - 1]);
}

// the shape of the fields CreateLagrangeField creates
TEST_CASE("serialize Lagrange shape", "[serialize]") {
  using LagrangeCtrlr =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int ***>;
  LagrangeCtrlr lagrange({1, 1, 16});
  testSerialize<3>(MeshField::makeField<LagrangeCtrlr, 0>(lagrange),
                   {1, 1, 16});
}

// Field::serialize and deserialize are currently broken for data types with
// fixed (static) extents: Field::base_type removes only the pointers, so e.g.
// int*[3] keeps its [3] and these do not compile.  That includes every Cabana
// field above rank 1.  Catch2 has no way to register a test case that is
// expected not to compile, so this case stays commented out.
// TEST_CASE("serialize fixed extents", "[serialize][!mayfail]") {
//   using KokkosInt = KokkosRanks<int>;
//   auto fixed = createRanks<KokkosInt>();
//   testSerialize<1>(MeshField::makeField<KokkosInt, 0>(fixed), rank1);
//   ...
// }
