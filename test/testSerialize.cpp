#include "MeshField_TestFields.hpp"
#include <string>

using MeshField::LO;

/*
 * serialize a field of rank R, check the row-major order of the result, and
 * deserialize it back
 */
template <int R, typename Field> void testSerialize(Field field, Extents n) {
  fill<R>(field, n);
  const auto serial = field.serialize();

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
        lerr += serial(p) !=
                indexValue(idx[0], idx[1], idx[2], idx[3], idx[4]);
      },
      orderErrors);
  const auto rank = "(rank " + std::to_string(R) + ")";
  report("testSerialize" + rank, orderErrors);

  fill<R>(field, n, 1); // so that a deserialize that does nothing fails
  field.deserialize(serial);
  report("testDeserialize" + rank, mismatches<R>(field, n));
}

// Field::serialize supports only data types without static extents, so only
// Kokkos controllers with runtime extents are tested
template <typename T> void testRuntimeRanks(std::string const &typeName) {
  detail::ContextScope scope(typeName);
  using Ctrlr = MeshField::KokkosController<MemorySpace, ExecutionSpace, T *,
                                            T **, T ***, T ****, T *****>;
  Ctrlr c({rank1[0], /**/ rank2[0], Y, /**/ rank3[0], Y, Z,
           /**/ rank4[0], Y, Z, A, /**/ rank5[0], Y, Z, A, B});
  testSerialize<1>(MeshField::makeField<Ctrlr, 0>(c), rank1);
  testSerialize<2>(MeshField::makeField<Ctrlr, 1>(c), rank2);
  testSerialize<3>(MeshField::makeField<Ctrlr, 2>(c), rank3);
  testSerialize<4>(MeshField::makeField<Ctrlr, 3>(c), rank4);
  testSerialize<5>(MeshField::makeField<Ctrlr, 4>(c), rank5);
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  testRuntimeRanks<int>("int");
  testRuntimeRanks<double>("double");
  // the shape of the fields CreateLagrangeField creates
  using LagrangeCtrlr =
      MeshField::KokkosController<MemorySpace, ExecutionSpace, int ***>;
  LagrangeCtrlr lagrange({1, 1, 16});
  testSerialize<3>(MeshField::makeField<LagrangeCtrlr, 0>(lagrange),
                  {1, 1, 16});

  // Field::serialize and deserialize are currently broken for data types
  // with fixed (static) extents: Field::base_type removes only the pointers,
  // so e.g. int*[3] keeps its [3] and these do not compile.  That includes
  // every Cabana field above rank 1.
  // {
  //   detail::ContextScope scope("Kokkos fixed extents");
  //   using KokkosInt = KokkosRanks<int>;
  //   auto fixed = createRanks<KokkosInt>();
  //   testSerialize<1>(MeshField::makeField<KokkosInt, 0>(fixed), rank1);
  //   testSerialize<2>(MeshField::makeField<KokkosInt, 1>(fixed), rank2);
  //   testSerialize<3>(MeshField::makeField<KokkosInt, 2>(fixed), rank3);
  //   testSerialize<4>(MeshField::makeField<KokkosInt, 3>(fixed), rank4);
  //   testSerialize<5>(MeshField::makeField<KokkosInt, 4>(fixed), rank5);
  // }
  // #ifdef MESHFIELDS_ENABLE_CABANA
  // {
  //   detail::ContextScope scope("Cabana");
  //   using CabanaInt = CabanaRanks<int>;
  //   auto cabana = createRanks<CabanaInt>();
  //   testSerialize<1>(MeshField::makeField<CabanaInt, 0>(cabana), rank1);
  //   testSerialize<2>(MeshField::makeField<CabanaInt, 1>(cabana), rank2);
  //   testSerialize<3>(MeshField::makeField<CabanaInt, 2>(cabana), rank3);
  //   testSerialize<4>(MeshField::makeField<CabanaInt, 3>(cabana), rank4);
  // }
  // #endif
  return t.exitCode();
}
