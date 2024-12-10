#ifndef meshfield_hpp
#define meshfield_hpp

#include <Kokkos_Array.hpp>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <type_traits> // std::same_v<t1,t2>

#include "MeshField_Utility.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace MeshField {

/**
 * @brief
 * Provides access to individual entries of a single Field provided by
 * MeshField::makeField and helper functions that operate on the entire Field.
 *
 * @details
 * Each Field stores DOFs associated with exactly one topological mesh entity
 * type.  For example, a quadratic field over triangles will have one Field for
 * DOFs at the vertices and another for the DOFs at edges.
 *
 * @tparam Slice is the underlying storage object provided by the
 * KokkosController or CabanaController
 *
 * @param Slice see Slice template parameter
 */
template <class Slice> class Field {

  Slice slice;
  typedef typename Slice::Type Type;
  typedef typename std::remove_pointer<Type>::type type_rank1;
  typedef typename std::remove_pointer<type_rank1>::type type_rank2;
  typedef typename std::remove_pointer<type_rank2>::type type_rank3;
  typedef typename std::remove_pointer<type_rank3>::type type_rank4;
  typedef typename std::remove_pointer<type_rank4>::type type_rank5;
  typedef type_rank5 base_type;

  // divisor[i] = product(i+1,5,size)
  const Kokkos::Array<size_t, Slice::RANK> divisors;
  auto computeDivisors() {
    Kokkos::Array<size_t, Slice::RANK> div;
    for (int r = 0; r < RANK; r++) {
      div[r] = 1;
      for (int i = r + 1; i < Slice::RANK; ++i) {
        div[r] *= size(i);
      }
    }
    return div;
  }

  KOKKOS_INLINE_FUNCTION
  auto linearIdxToTensorIdx(size_t index) const {
    Kokkos::Array<size_t, RANK> multiIndex;
    for (int i = 0; i < RANK; ++i) {
      multiIndex[i] = index / divisors[i];
      assert(multiIndex[i] < size(i));
      index %= divisors[i];
    }
    return multiIndex;
  }

public:
  static const int MAX_RANK = Slice::MAX_RANK;
  static const int RANK = Slice::RANK;
  using BaseType = base_type;

  Field(Slice s) : slice(s), divisors(computeDivisors()) {}

  KOKKOS_INLINE_FUNCTION
  auto size(int i) const { return slice.size(i); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s) const { return slice(s); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a) const { return slice(s, a); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i) const { return slice(s, a, i); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j) const {
    return slice(s, a, i, j);
  }
  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j, int k) const {
    return slice(s, a, i, j, k);
  }
  KOKKOS_INLINE_FUNCTION
  auto getFlatViewSize() const {
    size_t N = size(0);
    for (size_t i = 1; i < RANK; ++i)
      N *= size(i);
    return N;
  }
  void serialize_impl(Kokkos::View<base_type *> &serial) const {
    Kokkos::parallel_for(
        "field serializer", serial.size(),
        KOKKOS_CLASS_LAMBDA(const int index) {
          constexpr std::size_t rank = RANK;
          auto serial_data = serial;
          auto sIndex = linearIdxToTensorIdx(index);
          if constexpr (rank == 1) {
            serial_data(index) = slice(index);
          } else if constexpr (rank == 2) {
            serial_data(index) = slice(sIndex[0], sIndex[1]);
          } else if constexpr (rank == 3) {
            serial_data(index) = slice(sIndex[0], sIndex[1], sIndex[2]);
          } else if constexpr (rank == 4) {
            serial_data(index) =
                slice(sIndex[0], sIndex[1], sIndex[2], sIndex[3]);
          } else if constexpr (rank == 5) {
            serial_data(index) =
                slice(sIndex[0], sIndex[1], sIndex[2], sIndex[3], sIndex[4]);
          }
        });
  }
  Kokkos::View<base_type *> serialize() const {
    auto N = this->getFlatViewSize();
    Kokkos::View<base_type *> serial("serialized field", N);
    serialize_impl(serial);
    return std::move(serial);
  }
  void serialize(Kokkos::View<base_type *> &serial) const {
    size_t N = getFlatViewSize();
    assert(N == serial.size());
    serialize_impl(serial);
  }

  void deserialize(const Kokkos::View<const base_type *> &serialized) {
    size_t N = getFlatViewSize();
    assert(N == serialized.size());
    Kokkos::parallel_for(
        "field deserializer", N, KOKKOS_CLASS_LAMBDA(const int index) {
          auto serialized_data = serialized;

          constexpr std::size_t rank = RANK;
          auto sIndex = linearIdxToTensorIdx(index);
          if constexpr (rank == 1) {
            slice(index) = serialized_data(index);
          } else if constexpr (rank == 2) {
            slice(sIndex[0], sIndex[1]) = serialized_data(index);
          } else if constexpr (rank == 3) {
            slice(sIndex[0], sIndex[1], sIndex[2]) = serialized_data(index);
          } else if constexpr (rank == 4) {
            slice(sIndex[0], sIndex[1], sIndex[2], sIndex[3]) =
                serialized_data(index);
          } else if constexpr (rank == 5) {
            slice(sIndex[0], sIndex[1], sIndex[2], sIndex[3], sIndex[4]) =
                serialized_data(index);
          }
        });
  }
};

template <class Controller> class MeshField {

  Controller sliceController;

public:
  MeshField(Controller controller) : sliceController(std::move(controller)) {}

  int size(int type_index, int dimension_index) {
    // given a Controller w/ types Tx = <int*[2][3],double[1][2][3],char*>
    // size(i,j) will return dimension size from Tx[i,j];
    // So in the above 'Tx' example, size(0,2) == 3.
    return sliceController.size(type_index, dimension_index);
  }

  template <std::size_t index> auto makeField() {
    auto slice = sliceController.template makeSlice<index>();
    return Field(std::move(slice));
  }

  template <class Field, class View>
  void setFieldRankOne(Field &field, View &view) {
    using EXE_SPACE = typename Controller::exe;
    Kokkos::RangePolicy<EXE_SPACE> p(0, field.size(0));
    Kokkos::parallel_for(
        p, KOKKOS_LAMBDA(const int &i) { field(i) = view(i); });
  }

  template <class Field, class View>
  void setFieldRankTwo(Field &field, View &view) {
    using EXE_SPACE = typename Controller::exe;
    Kokkos::Array a = MeshFieldUtil::to_kokkos_array<Field::RANK>({0, 0});
    Kokkos::Array b = MeshFieldUtil::to_kokkos_array<Field::RANK>(
        {field.size(0), field.size(1)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::RANK>, EXE_SPACE> p(a, b);
    Kokkos::parallel_for(
        p, KOKKOS_LAMBDA(const int &i, const int &j) {
          field(i, j) = view(i, j);
        });
  }

  template <class Field, class View>
  void setFieldRankThree(Field &field, View &view) {
    using EXE_SPACE = typename Controller::exe;
    Kokkos::Array a = MeshFieldUtil::to_kokkos_array<Field::RANK>({0, 0, 0});
    Kokkos::Array b = MeshFieldUtil::to_kokkos_array<Field::RANK>(
        {field.size(0), field.size(1), field.size(2)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::RANK>, EXE_SPACE> p(a, b);
    Kokkos::parallel_for(
        p, KOKKOS_LAMBDA(const int &i, const int &j, const int &k) {
          field(i, j, k) = view(i, j, k);
        });
  }
  template <class Field, class View>
  void setFieldRankFour(Field &field, View &view) {
    using EXE_SPACE = typename Controller::exe;
    Kokkos::Array a = MeshFieldUtil::to_kokkos_array<Field::RANK>({0, 0, 0, 0});
    Kokkos::Array b = MeshFieldUtil::to_kokkos_array<Field::RANK>(
        {field.size(0), field.size(1), field.size(2), field.size(3)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::RANK>, EXE_SPACE> p(a, b);
    Kokkos::parallel_for(
        p,
        KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l) {
          field(i, j, k, l) = view(i, j, k, l);
        });
  }
  template <class Field, class View>
  void setFieldRankFive(Field &field, View &view) {
    using EXE_SPACE = typename Controller::exe;
    Kokkos::Array a =
        MeshFieldUtil::to_kokkos_array<Field::RANK>({0, 0, 0, 0, 0});
    Kokkos::Array b = MeshFieldUtil::to_kokkos_array<Field::RANK>(
        {field.size(0), field.size(1), field.size(2), field.size(3),
         field.size(4)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::RANK>, EXE_SPACE> p(a, b);
    Kokkos::parallel_for(
        p, KOKKOS_LAMBDA(const int &i, const int &j, const int &k, const int &l,
                         const int &m) {
          field(i, j, k, l, m) = view(i, j, k, l, m);
        });
  }

  template <class FieldT, class View> void setField(FieldT &field, View &view) {
    // Accepts a Field object and Kokkos::View
    // -> sets field(i,j,...) = view(i,j,...) for all i,j,...
    //    up to rank 5.
    constexpr std::size_t view_rank = View::rank;
    constexpr std::size_t field_rank = FieldT::RANK;
    static_assert(field_rank <= FieldT::MAX_RANK);
    static_assert(view_rank == field_rank);

    if constexpr (field_rank == 1) {
      setFieldRankOne(field, view);
    } else if constexpr (field_rank == 2) {
      setFieldRankTwo(field, view);
    } else if constexpr (field_rank == 3) {
      setFieldRankThree(field, view);
    } else if constexpr (field_rank == 4) {
      setFieldRankFour(field, view);
    } else if constexpr (field_rank == 5) {
      setFieldRankFive(field, view);
    } else {
      fprintf(stderr, "setField error: Invalid Field Rank\n");
    }
  }

  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS> &start,
                    const std::initializer_list<IE> &end,
                    FunctorType &vectorKernel, std::string tag) {
    sliceController.parallel_for(start, end, vectorKernel, tag);
  }

  template <typename FunctorType, class IS, class IE, class ReducerType>
  void parallel_reduce(std::string tag, const std::initializer_list<IS> &start,
                       const std::initializer_list<IE> &end,
                       FunctorType &reductionKernel, ReducerType &reducer) {
    /* TODO: infinite reducers */
    /* Number of arguements to lambda should be equal to number of ranks +
     * number of reducers
     * -> adjust 'RANK' accordingly */
    constexpr std::size_t reducer_count = 1;
    constexpr auto RANK =
        MeshFieldUtil::function_traits<FunctorType>::arity - reducer_count;

    using EXE_SPACE = typename Controller::exe;
    assert(start.size() == end.size());
    if constexpr (RANK <= 1) {
      Kokkos::RangePolicy<EXE_SPACE> policy((*start.begin()), (*end.begin()));
      Kokkos::parallel_reduce(tag, policy, reductionKernel, reducer);
    } else {
      auto a_start = MeshFieldUtil::to_kokkos_array<RANK>(start);
      auto a_end = MeshFieldUtil::to_kokkos_array<RANK>(end);
      Kokkos::MDRangePolicy<Kokkos::Rank<RANK>, EXE_SPACE> policy(a_start,
                                                                  a_end);
      Kokkos::parallel_reduce(tag, policy, reductionKernel, reducer);
    }
  }

  template <typename KernelType, typename resultant>
  void parallel_scan(std::string tag, int64_t start_index, int64_t end_index,
                     KernelType &scanKernel, resultant &result) {
    static_assert(std::is_pod<resultant>::value);
    Kokkos::RangePolicy<Kokkos::IndexType<int64_t>> p(start_index, end_index);
    Kokkos::parallel_scan(tag, p, scanKernel, result);
  }
  // depending on size of dimensions, take variable number of arguements
  // that give pairs of lower and upper bound for the multi-dim views.
};

} // namespace MeshField

#endif
