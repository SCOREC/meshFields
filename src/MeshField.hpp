#ifndef meshfield_hpp
#define meshfield_hpp

#include <Kokkos_Array.hpp>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <type_traits> // std::same_v<t1,t2>

#include "MeshField_Fail.hpp"
#include "MeshField_Utility.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace {
template <class Field, class View>
void checkExtents(Field &field, View &view, std::string key) {
  bool matches = true;
  for (int i = 0; i < Field::Rank; i++) {
    matches = matches && (view.extent(i) == field.size(i));
  }
  if (!matches) {
    MeshField::fail("%s: the extents of the view does not match the field\n",
                    key.c_str());
  }
}
} // namespace

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

  using ExecutionSpace = typename Slice::ExecutionSpace;

  const Kokkos::Array<size_t, Slice::RANK> divisors;

  /**
   * @brief
   * precompute divisors needed for linearIdxToTensorIdx
   * @details
   * divisor[i] = product(i+1,5,size)
   */
  auto computeDivisors() {
    Kokkos::Array<size_t, Rank> div;
    for (int r = 0; r < Rank; r++) {
      div[r] = 1;
      for (int i = r + 1; i < Rank; ++i) {
        div[r] *= size(i);
      }
    }
    return div;
  }

  /**
   * @brief
   * given an index in the range of 0 to the total capacity of the Field,
   * return the row-major (i.e., C/C++ layout or Kokkos::LayoutRight)
   * multi-rank index
   * @details
   * relies on computeDivisors
   * @return a Kokkos::Array with the row-major multi-rank index
   */
  KOKKOS_INLINE_FUNCTION
  auto linearIdxToTensorIdx(size_t index) const {
    Kokkos::Array<size_t, Rank> multiIndex;
    for (int i = 0; i < Rank; ++i) {
      multiIndex[i] = index / divisors[i];
      assert(multiIndex[i] < size(i));
      index %= divisors[i];
    }
    return multiIndex;
  }

  /**
   * get the product of the rank extents
   */
  KOKKOS_INLINE_FUNCTION
  auto totalSize() const { return size(0) * divisors[0]; }

public:
  static const int MAX_RANK = Slice::MAX_RANK;
  static const int Rank = Slice::RANK;
  using BaseType = base_type;

  Field(Slice s) : slice(s), divisors(computeDivisors()) {}

  /**
   * @brief
   * get the size/extent of the specified rank
   * @param i (in) the rank to query
   * @return the size/extent
   */
  KOKKOS_INLINE_FUNCTION
  auto size(int i) const { return slice.size(i); }

  /**
   * access the underlying field at the specified index
   */
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

  void serialize_impl(Kokkos::View<base_type *> &serial) const {
    assert(serial.size() == totalSize());
    Kokkos::parallel_for(
        "field serializer", serial.size(),
        KOKKOS_CLASS_LAMBDA(const int index) {
          constexpr std::size_t rank = Rank;
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

  /**
   * @brief
   * copy the Field into a single rank Kokkos View
   * @details
   * See linearIdxToTensorIdx
   * @return the Kokkos View
   */
  Kokkos::View<base_type *> serialize() const {
    auto N = totalSize();
    Kokkos::View<base_type *> serial("serialized field", N);
    serialize_impl(serial);
    return std::move(serial);
  }

  /**
   * @brief
   * copy the Field into a given Kokkos View
   * @details
   * The Kokkos View needs to have extent = the product of the Field extents.
   * See linearIdxToTensorIdx
   * @param serial (in/out) the Kokkow View to copy into
   */
  void serialize(Kokkos::View<base_type *> &serial) const {
    const size_t N = totalSize();
    assert(N == serial.size());
    serialize_impl(serial);
  }

  /**
   * @brief
   * copy the given Kokkos View into the Field
   * @details
   * See linearIdxToTensorIdx
   * @param serial (in/out) the Kokkow View to copy into
   */
  void deserialize(const Kokkos::View<const base_type *> &serialized) {
    const size_t N = totalSize();
    assert(N == serialized.size());
    Kokkos::parallel_for(
        "field deserializer", N, KOKKOS_CLASS_LAMBDA(const int index) {
          auto serialized_data = serialized;

          constexpr std::size_t rank = Rank;
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

  template <class View> void setRankOne(View &view) {
    Kokkos::RangePolicy<ExecutionSpace> p(0, size(0));
    Kokkos::parallel_for(
        p, KOKKOS_CLASS_LAMBDA(const int &i) { operator()(i) = view(i); });
  }
  template <class View> void setRankTwo(View &view) {
    Kokkos::Array a = MeshFieldUtil::to_kokkos_array<Field::Rank>({0, 0});
    Kokkos::Array b =
        MeshFieldUtil::to_kokkos_array<Field::Rank>({size(0), size(1)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::Rank>, ExecutionSpace> p(a, b);
    Kokkos::parallel_for(
        p, KOKKOS_CLASS_LAMBDA(const int &i, const int &j) {
          operator()(i, j) = view(i, j);
        });
  }
  template <class View> void setRankThree(View &view) {
    Kokkos::Array a = MeshFieldUtil::to_kokkos_array<Field::Rank>({0, 0, 0});
    Kokkos::Array b = MeshFieldUtil::to_kokkos_array<Field::Rank>(
        {size(0), size(1), size(2)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::Rank>, ExecutionSpace> p(a, b);
    Kokkos::parallel_for(
        p, KOKKOS_CLASS_LAMBDA(const int &i, const int &j, const int &k) {
          operator()(i, j, k) = view(i, j, k);
        });
  }
  template <class View> void setRankFour(View &view) {
    Kokkos::Array a = MeshFieldUtil::to_kokkos_array<Field::Rank>({0, 0, 0, 0});
    Kokkos::Array b = MeshFieldUtil::to_kokkos_array<Field::Rank>(
        {size(0), size(1), size(2), size(3)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::Rank>, ExecutionSpace> p(a, b);
    Kokkos::parallel_for(
        p, KOKKOS_CLASS_LAMBDA(const int &i, const int &j, const int &k,
                               const int &l) {
          operator()(i, j, k, l) = view(i, j, k, l);
        });
  }
  template <class View> void setRankFive(View &view) {
    Kokkos::Array a =
        MeshFieldUtil::to_kokkos_array<Field::Rank>({0, 0, 0, 0, 0});
    Kokkos::Array b = MeshFieldUtil::to_kokkos_array<Field::Rank>(
        {size(0), size(1), size(2), size(3), size(4)});
    Kokkos::MDRangePolicy<Kokkos::Rank<Field::Rank>, ExecutionSpace> p(a, b);
    Kokkos::parallel_for(
        p, KOKKOS_CLASS_LAMBDA(const int &i, const int &j, const int &k,
                               const int &l, const int &m) {
          operator()(i, j, k, l, m) = view(i, j, k, l, m);
        });
  }
  /**
   * sets field(i,j,...) = view(i,j,...) for all i,j,...  up to rank 5.
   * @todo check datatype of View
   * @tparam View to copy from, must have matching extent and rank
   */
  template <class View> void set(View &view) {
    constexpr std::size_t view_rank = View::rank;
    constexpr std::size_t field_rank = Slice::RANK;
    static_assert(field_rank <= Slice::MAX_RANK);
    static_assert(view_rank == field_rank);
    checkExtents(*this, view, __func__);
    if constexpr (field_rank == 1) {
      setRankOne(view);
    } else if constexpr (field_rank == 2) {
      setRankTwo(view);
    } else if constexpr (field_rank == 3) {
      setRankThree(view);
    } else if constexpr (field_rank == 4) {
      setRankFour(view);
    } else if constexpr (field_rank == 5) {
      setRankFive(view);
    } else {
      fail("Field::set error: field rank is %d, it must be [1:5]\n",
           field_rank);
    }
  }
};

/**
 * @brief
 * Create a Field from a Controller Slice
 *
 * @tparam Controller manages and provides the storage for Fields, current
 * options are KokkosController or CabanaController
 *
 * @param controller see Controller
 */
template <class Controller, std::size_t index> auto makeField(Controller controller) {
  auto slice = controller.template makeSlice<index>();
  return Field(std::move(slice));
}

} // namespace MeshField

#endif
