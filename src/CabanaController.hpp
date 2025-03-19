#ifndef cabanaslicewrapper_hpp
#define cabanaslicewrapper_hpp

#include <initializer_list>
#include <type_traits>

#include <Cabana_Core.hpp>
#include <MeshField_Utility.hpp>

namespace MeshField {

/**
 * @todo use a class; following
 * https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c2-use-class-if-the-class-has-an-invariant-use-struct-if-the-data-members-can-vary-independently
 */
template <class SliceType, class T> struct CabanaSliceWrapper {

  using ExecutionSpace = typename SliceType::execution_space;
  static const int MAX_RANK = 4;
  static const std::size_t RANK = Kokkos::View<T>::rank + 1;
  int dimensions[MAX_RANK];
  SliceType slice;
  typedef T Type;

  CabanaSliceWrapper(SliceType slice_in, int *sizes) : slice(slice_in) {
    for (int i = 0; i < MAX_RANK; i++) {
      dimensions[i] = sizes[i];
    }
  }
  CabanaSliceWrapper() {}
  // TODO change size to extent
  KOKKOS_INLINE_FUNCTION
  auto size(int i) const {
    assert(i >= 0);
    assert(i <= MAX_RANK);
    return dimensions[i];
  }

  KOKKOS_INLINE_FUNCTION
  auto rank() const { return std::rank<Type>{} + 1; }

  /* 1D access */
  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int i) const {
    auto s = SliceType::index_type::s(i);
    auto a = SliceType::index_type::a(i);
    return slice.access(s, a);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int i, int j) const {
    auto s = SliceType::index_type::s(i);
    auto a = SliceType::index_type::a(i);
    return slice.access(s, a, j);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int i, int j, int k) const {
    auto s = SliceType::index_type::s(i);
    auto a = SliceType::index_type::a(i);
    return slice.access(s, a, j, k);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int i, int j, int k, int l) const {
    auto s = SliceType::index_type::s(i);
    auto a = SliceType::index_type::a(i);
    return slice.access(s, a, j, k, l);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int i, int j, int k, int l, int m) const {
    auto s = SliceType::index_type::s(i);
    auto a = SliceType::index_type::a(i);
    return slice.access(s, a, j, k, l, m);
  }
};

template <class ExeSpace, class MemorySpace, class... Ts>
class CabanaController {

  // type definitions
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExeSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;

public:
  using ExecutionSpace = ExeSpace;
  // Including num_tuples -> so 3 additional extents
  static const int MAX_RANK = 4;
  static constexpr int vecLen =
      Cabana::AoSoA<DataTypes, MemorySpace>::vector_length;

private:
  // all the type defenitions that are needed us to get the type of the slice
  // returned by the underlying AoSoA
  template <std::size_t index>
  using member_data_t =
      typename Cabana::MemberTypeAtIndex<index, DataTypes>::type;

  template <std::size_t index>
  using member_value_t =
      typename std::remove_all_extents<member_data_t<index>>::type;

  template <class T, int stride>
  using member_slice_t =
      Cabana::Slice<T, MemorySpace, Cabana::DefaultAccessMemory, vecLen,
                    stride>;

  template <class T, int stride>
  using wrapper_slice_t = CabanaSliceWrapper<member_slice_t<T, stride>, T>;

  template <size_t... I>
  static auto construct(std::integer_sequence<size_t, I...>,
                        std::vector<int> &obj) {
    return std::tuple<
        Cabana::AoSoA<Cabana::MemberTypes<Ts>, MemorySpace, vecLen>...>{
        Cabana::AoSoA<Cabana::MemberTypes<Ts>, MemorySpace, vecLen>(
            "sliceAoSoA", obj[I])...};
  }

  template <typename T1, typename... Tx> void construct_sizes() {
    // There are no dynamic ranks w/ cabana controller.
    // So we only need to know extent.
    // Cabana SoA only supports up to 3 additional ranks...
    std::size_t rank = std::rank<T1>{};
    assert(rank < MAX_RANK);
    switch (rank) {
    case 3:
      extent_sizes[theta][3] = (int)std::extent<T1, 2>::value;
    case 2:
      extent_sizes[theta][2] = (int)std::extent<T1, 1>::value;
    case 1:
      extent_sizes[theta][1] = (int)std::extent<T1, 0>::value;
    default:
      for (int i = MAX_RANK - 1; i > rank; i--) {
        // Since extent_sizes 2nd dimension
        // is explicitly MAX_RANK, we fill the
        // remaining space w/ 0
        extent_sizes[theta][i] = 0;
      }
      break;
    }
    extent_sizes[theta][0] = num_tuples[this->theta];
    this->theta += 1;
    if constexpr (sizeof...(Tx) != 0)
      construct_sizes<Tx...>();
  }

  // member vaiables
  std::tuple<Cabana::AoSoA<Cabana::MemberTypes<Ts>, MemorySpace, vecLen>...>
      aosoa;
  // Cabana::AoSoA<DataTypes, MemorySpace, vecLen> aosoa;
  int num_tuples[sizeof...(Ts)];
  unsigned short theta = 0;
  int extent_sizes[sizeof...(Ts)][MAX_RANK];

public:
  CabanaController() {
    static_assert(sizeof...(Ts) != 0);
    for (int i = 0; i < sizeof...(Ts); ++i) {
      num_tuples[i] = 0;
    }
    construct_sizes<Ts...>();
  }
  CabanaController(const std::initializer_list<int> items) {
    static_assert(sizeof...(Ts) != 0);
    std::vector<int> obj(items);
    for (std::size_t i = 0; i < obj.size(); ++i) {
      num_tuples[i] = obj[i];
    }
    aosoa = construct(std::make_integer_sequence<size_t, sizeof...(Ts)>{}, obj);
    construct_sizes<Ts...>();
  }

  int size(int i, int j) const {
    // returns slice dimensions.
    assert(j >= 0);
    assert(j <= MAX_RANK);
    if (j == 0)
      return this->tuples(i);
    else
      return extent_sizes[i][j];
  }

  int tuples(int i) const { return num_tuples[i]; }

  template <std::size_t index> auto makeSlice() {
    // Creates wrapper object w/ meta data about the slice.
    using type = std::tuple_element_t<index, TypeTuple>;
    const int stride = sizeof(Cabana::SoA<Cabana::MemberTypes<type>, vecLen>) /
                       sizeof(member_value_t<index>);
    auto slice = Cabana::slice<0>(std::get<index>(aosoa));
    int sizes[MAX_RANK];
    for (int i = 0; i < MAX_RANK; i++)
      sizes[i] = this->size(index, i);
    return wrapper_slice_t<type, stride>(std::move(slice), sizes);
  }
};
} // namespace MeshField

#endif
