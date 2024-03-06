#ifndef cabanaslicewrapper_hpp
#define cabanaslicewrapper_hpp

#include <initializer_list>
#include <type_traits>

#include <Cabana_Core.hpp>
#include <MeshField_Utility.hpp>

namespace Controller {

template <class SliceType, class T> struct CabanaSliceWrapper {

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

using namespace Cabana;

template <class ExecutionSpace, class MemorySpace, class... Ts>
class CabanaController {

  // type definitions
  using TypeTuple = std::tuple<Ts...>;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using DataTypes = Cabana::MemberTypes<Ts...>;

public:
  typedef ExecutionSpace exe;
  // Including num_tuples -> so 3 additional extents
  static const int MAX_RANK = 4;
  static constexpr int vecLen =
      Cabana::AoSoA<DataTypes, MemorySpace>::vector_length;

private:
  // all the type defenitions that are needed us to get the type of the slice
  // returned by the underlying AoSoA
  using soa_t = SoA<DataTypes, vecLen>;

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
    extent_sizes[theta][0] = num_tuples;
    this->theta += 1;
    if constexpr (sizeof...(Tx) != 0)
      construct_sizes<Tx...>();
  }

  // member vaiables
  Cabana::AoSoA<DataTypes, MemorySpace, vecLen> aosoa;
  const int num_tuples;
  unsigned short theta = 0;
  int extent_sizes[sizeof...(Ts)][MAX_RANK];

public:
  CabanaController() : num_tuples(0) {
    static_assert(sizeof...(Ts) != 0);
    construct_sizes<Ts...>();
  }

  CabanaController(int n) : aosoa("sliceAoSoA", n), num_tuples(n) {
    static_assert(sizeof...(Ts) != 0);
    construct_sizes<Ts...>();
  }

  int size(int i, int j) const {
    // returns slice dimensions.
    assert(j >= 0);
    assert(j <= MAX_RANK);
    if (j == 0)
      return this->tuples();
    else
      return extent_sizes[i][j];
  }

  int tuples() const { return num_tuples; }

  template <std::size_t index> auto makeSlice() {
    // Creates wrapper object w/ meta data about the slice.
    using type = std::tuple_element_t<index, TypeTuple>;
    const int stride = sizeof(soa_t) / sizeof(member_value_t<index>);
    auto slice = Cabana::slice<index>(aosoa);
    int sizes[MAX_RANK];
    for (int i = 0; i < MAX_RANK; i++)
      sizes[i] = this->size(index, i);
    return wrapper_slice_t<type, stride>(std::move(slice), sizes);
  }

  template <class Fn>
  typename std::enable_if<1 == MeshFieldUtil::function_traits<Fn>::arity>::type
  simd_for(Fn &kernel, const Kokkos::Array<int64_t, 1> &start,
           const Kokkos::Array<int64_t, 1> &end, std::string tag) {
    Cabana::SimdPolicy<vecLen> policy(start[0], end[0]);
    Cabana::simd_parallel_for(
        policy,
        KOKKOS_LAMBDA(const int &s, const int &a) {
          const std::size_t i = Cabana::Impl::Index<vecLen>::i(s, a);
          kernel(i);
        },
        tag);
  }

  template <class Fn>
  typename std::enable_if<2 == MeshFieldUtil::function_traits<Fn>::arity>::type
  simd_for(Fn &kernel, const Kokkos::Array<int64_t, 2> &start,
           const Kokkos::Array<int64_t, 2> &end, std::string tag) {
    Cabana::SimdPolicy<vecLen> policy(start[0], end[0]);
    const int64_t s1 = start[1];
    const int64_t e1 = end[1];
    Cabana::simd_parallel_for(
        policy,
        KOKKOS_LAMBDA(const int &s, const int &a) {
          const std::size_t i = Cabana::Impl::Index<vecLen>::i(s, a);
          for (int j = s1; j < e1; j++)
            kernel(i, j);
        },
        tag);
  }

  template <class Fn>
  typename std::enable_if<3 == MeshFieldUtil::function_traits<Fn>::arity>::type
  simd_for(Fn &kernel, const Kokkos::Array<int64_t, 3> &start,
           const Kokkos::Array<int64_t, 3> &end, std::string tag) {
    Cabana::SimdPolicy<vecLen> policy(start[0], end[0]);
    const int s1 = start[1], s2 = start[2];
    const int e1 = end[1], e2 = end[2];
    Cabana::simd_parallel_for(
        policy,
        KOKKOS_LAMBDA(const int &s, const int &a) {
          const std::size_t i = Cabana::Impl::Index<vecLen>::i(s, a);
          for (int j = s1; j < e1; j++) {
            for (int k = s2; k < e2; k++) {
              kernel(i, j, k);
            }
          }
        },
        tag);
  }

  template <class Fn>
  typename std::enable_if<4 == MeshFieldUtil::function_traits<Fn>::arity>::type
  simd_for(Fn &kernel, const Kokkos::Array<int64_t, 4> &start,
           const Kokkos::Array<int64_t, 4> &end, std::string tag) {
    Cabana::SimdPolicy<vecLen> policy(start[0], end[0]);
    const int s1 = start[1], s2 = start[2], s3 = start[3];
    const int e1 = end[1], e2 = end[2], e3 = end[3];
    Cabana::simd_parallel_for(
        policy,
        KOKKOS_LAMBDA(const int &s, const int &a) {
          const std::size_t i = Cabana::Impl::Index<vecLen>::i(s, a);
          for (int j = s1; j < e1; j++) {
            for (int k = s2; k < e2; k++) {
              for (int l = s3; l < e3; l++) {
                kernel(i, j, k, l);
              }
            }
          }
        },
        tag);
  }

  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS> &start_init,
                    const std::initializer_list<IE> &end_init,
                    FunctorType &vectorKernel, std::string tag) {
    static_assert(std::is_integral<IS>::value, "Integral required\n");
    static_assert(std::is_integral<IE>::value, "Integral required\n");
    constexpr std::size_t RANK =
        MeshFieldUtil::function_traits<FunctorType>::arity;
    assert(start_init.size() >= RANK);
    assert(end_init.size() >= RANK);
    Kokkos::Array<int64_t, RANK> a_start =
        MeshFieldUtil::to_kokkos_array<RANK>(start_init);
    Kokkos::Array<int64_t, RANK> a_end =
        MeshFieldUtil::to_kokkos_array<RANK>(end_init);
    simd_for<FunctorType>(vectorKernel, a_start, a_end, tag);
  }
};
} // namespace Controller

#endif
