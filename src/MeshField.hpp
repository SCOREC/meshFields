#ifndef meshfield_hpp
#define meshfield_hpp

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cstdio>
#include <type_traits> // std::same_v<t1,t2>
#include <array>

#include "MeshField_Utility.hpp"

//#include "CabanaSliceWrapper.hpp"

namespace MeshField {

template <class Slice> class Field {

  Slice slice;
  typedef typename Slice::Type Type;

public:

  Field(Slice s) : slice(s) {}
  
  /* operator() -> 1D Access */
  /* access() -> 2D Access */

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

};

template <class Controller> class MeshField {

  Controller sliceController;

public:

  MeshField(Controller controller) : sliceController(std::move(controller)) {}

  template <std::size_t index> auto makeField() {
    auto slice = sliceController.template makeSlice<index>();
    return Field(std::move(slice));
  }
  /*
  template <class Field, class View> void setField(Field &field, View &view) {
    auto indexToSA = sliceController.indexToSA;
    Kokkos::parallel_for(
        "FillFieldFromViewLoop", sliceController.size(),
        KOKKOS_LAMBDA(const int &i) {
          int s, a;
          indexToSA(i, s, a);
          field(s, a) = view(i);
        });
  }


  template <class FieldType, class T = typename FieldType::Type>
  T sum(FieldType &field) {
    T result;
    auto indexToSA = sliceController.indexToSA;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lsum) {
      int s, a;
      indexToSA(i, s, a);
      lsum += field(s, a);
    };
    sliceController.parallel_reduce(reductionKernel, result, "sum_reduce");
    return result;
  }

  template <class FieldType, class T = typename FieldType::Type>
  T min(FieldType &field) {
    T min;
    auto indexToSA = sliceController.indexToSA;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lmin) {
      int s, a;
      indexToSA(i, s, a);
      lmin = lmin < field(s, a) ? lmin : field(s, a);
    };
    auto reducer = Kokkos::Min<T>(min);
    sliceController.parallel_reduce(reductionKernel, reducer, "min_reduce");
    return min;
  }

  template <class FieldType, class T = typename FieldType::Type>
  T max(FieldType &field) {
    T max;
    auto indexToSA = sliceController.indexToSA;
    auto reductionKernel = KOKKOS_LAMBDA(const int &i, T &lmax) {
      int s, a;
      indexToSA(i, s, a);
      lmax = lmax > field(s, a) ? lmax : field(s, a);
    };
    auto reducer = Kokkos::Max<T>(max);
    sliceController.parallel_reduce(reductionKernel, reducer, "max_reduce");
    return max;
  }

  template <class FieldType> double mean(FieldType &field) {
    return static_cast<double>(sum(field)) / sliceController.size();
  }
  */

  template <typename FunctorType, class IS, class IE>
  void parallel_for(const std::initializer_list<IS>& start, 
                    const std::initializer_list<IE>& end,
                    FunctorType &vectorKernel,
                    std::string tag) {
    constexpr auto RANK = MeshFieldUtil::function_traits<FunctorType>::arity;
    Kokkos::Array<int64_t,RANK> a_start{};
    Kokkos::Array<int64_t,RANK> a_end{};
    auto x = start.begin();
    auto y = end.begin();
    for( std::size_t i = 0; i < RANK; i++ ) 
      { a_start[i] = (*x); a_end[i] = (*y); x++; y++;}
    sliceController.template parallel_for< RANK > (a_start, a_end, vectorKernel, tag);
  }

  /*
  template <typename FunctorType, class ReducerType>
  void parallel_reduce(FunctorType &reductionKernel, ReducerType &reducer,
                       std::string tag) {
    sliceController.parallel_reduce(reductionKernel, reducer, tag);
  }

  template <typename KernelType>
  void parallel_scan(KernelType &scanKernel, std::string tag) {
    Kokkos::parallel_scan(tag, sliceController.size() + 1, scanKernel);
  }
  // depending on size of dimensions, take variable number of arguements
  // that give pairs of lower and upper bound for the multi-dim views.
  */
};

} // namespace MeshField

#endif
