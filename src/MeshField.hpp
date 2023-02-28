#ifndef meshfield_hpp
#define meshfield_hpp

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cstdio>

//#include "CabanaSliceWrapper.hpp"

namespace MeshField {

template <class Slice> class Field {

  Slice slice;

public:
  typedef typename Slice::Type Type;

  Field(Slice s) : slice(s) {}

  /* access functions

     The access functions are used to get a specific element from a Field.
     If the user creates a field of an array, the third argument will be the
     index of the array. If the user craetes a field of a 2d array, the third
     and fourth argument will indexes the two arrays, and so on
  */

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a) const { return slice.access(s, a); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i) const { return slice.access(s, a, i); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j) const {
    return slice.access(s, a, i, j);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int s, int a, int i, int j, int k) const {
    return slice.access(s, a, i, j, k);
  }
};

template <class Controller> class MeshField {

  Controller sliceController;

public:
  // constructor

  MeshField(Controller controller) : sliceController(std::move(controller)) {}

  /* makeField

     makeField makes a field from the underlying controller using the index
     given by the user
  */
  // TODO: Use index for kokkos view(s), given <Ts...> in the controller
  // create based on Ts... data types?
  template <std::size_t index> auto makeField() {
    auto slice = sliceController.template makeSlice<index>();
    return Field(std::move(slice));
  }
  /* 
     setField
     fills a field from a Kokkos::View with a parallel_for
  */

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

  /* pre-defined reductions

     These reductions take as input a field and return a single value
     sum - the sum of all the values in the field
     min - the minimum value in the field
     max - the maximum value in the field
     mean - the average value of the field (will always be a double)
  */

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

  /* Parallel functions

     These functions run the given kernel in parallel on the GPU
     (unless the execution space is serial)
     parallel_for - a for loop that will iterate from lowerBound to upperbound
     parallel_reduce - a way for a user to pass in a reduction kernel and a
     reducer to make their own reductions
     parallel_scan - a scan can be used to
     create a view where each value is the sum of the previous values.
  */

  template <typename FunctorType>
  void parallel_for(int lowerBound, int upperBound, FunctorType &vectorKernel,
                    std::string tag) {
    sliceController.parallel_for(lowerBound, upperBound, vectorKernel, tag);
  }

  template <typename FunctorType, class ReducerType>
  void parallel_reduce(FunctorType &reductionKernel, ReducerType &reducer,
                       std::string tag) {
    sliceController.parallel_reduce(reductionKernel, reducer, tag);
  }

  template <typename KernelType>
  void parallel_scan(KernelType &scanKernel, std::string tag) {
    Kokkos::parallel_scan(tag, sliceController.size() + 1, scanKernel);
  }
};

} // namespace MeshField

#endif
