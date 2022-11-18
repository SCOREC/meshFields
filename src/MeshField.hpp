#ifndef meshfield_hpp
#define meshfield_hpp

#include "SliceWrapper.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cstdio>

namespace MeshField {

template <class Slice>
class Field {

  Slice slice;
public:
  Field(Slice s) : slice(s) {}
  KOKKOS_INLINE_FUNCTION
  auto& operator()(int s, int a) const {
    return slice.access(s,a);
  }

  KOKKOS_INLINE_FUNCTION
  auto& operator()(int s, int a, int i) const {
    return slice.access(s,a,i);
  }

  KOKKOS_INLINE_FUNCTION
  auto& operator()(int s, int a, int i, int j) const {
    return slice.access(s,a,i,j);
  }

  KOKKOS_INLINE_FUNCTION
  auto& operator()(int s, int a, int i, int j, int k) const {
    return slice.access(s,a,i,j,k);
  }
};

template <class Controller>
class MeshField {

  Controller sliceController;  

public:
  MeshField(Controller sc) : sliceController(std::move(sc)) {}

  template <std::size_t index>
  auto makeField() {
    auto slice = sliceController.template makeSlice<index>();
    return Field(std::move(slice));
  }

  template<class FieldType>
  double sum(FieldType& field) {
    double result;
    Controller* sc = &sliceController;
    auto reduction_kernel = KOKKOS_LAMBDA (const int i, double& lsum )
    {
      int s;
      int a;
      sc->indexToSA(i,s,a);
      lsum += field(s, a);
    };
    sliceController.parallel_reduce(reduction_kernel, result, "sum_reduce");
    return result;
  }
  
  template<class FieldType>
  double min(FieldType& field) {
    double min;
    Controller* sc = &sliceController;
    auto reduce_kernel = KOKKOS_LAMBDA (const int& i, double& lmin )
    {
      int s;
      int a;
      sc->indexToSA(i,s,a);
      lmin = lmin < field(s,a) ? lmin : field(s,a);
    };
    auto reducer = Kokkos::Min<double>(min);
    sliceController.parallel_reduce(reduce_kernel, reducer, "min_reduce");
    return min;
  }

  template<class FieldType>
  double max(FieldType& field) {
    double max;
    Controller* sc = &sliceController;
    auto reduce_kernel = KOKKOS_LAMBDA (const int& i, double& lmax )
    {
      int s;
      int a;
      sc->indexToSA(i,s,a);
      lmax = lmax > field(s,a) ? lmax : field(s,a);
    };
    auto reducer = Kokkos::Max<double>(max);
    sliceController.parallel_reduce(reduce_kernel, reducer, "max_reduce");
    return max;
  }


  template<class FieldType>
  double mean(FieldType& field) {
    return sum(field) / sliceController.size();
  }
  
  template<typename FunctorType>
  void parallel_for(int lower_bound, int upper_bound,
		    FunctorType& vector_kernel,
		    std::string tag) {
    sliceController.parallel_for(lower_bound, upper_bound, vector_kernel, tag);
  }
  
};

}
#endif
