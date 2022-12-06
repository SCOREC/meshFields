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
  typedef typename Slice::Type Type;
  
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

  template <class Field, class View>
  void setField(Field& field, View& view) {
    auto indexToSA = sliceController.indexToSA;
    Kokkos::parallel_for("FillFieldFromViewLoop", sliceController.size(),
    KOKKOS_LAMBDA (const int& i)
    {
      int s;
      int a;
      indexToSA(i,s,a);
      field(s,a) = view(i);
    });
  }
  
  template<class FieldType, class T = typename FieldType::Type>
   T sum(FieldType& field) {
    T result;
    auto indexToSA = sliceController.indexToSA;
    auto reduction_kernel = KOKKOS_LAMBDA (const int& i, T& lsum )
    {
      int s;
      int a;
      indexToSA(i,s,a);
      lsum += field(s, a);
    };
    sliceController.parallel_reduce(reduction_kernel, result, "sum_reduce");
    return result;
  }
  
  template<class FieldType, class T = typename FieldType::Type>
  T min(FieldType& field) {
    T min;
    auto indexToSA = sliceController.indexToSA;
    auto reduce_kernel = KOKKOS_LAMBDA (const int& i, T& lmin )
    {
      int s;
      int a;
      indexToSA(i,s,a);
      lmin = lmin < field(s,a) ? lmin : field(s,a);
    };
    auto reducer = Kokkos::Min<T>(min);
    sliceController.parallel_reduce(reduce_kernel, reducer, "min_reduce");
    return min;
  }

  template<class FieldType, class T = typename FieldType::Type>
  T max(FieldType& field) {
    T max;
    auto indexToSA = sliceController.indexToSA;
    auto reduce_kernel = KOKKOS_LAMBDA (const int& i, T& lmax )
    {
      int s;
      int a;
      indexToSA(i,s,a);
      lmax = lmax > field(s,a) ? lmax : field(s,a);
    };
    auto reducer = Kokkos::Max<T>(max);
    sliceController.parallel_reduce(reduce_kernel, reducer, "max_reduce");
    return max;
  }


  template<class FieldType>
  double mean(FieldType& field) {
    return static_cast<double>(sum(field)) / sliceController.size();
  }
  
  template<typename FunctorType>
  void parallel_for(int lower_bound, int upper_bound,
		    FunctorType& vector_kernel,
		    std::string tag) {
    sliceController.parallel_for(lower_bound, upper_bound, vector_kernel, tag);
  }

  template<typename FunctorType, class ReducerType>
  void parallel_reduce(FunctorType& reduction_kernel, ReducerType& reducer, std::string tag) {
    sliceController.parallel_reduce(reduction_kernel, reducer, tag);
  }

  template <typename FieldType, typename ViewType>
  void parallel_scan(FieldType& field, ViewType& result, std::string tag) {
    
    auto indexToSA = sliceController.indexToSA;
    auto binOp = KOKKOS_LAMBDA(int i, int& partial_sum, bool is_final)
      {
       int s,a;
       indexToSA(i,s,a);
       if (is_final) {
	 result(i) = partial_sum;
	 printf("%d\n", partial_sum);
       }
       partial_sum += field(s,a);
      };
    Kokkos::parallel_scan(tag, sliceController.size()+1, binOp, result);
  }
};
}
#endif
