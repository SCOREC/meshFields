#ifndef meshfield_hpp
#define meshfield_hpp

#include "SliceWrapper.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

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
    int num_tuples = sliceController.size();
    int vec_len = sliceController.vecLen;
    double result;
    Kokkos::parallel_reduce("Loop1", num_tuples, KOKKOS_LAMBDA (const int i, double& lsum )
      {
	int numSoa = num_tuples / vec_len;
	const int s = i / vec_len;
	const int a = i % numSoa;
	lsum += field(s, a);
      }, result);
    return result;
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
