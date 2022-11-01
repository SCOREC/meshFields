#ifndef meshfield_hpp
#define meshfield_hpp

#include "SliceWrapper.hpp"

namespace MeshField {

template <class Slice>
class Field {

  Slice slice;

public:
  Field(Slice& s) {
    slice = s;
  }

  KOKKOS_INLINE_FUNCTION
  auto& access(int s, int a) const {
    return slice.access(s,a);
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
    return Field(slice);
  }

  template<typename FunctorType>
  void parallel_for(int lower_bound, int upper_bound,
		    FunctorType vector_kernel,
		    std::string tag) {
    sliceController.parallel_for(lower_bound, upper_bound, vector_kernel, tag);
  }
  
};

}
#endif
