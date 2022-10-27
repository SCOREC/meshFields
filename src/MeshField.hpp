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
  
};

}
#endif
