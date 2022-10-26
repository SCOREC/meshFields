#ifndef meshfield_hpp
#define meshfield_hpp

#include <Cabana_Core.hpp>

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
  MeshField(Controller& sc) {
    sliceController = sc;
  }

  template <int index>
  auto makeField() {
    auto slice = sliceController.makeSliceCab<index>();
    return Field(slice);
  }
  
};


#endif
