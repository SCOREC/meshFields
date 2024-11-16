#include "MeshField_Fail.hpp"
#include <cstdarg> // va_list and friends
#include <cstdio>  // vfprintf
#include <cstdlib> // abort

namespace MeshField {

void fail(char const *format, ...) {
  va_list vlist;
  va_start(vlist, format);
  std::vfprintf(stderr, format, vlist);
  va_end(vlist);
  std::abort();
}

} // namespace MeshField
