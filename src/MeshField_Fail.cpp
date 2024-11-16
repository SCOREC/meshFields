#include "MeshField_Fail.hpp"
#include "MeshField_Config.hpp"
#include <cstdarg> // va_list and friends
#include <cstdio>  // vfprintf
#include <cstdlib> // abort
#include <string>

namespace MeshField {

void fail(char const *format, ...) {
  va_list vlist;
  va_start(vlist, format);
#ifdef MeshFields_USE_EXCEPTIONS
  char buffer[2048];
  std::vsnprintf(buffer, sizeof(buffer), format, vlist);
  va_end(vlist);
  std::string buf(buffer);
  throw exception{buf};
#else
  std::vfprintf(stderr, format, vlist);
  va_end(vlist);
  std::abort();
#endif
}

} // namespace MeshField
