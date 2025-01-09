#include "MeshField_Fail.hpp"
#include "MeshField_Config.hpp"
#include <cstdarg>  // va_list and friends
#include <cstdio>   // vfprintf
#include <cstdlib>  // abort
#include <iostream> // cerr
#include <string>

namespace MeshField {

void fail(char const *format, ...) {
  va_list vlist;
  va_start(vlist, format);
  char buffer[2048];
  std::vsnprintf(buffer, sizeof(buffer), format, vlist);
  va_end(vlist);
  std::string buf(buffer);
  fail(buf);
}

void fail(std::string msg) {
#ifdef MeshFields_USE_EXCEPTIONS
  throw exception{msg};
#else
  std::cerr << msg;
  std::abort();
#endif
}

} // namespace MeshField
