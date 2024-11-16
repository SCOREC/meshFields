#ifndef MESHFIELDS_FAIL_H
#define MESHFIELDS_FAIL_H

#include <stdexcept> // std::runtime_error

namespace MeshField {

struct exception : public std::runtime_error {
  exception(std::string const &msg_in) : std::runtime_error(msg_in) {}
};

// print a message then throw an execution
void fail(char const *format, ...);
} // namespace MeshField
#endif // MESHFIELDS_FAIL_H
