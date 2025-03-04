#ifndef MESHFIELDS_FAIL_H
#define MESHFIELDS_FAIL_H

#include <stdexcept>   // std::runtime_error
#include <string_view> // std::string_view

namespace MeshField {

struct exception : public std::runtime_error {
  exception(std::string const &msg_in) : std::runtime_error(msg_in) {}
};

/** print a message then throw an execution or abort
 * \brief an exception is thrown if MeshFields_USE_EXCEPTIONS is true,
 * otherwise abort
 * \param format (in) printf like format string
 * \param ... (in) variadic list of args to match the format specifiers used in
 * the format string
 */
void fail(char const *format, ...);

/** see fail(char const* format, ...)
 * \param msg (in) string to print
 */
void fail(std::string msg);
} // namespace MeshField
#endif // MESHFIELDS_FAIL_H
