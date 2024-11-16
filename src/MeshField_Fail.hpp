#ifndef MESHFIELDS_FAIL_H
#define MESHFIELDS_FAIL_H

namespace MeshField {
// print a message then terminate execution
[[noreturn]] void fail(char const *format, ...);
} // namespace MeshField
#endif // MESHFIELDS_FAIL_H
