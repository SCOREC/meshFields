#include "MeshField_Fail.hpp"
#include <iostream>

void viaVariadic() {
  try {
    MeshField::fail("%s\n", "variadic");
  } catch (const std::exception &ex) {
    std::cerr << ex.what();
  }
}

void viaString() {
  try {
    std::string msg = "string\n";
    MeshField::fail(msg);
  } catch (const std::exception &ex) {
    std::cerr << ex.what();
  }
}

int main(int argc, char **argv) {
  viaVariadic();
  viaString();
  return 0;
}
