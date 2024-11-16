#include "MeshField_Fail.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    MeshField::fail("ERROR: failed\n");
  } catch (const std::exception &ex) {
    std::cerr << ex.what();
  }
  return 0;
}
