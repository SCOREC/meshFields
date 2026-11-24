#include "MeshField_Config.hpp"
#include "MeshField_Fail.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

TEST_CASE("fail with variadic formatting", "[exceptions][variadic]") {
#ifdef MeshFields_USE_EXCEPTIONS
  CHECK_THROWS_AS(MeshField::fail("%s %d\n", "variadic", 42),
                  MeshField::exception);
  CHECK_THROWS_WITH(MeshField::fail("%s %d\n", "variadic", 42),
                    "variadic 42\n");
#else
  MeshField::fail("%s %d\n", "variadic", 42);
#endif
}

TEST_CASE("fail with a string", "[exceptions][string]") {
#ifdef MeshFields_USE_EXCEPTIONS
  CHECK_THROWS_AS(MeshField::fail(std::string("string\n")),
                  MeshField::exception);
  CHECK_THROWS_WITH(MeshField::fail(std::string("string\n")), "string\n");
#else
  MeshField::fail(std::string("string\n"));
#endif
}
