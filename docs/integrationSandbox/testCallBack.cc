#include <cassert>
#include <iostream>
#include <type_traits>
#include <Kokkos_Core.hpp>

template <typename Functor>
void loop(Functor a) {
    Kokkos::parallel_for(10, 
      KOKKOS_LAMBDA(const int i) {
        auto val = a(i);
        Kokkos::printf("Hello from i = %i\n", val);
    });
}

struct Foo {
    Kokkos::View<int*> x;
    Foo() {
       x = Kokkos::View<int*>("d", 10); 
    };
    KOKKOS_INLINE_FUNCTION int operator()(int i) const {
      x(i) += i+10;
      return x(i);
    };
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Foo f;
    loop(f);
  }
  Kokkos::finalize();
  return 0;
}
