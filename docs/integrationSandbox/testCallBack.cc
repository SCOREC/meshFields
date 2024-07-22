#include <cassert>
#include <iostream>
#include <type_traits>
#include <Kokkos_Core.hpp>

template <typename OuterFunc, typename InnerFunc>
void loop(OuterFunc& outer, InnerFunc& inner) {
    static_assert( std::is_invocable_v<decltype(&OuterFunc::outer), OuterFunc&, int> );
    static_assert( std::is_invocable_v<decltype(&InnerFunc::inner), InnerFunc&, int, int> );
    Kokkos::parallel_for(10, 
      KOKKOS_LAMBDA(const int i) {
        outer.outer(i); //does not compile, error indicates that outer is const, possibly due to pass by value
//        for (int j = 0; j < 5; j++) {
//            inner.inner(i,j);
//        }
        Kokkos::printf("Hello from i = %i\n", i);
    });
}

struct Foo {
    int x;
    void outer(int i) {
       x+=i;
    };
};

struct Bar {
    int x;
    void inner(int i, int j) {
       x+=i+j;
    };
};

struct Baz {
    int x;
    void something(int i, int j) {
       x+=i+j;
    };
};


int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    Foo f{0};
    Bar b{0};

    std::cout << f.x << " " << b.x << "\n";
    loop(f,b);

    //Baz z{0};
    //loop(f,z); //fails static assert, as expected
    Kokkos::finalize();
    return 0;
}
