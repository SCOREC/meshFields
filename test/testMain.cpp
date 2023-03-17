#include "MeshField.hpp"
#include "CabanaController.hpp"
#include "KokkosController.hpp"
#include "MeshField_Macros.hpp"
#include "MeshField_Utility.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <vector>
#include <iostream>
#include <initializer_list>

#define TOLERANCE 1e-10;

// helper testing functions

// compare doubles within a tolerance
KOKKOS_INLINE_FUNCTION
bool doubleCompare(double d1, double d2) {
  double diff = fabs(d1 - d2);
  return diff < TOLERANCE;
}

using ExecutionSpace = Kokkos::Cuda;
using MemorySpace = Kokkos::CudaUVMSpace;

/*
void single_type(int num_tuples) {
  using Controller =
      SliceWrapper::CabanaController<ExecutionSpace, MemorySpace, double>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double d0 = 10;
    field0(s, a) = d0;
    assert(doubleCompare(field0(s, a), d0));
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "single_type_pfor");
}

void multi_type(int num_tuples) {
  using Controller =
      SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace, double,
                                       double, float, int, char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();
  auto field4 = cabMeshField.makeField<4>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double d0 = 10.456;
    field0(s, a) = d0;
    double d1 = 43.973234567;
    field1(s, a) = d1;
    float f0 = 123.45;
    field2(s, a) = f0;
    int i0 = 22;
    field3(s, a) = i0;
    char c0 = 'a';
    field4(s, a) = c0;

    assert(doubleCompare(field0(s, a), d0));
    assert(doubleCompare(field1(s, a), d1));
    assert(doubleCompare(field2(s, a), f0));
    assert(field3(s, a) == i0);
    assert(field4(s, a) == c0);
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "multi_type_pfor");
}

void many_type(int num_tuples) {

  using Controller =
      SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace, double,
                                       double, float, float, int, short int,
                                       char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();
  auto field4 = cabMeshField.makeField<4>();
  auto field5 = cabMeshField.makeField<5>();
  auto field6 = cabMeshField.makeField<6>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    double d0 = 10.456;
    field0(s, a) = d0;
    double d1 = 43.973234567;
    field1(s, a) = d1;
    float f0 = 123.45;
    field2(s, a) = f0;
    float f1 = 543.21;
    field3(s, a) = f1;
    int i0 = 222;
    field4(s, a) = i0;
    short int i1 = 50;
    field5(s, a) = i1;
    char c0 = 'h';
    field6(s, a) = c0;

    assert(doubleCompare(field0(s, a), d0));
    assert(doubleCompare(field1(s, a), d1));
    assert(doubleCompare(field2(s, a), f0));
    assert(doubleCompare(field3(s, a), f1));
    assert(field4(s, a) == i0);
    assert(field5(s, a) == i1);
    assert(field6(s, a) == c0);
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "many_type_pfor");
}

void rank1_arr(int num_tuples) {
  const int width = 3;
  using Controller =
      SliceWrapper::CabPackedController<ExecutionSpace, MemorySpace,
                                       double[width]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a) {
    for (int i = 0; i < width; i++) {
      double d0 = 10 + i;
      field0(s, a, i) = d0;
      assert(doubleCompare(field0(s, a, i), d0));
    }
  };

  cabMeshField.parallel_for(0, num_tuples, vector_kernel, "rank1_arr_pfor");
}
*/
/*
void testMakeSliceCabana( int num_tuples ) {

  using Ctrl = Controller::CabanaController<ExecutionSpace,MemorySpace,double>;
  Ctrl c( num_tuples );
  MeshField::MeshField<Ctrl> cabanaMeshField(c);

  auto field0 = cabanaMeshField.makeField<0>();

  auto testKernel = KOKKOS_LAMBDA( const int x ) {
    double gamma = (double)x;
    field0(x) = gamma;
    assert(doubleCompare(field0(x),gamma));
  };
  Kokkos::parallel_for("testMakeSliceCabana()", num_tuples, testKernel);

}

void testMakeSliceKokkos( int num_tuples ) {
  using Ctrlr = Controller::KokkosController<MemorySpace,double*>;
  std::vector<int> theta;
  theta.push_back(num_tuples);
  Ctrlr c(theta);
  MeshField::MeshField<Ctrlr> kokkosMeshField(c);

  auto field0 = kokkosMeshField.makeField<0>();
  
  auto testKernel = KOKKOS_LAMBDA( const int x ) {
    double gamma = (double)x;
    field0(x) = gamma;
    assert(doubleCompare(field0(x),gamma));
  };

  Kokkos::parallel_for("testMakeSliceKokkos()", num_tuples, testKernel);
  
}

void testKokkosConstructor( int num_tuples ) {
  
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,double**[3]>;
    std::vector<int> theta;
    theta.push_back(num_tuples);
    theta.push_back(num_tuples);
    Ctrlr c(theta);
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,double[3]>;
    Ctrlr c;
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,int*****>;
    std::vector<int> theta;
    theta.push_back(10);
    theta.push_back(10);
    theta.push_back(10);
    theta.push_back(10);
    theta.push_back(10);
    Ctrlr c(theta);
    MeshField::MeshField<Ctrlr> kok(c);
  }
  {
    using Ctrlr = Controller::KokkosController<MemorySpace,int>;
    Ctrlr c;
    MeshField::MeshField<Ctrlr> kok(c);
  }
}

*/
/*
void testingStufffs() {
  using Ctrlr = Controller::KokkosController<MemorySpace, int*>;
  std::vector<int> dims;
  dims.push_back(10);
  Ctrlr c(dims);
  MeshField::MeshField<Ctrlr> kok(c);
  
  auto field0 = kok.makeField<0>();

  auto vectorKernel = KOKKOS_LAMBDA(const int &s) {
    field0(s) = 3;
  };
  Kokkos::parallel_for("tag", 10, vectorKernel );
}
*/

void testKokkosParallelFor() {
  using Ctrlr = Controller::KokkosController<MemorySpace, int**>;
  std::vector<int> dims;
  dims.push_back(10);
  dims.push_back(10);
  Ctrlr c(dims);
  MeshField::MeshField<Ctrlr> kok(c);
  
  auto field0 = kok.makeField<0>();

  auto vectorKernel = KOKKOS_LAMBDA (const int s, const int a) {
    field0(s,a) = s+a;
    assert(field0(s,a) == s+a);
  };

  kok.parallel_for({0,0},{10,10},vectorKernel, "testKokkosParallelFor()");
}

void kokkosDocumentationLiesTest() { // They dont...
  Kokkos::Array<int64_t,3> start = {0,0,0};
  Kokkos::Array<int64_t,3> end = {3,3,3};
  Kokkos::parallel_for("0_0", Kokkos::MDRangePolicy< Kokkos::Rank<3> >(start,end),
    KOKKOS_LAMBDA (const int c, const int f, const int p) {
      printf("Kokkos documentation lies!!!: c:%d f:%d p:%d\n",c,f,p);
    });
}

int main(int argc, char *argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  
  testKokkosParallelFor();
  //kokkosDocumentationLiesTest();

  /*
  testingStufffs();
  testKokkosConstructor(num_tuples);
  testMakeSliceCabana(num_tuples);
  testMakeSliceKokkos(num_tuples);
  */

  //single_type(num_tuples);
  //multi_type(num_tuples);
  //many_type(num_tuples);
  //rank1_arr(num_tuples);

  return 0;
}
