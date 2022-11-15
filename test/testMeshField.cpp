#include "MeshField.hpp"
#include "SliceWrapper.hpp"

#include <Cabana_Core.hpp>

#define TOLERANCE 1e-10;

KOKKOS_INLINE_FUNCTION
bool doubleCompare(double d1, double d2) {
  double diff = fabs(d1 - d2);
  return diff < TOLERANCE;
}

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

void test_reductions(int num_tuples) {
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace, double>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  
  auto initField = KOKKOS_LAMBDA(const int s, const int a)
  {
   field0(s,a) = 10.0;
  };

  cabMeshField.parallel_for(0,num_tuples, initField, "initField");
  
  double sum = cabMeshField.sum(field0);
  printf("sum: %lf\n", sum);
  double mean = cabMeshField.mean(field0);
  printf("mean: %lf\n", mean);
}

void single_type(int num_tuples) {
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace, double>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   double d0 = 10;
   field0(s,a) = d0;
   assert(doubleCompare(field0(s,a), d0));
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"single_type_pfor");
  
}

void multi_type(int num_tuples) {
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
						   double,double, float, int, char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();
  auto field4 = cabMeshField.makeField<4>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   double d0 = 10.456;
   field0(s,a) = d0;
   double d1 = 43.973234567;
   field1(s,a) = d1;
   float f0 = 123.45;
   field2(s,a) = f0;
   int i0 = 22;
   field3(s,a) = i0;
   char c0 = 'a';
   field4(s,a) = c0;
   
   assert(doubleCompare(field0(s,a), d0));
   assert(doubleCompare(field1(s,a), d1));
   assert(doubleCompare(field2(s,a), f0));
   assert(field3(s,a) == i0);
   assert(field4(s,a) == c0);
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"multi_type_pfor");
}

void many_type(int num_tuples) {
  
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
					double, double, float, float, int,
					short int, char>;

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

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   double d0 = 10.456;
   field0(s,a) = d0;
   double d1 = 43.973234567;
   field1(s,a) = d1;
   float f0 = 123.45;
   field2(s,a) = f0;
   float f1 = 543.21;
   field3(s,a) = f1;
   int i0 = 222;
   field4(s,a) = i0;
   short int i1 = 50;
   field5(s,a) = i1;
   char c0 = 'h';
   field6(s,a) = c0;
   
   assert(doubleCompare(field0(s,a), d0));
   assert(doubleCompare(field1(s,a), d1));
   assert(doubleCompare(field2(s,a), f0));
   assert(doubleCompare(field3(s,a), f1));
   assert(field4(s,a) == i0);
   assert(field5(s,a) == i1);
   assert(field6(s,a) == c0);
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"many_type_pfor");
}

void rank1_arr(int num_tuples) {
  const int width = 3;
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
						   double[width]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  
  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   for (int i = 0; i < width; i++)
   {
    double d0 = 10+i;
    field0(s,a,i) = d0;
    assert(doubleCompare(field0(s,a,i), d0));
   }
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"rank1_arr_pfor");
}

void rank2_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
						   double[width][height]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   for (int i = 0; i < width; i++)
   {
    for (int j = 0; j < height; j++)
    {
     double d0 = (10+i)/(j+1);
     field0(s,a,i,j) = d0;
     assert(doubleCompare(field0(s,a,i,j), d0));
    }
   }
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"rank2_arr_pfor");
}

void rank3_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  const int depth = 2;
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
						   double[width][height][depth]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   for (int i = 0; i < width; i++)
   {
    for (int j = 0; j < height; j++)
    {
     for (int k = 0; k < depth; k++)
     {
      double d0 = ((10+i)*(k+1))/(j+1);
      field0(s,a,i,j,k) = d0;
      assert(doubleCompare(field0(s,a,i,j,k), d0));
     }
    }
   }
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"rank3_arr_pfor");

}

void mix_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  const int depth = 2;
  using Controller = SliceWrapper::CabSliceController<ExecutionSpace, MemorySpace,
						   double[width][height][depth],
						   float[width][height],
						   int[width], char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   float f0;
   int i0;
   char c0 = 's';
   field3(s,a) = c0;
   
   for (int i = 0; i < width; i++)
   {
    i0 = i+s+a;
    field2(s,a,i) = i0;
    for (int j = 0; j < height; j++)
    {
     f0 = i0 / (i+j+1.123);
     field1(s,a,i,j) = f0;
     for (int k = 0; k < depth; k++)
     {
      double d0 = ((10+i)*(k+1))/(j+1);
      field0(s,a,i,j,k) = d0;
      assert(doubleCompare(field0(s,a,i,j,k), d0));
     }
     assert(doubleCompare(field1(s,a,i,j), f0));
    }
    assert(field2(s,a,i) == i0);
   }
   assert(field3(s,a) == c0);
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"mix_arr_pfor");

}

int main(int argc, char* argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  
  single_type(num_tuples);
  multi_type(num_tuples);
  many_type(num_tuples);
  rank1_arr(num_tuples);
  rank2_arr(num_tuples);
  rank3_arr(num_tuples);
  mix_arr(num_tuples);

  test_reductions(num_tuples);
  
  return 0;
}
