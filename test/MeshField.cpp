#include "MeshField.hpp"
#include "SliceWrapper.hpp"

#include <Cabana_Core.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;


void single_type(int num_tuples) {
  using Controller = CabSliceController<ExecutionSpace, MemorySpace, double>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field = cabMeshField.makeField<0>();

  auto vector_kernel = KOKKOS_LAMBDA(const int s, const int a)
  {
   field.access(s,a);
   printf("test %d %d\n", s,a);
  };
  
  cabMeshField.parallel_for(0,num_tuples,vector_kernel,"single_type_pfor");
  
}

void multi_type(int num_tuples) {
  using Controller = CabSliceController<ExecutionSpace, MemorySpace, double,
					double, float, int, char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();
  auto field4 = cabMeshField.makeField<4>();
 
}

void many_type(int num_tuples) {
  
  using Controller = CabSliceController<ExecutionSpace, MemorySpace,
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
}

void rank1_arr(int num_tuples) {
  const int width = 3;
  using Controller = CabSliceController<ExecutionSpace, MemorySpace, double[width]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
}

void rank2_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  using Controller = CabSliceController<ExecutionSpace, MemorySpace,
					double[width][height]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
}

void rank3_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  const int depth = 2;
  using Controller = CabSliceController<ExecutionSpace, MemorySpace,
					double[width][height][depth]>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
}

void mix_arr(int num_tuples) {
  const int width = 3;
  const int height = 4;
  const int depth = 2;
  using Controller = CabSliceController<ExecutionSpace, MemorySpace,
					double[width][height][depth],
				        float[width][height], int[width],
					char>;

  // Slice Wrapper Controller
  Controller c(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field0 = cabMeshField.makeField<0>();
  auto field1 = cabMeshField.makeField<1>();
  auto field2 = cabMeshField.makeField<2>();
  auto field3 = cabMeshField.makeField<3>();
}

int main(int argc, char* argv[]) {
  int num_tuples = (argc < 2) ? (1000) : (atoi(argv[1]));
  Kokkos::ScopeGuard scope_guard(argc, argv);
  printf("num_tuples: %d\n", num_tuples);
  single_type(num_tuples);
  //multi_type(num_tuples);
  //many_type(num_tuples);
  //rank1_arr(num_tuples);
  //rank2_arr(num_tuples);
  //rank3_arr(num_tuples);
  //mix_arr(num_tuples);
  
  return 0;
}
