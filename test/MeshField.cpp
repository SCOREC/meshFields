#include "MeshField.hpp"
#include "SliceWrapper.hpp"

#include <Cabana_Core.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using Controller = CabSliceController<ExecutionSpace, MemorySpace, double>;


int main(int argc, char* argv[]) {

  int num_tuples = (argc < 2) ? (10) : (atoi(argv[1]));
  
  Kokkos::ScopeGuard scope_guard(argc, argv);
  
  // Slice Wrapper Controller
  Controller c = Controller(num_tuples);
  MeshField::MeshField<Controller> cabMeshField(c);

  auto field = cabMeshField.makeField<0>();
  
  
  return 0;
}
