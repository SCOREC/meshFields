#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "KokkosController.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>


//evaluate a field at the specified local coordinate for each element
void triangleLocalPointEval() {
  const auto numElms = 3; //provided by the mesh
  const int numVerts = 5; //provided by the mesh
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using Ctrlr =
      Controller::KokkosController<MemorySpace, ExecutionSpace, double ***>;
  Ctrlr kk_ctrl({1, 1, numVerts}); //1 dof with 1 component per vtx
  MeshField::MeshField<Ctrlr> kokkosMeshField(kk_ctrl);

  auto field0 = kokkosMeshField.makeField<0>();

  MeshFields::FieldElement f(numElms, MeshFields::LinearTriangleShape(), field0);

  std::array<MeshFields::Real,9> localCoords = {0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5};
  Kokkos::View<MeshFields::Real[9], Kokkos::HostSpace, Kokkos::MemoryUnmanaged> lc_h(localCoords.data(), localCoords.size());
  Kokkos::View<MeshFields::Real[9]> lc("localCoords");
  Kokkos::deep_copy(lc, lc_h);
  auto x = MeshFields::evaluate(f, lc);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  triangleLocalPointEval();
  std::cerr << "done\n";
  Kokkos::finalize();
  return 0;
}


