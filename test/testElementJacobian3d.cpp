#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_For.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearTetrahedronToVertexField {
  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Tetrahedron};
  }
  KOKKOS_FUNCTION MeshField::LO operator()(MeshField::LO tetNodeIdx) const {
    return tetNodeIdx;
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO tetNodeIdx, MeshField::LO tetCompIdx,
             MeshField::LO tet, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Tetrahedron);
    MeshField::LO tetNode2Vtx[1][4] = {{0, 1, 2, 3}};
    const MeshField::LO vtx = tetNode2Vtx[tet][tetNodeIdx];
    return {0, tetCompIdx, vtx, MeshField::Vertex};
  }
};

class TetrahedronTestCase {
public:
  TetrahedronTestCase(std::vector<MeshField::Real> coords_in,
                      std::vector<MeshField::Real> jacobian_in,
                      MeshField::Real jacobianDeterminant_in)
      : coords(coords_in), jacobian(jacobian_in),
        jacobianDeterminant(jacobianDeterminant_in) {
    assert(coords.size() == 12);
    assert(jacobian.size() == 9);
  };
  const std::vector<MeshField::Real> coords;
  const std::vector<MeshField::Real> jacobian;
  const MeshField::Real jacobianDeterminant;

private:
  TetrahedronTestCase();
};

template <typename ShapeField>
void setVtxCoords(size_t numVerts, size_t meshDim, TetrahedronTestCase testTri,
                  ShapeField field) {
  Kokkos::View<MeshField::Real *, Kokkos::HostSpace> coords_h(
      "coords_h", numVerts * meshDim);
  for (size_t i = 0; i < numVerts * meshDim; i++)
    coords_h[i] = testTri.coords.at(i);
  auto coords = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), coords_h);
  auto setCoordField = KOKKOS_LAMBDA(const int &vtx) {
    field(vtx, 0, 0, MeshField::Vertex) = coords(vtx * meshDim);
    field(vtx, 0, 1, MeshField::Vertex) = coords(vtx * meshDim + 1);
    field(vtx, 0, 2, MeshField::Vertex) = coords(vtx * meshDim + 2);
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {numVerts}, setCoordField,
                          "setCoordField");
}

void tetJacobian() {
  const MeshField::MeshInfo meshInfo{
      .numVtx = 4, .numEdge = 6, .numTet = 1, .dim = 3};
  auto coordField = MeshField::CreateCoordinateField<
      ExecutionSpace, MeshField::KokkosController, 3>(meshInfo);
  TetrahedronTestCase identityTet({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
                                  {1, 0, 0, 0, 1, 0, 0, 0, 1}, 1);
  TetrahedronTestCase skewedTet({0, 0, 0, 5, 1, 2, 3, 4, 6, 10, 8, 9},
                                {5, 1, 2, 3, 4, 6, 10, 8, 9}, -59);
  for (auto testCase : {identityTet, skewedTet}) {
    setVtxCoords(meshInfo.numVtx, meshInfo.dim, testCase, coordField);

    MeshField::FieldElement f(meshInfo.numTet, coordField,
                              MeshField::LinearTetrahedronShape(),
                              LinearTetrahedronToVertexField());

    Kokkos::View<MeshField::Real *[4]> lc("localCoords", 1);
    Kokkos::deep_copy(lc, 1.0 / 4);
    const auto numPtsPerElement = 1;
    const auto J = MeshField::getJacobians(f, lc, numPtsPerElement);
    const auto J_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
    assert(J_h.rank() == 3);
    assert(J_h.extent(0) == 1);
    assert(J_h.extent(1) == 3);
    assert(J_h.extent(2) == 3);
    std::cout << "tet jacobian\n"
              << J_h(0, 0, 0) << " " << J_h(0, 0, 1) << " " << J_h(0, 0, 2)
              << "\n"
              << J_h(0, 1, 0) << " " << J_h(0, 1, 1) << " " << J_h(0, 1, 2)
              << "\n"
              << J_h(0, 2, 0) << " " << J_h(0, 2, 1) << " " << J_h(0, 2, 2)
              << "\n";
    const auto determinants = MeshField::getJacobianDeterminants(f, J);
    const auto determinants_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), determinants);
    std::cout << "tet jacobian determinant " << determinants_h(0) << "\n";
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        assert(std::fabs(J_h(0, i, j) - testCase.jacobian.at(i * 3 + j)) <=
               MeshField::MachinePrecision);
      }
    }
    assert(determinants.rank() == 1);
    assert(determinants.extent(0) == 1);
    assert(std::fabs(determinants_h(0) - testCase.jacobianDeterminant) <=
           MeshField::MachinePrecision);
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  tetJacobian();
  Kokkos::finalize();
  return 0;
}
