#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_For.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearTriangleToVertexField {
  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Triangle};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx,
             MeshField::LO tri, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Triangle);
    // Need to find which mesh vertex is described by the triangle and one of
    // its node indices. This could be implemented using element-to-dof holder
    // adjacencies, canonical ordering provided by the mesh database, which
    // would provide the index to the vertex in the dof holder array (assuming
    // the dof holder array is in the same order as vertex local numbering in
    // the mesh). For the simplicity of the test case, it is hard coded here
    // using local dof holder numbering:
    //      node
    // tri 0 1 2
    // 0   0 1 2
    MeshField::LO triNode2Vtx[1][3] = {{0, 1, 2}};
    const MeshField::LO vtx = triNode2Vtx[tri][triNodeIdx];
    return {0, triCompIdx, vtx, MeshField::Vertex};
  }
};

class TriangleTestCase {
public:
  TriangleTestCase(std::vector<MeshField::Real> coords_in,
                   std::vector<MeshField::Real> jacobian_in,
                   MeshField::Real jacobianDeterminant_in)
      : coords(coords_in), jacobian(jacobian_in),
        jacobianDeterminant(jacobianDeterminant_in) {
    assert(coords.size() == 6);
    assert(jacobian.size() == 4);
  };
  const std::vector<MeshField::Real> coords;
  const std::vector<MeshField::Real> jacobian;
  const MeshField::Real jacobianDeterminant;

private:
  TriangleTestCase();
};

template <typename ShapeField>
void setVtxCoords(size_t numVerts, size_t meshDim, TriangleTestCase testTri,
                  ShapeField field) {
  Kokkos::View<MeshField::Real *, Kokkos::HostSpace> coords_h(
      "coords_h", numVerts * meshDim);
  for (size_t i = 0; i < numVerts * meshDim; i++)
    coords_h[i] = testTri.coords.at(i);
  auto coords = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), coords_h);
  auto setCoordField = KOKKOS_LAMBDA(const int &vtx) {
    field(vtx, 0, 0, MeshField::Vertex) = coords(vtx * meshDim);
    field(vtx, 0, 1, MeshField::Vertex) = coords(vtx * meshDim + 1);
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {numVerts}, setCoordField,
                          "setCoordField");
}

// evaluate a field at the specified local coordinate for each edge
void triJacobian() {
  const MeshField::MeshInfo meshInfo{
      .numVtx = 3, .numEdge = 3, .numTri = 1, .dim = 2};
  auto coordField =
      MeshField::CreateCoordinateField<ExecutionSpace,
                                       MeshField::KokkosController>(meshInfo);

  TriangleTestCase rightTriangle({0, 0, 1, 0, 0, 1}, {1, 0, 0, 1}, 1);
  TriangleTestCase skewedTriangle({0, 0, 5, 1, 3, 4}, {5, 1, 3, 4}, 17);
  for (auto testCase : {rightTriangle, skewedTriangle}) {
    setVtxCoords(meshInfo.numVtx, meshInfo.dim, testCase, coordField);

    MeshField::FieldElement f(meshInfo.numTri, coordField,
                              MeshField::LinearTriangleShape(),
                              LinearTriangleToVertexField());

    Kokkos::View<MeshField::Real *[3]> lc("localCoords", 1);
    Kokkos::deep_copy(lc, 1.0 / 2);
    const auto numPtsPerElement = 1;
    const auto J = MeshField::getJacobians(f, lc, numPtsPerElement);
    const auto J_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
    assert(J_h.rank() == 3);
    assert(J_h.extent(0) == 1);
    assert(J_h.extent(1) == 2);
    assert(J_h.extent(2) == 2);
    std::cout << "tri jacobian\n"
              << J_h(0, 0, 0) << " " << J_h(0, 0, 1) << "\n"
              << J_h(0, 1, 0) << " " << J_h(0, 1, 1) << "\n";
    const auto determinants = MeshField::getJacobianDeterminants(f, J);
    const auto determinants_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), determinants);
    std::cout << "tri jacobian determinant " << determinants_h(0) << "\n";
    assert(std::fabs(J_h(0, 0, 0) - testCase.jacobian.at(0)) <=
           MeshField::MachinePrecision);
    assert(std::fabs(J_h(0, 0, 1) - testCase.jacobian.at(1)) <=
           MeshField::MachinePrecision);
    assert(std::fabs(J_h(0, 1, 0) - testCase.jacobian.at(2)) <=
           MeshField::MachinePrecision);
    assert(std::fabs(J_h(0, 1, 1) - testCase.jacobian.at(3)) <=
           MeshField::MachinePrecision);
    assert(determinants.rank() == 1);
    assert(determinants.extent(0) == 1);
    assert(std::fabs(determinants_h(0) - testCase.jacobianDeterminant) <=
           MeshField::MachinePrecision);
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  triJacobian();
  Kokkos::finalize();
  return 0;
}
