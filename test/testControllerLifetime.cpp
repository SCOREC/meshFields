// Diagnostic test for https://github.com/SCOREC/meshFields/issues/37
//
// Reproduces the quadratic-triangle regression seen after removing the
// unused ShapeField::meshField member (commit "don't store the controller").
// Manually builds the KokkosController and quadratic ShapeField, keeping
// the controller alive in the caller's scope for the whole test, to check
// whether the failure is caused by the controller going out of scope.
#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using namespace MeshField;

struct QuadraticTriangleToField {
  KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() const {
    return {MeshField::Triangle};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO triNodeIdx, MeshField::LO triCompIdx,
             MeshField::LO ent, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Triangle);
    assert(ent == 0);
    MeshField::LO triNode2DofHolder[6] = {/*vertices*/ 0, 1, 2,
                                          /*edges*/ 0, 1, 2};
    MeshField::Mesh_Topology triNode2DofHolderTopo[6] = {
        MeshField::Vertex, MeshField::Vertex, MeshField::Vertex,
        MeshField::Edge,   MeshField::Edge,   MeshField::Edge};
    const auto dofHolder = triNode2DofHolder[triNodeIdx];
    const auto dofHolderTopo = triNode2DofHolderTopo[triNodeIdx];
    return {0, 0, dofHolder, dofHolderTopo};
  }
};

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    // triangle: vtx 0,1,2 with edges 0(v0-v1), 1(v1-v2), 2(v2-v0)
    const MeshInfo meshInfo{
        .numVtx = 3, .numEdge = 3, .numTri = 1, .dim = 2};
    const int numComp = 1;

    using Ctrlr =
        MeshField::KokkosController<MemorySpace, ExecutionSpace,
                                    MeshField::Real ***, MeshField::Real ***>;
    // kept alive for the entire duration of the test - controller is NOT
    // stored inside the returned ShapeField anymore.
    Ctrlr kk_ctrl({/*field 0*/ meshInfo.numVtx, 1, numComp,
                  /*field 1*/ meshInfo.numEdge, 1, numComp});

    auto vtxField = MeshField::makeField<Ctrlr, 0>(kk_ctrl);
    auto edgeField = MeshField::makeField<Ctrlr, 1>(kk_ctrl);
    using LA = decltype(vtxField);
    using EA = decltype(edgeField);
    using QA = QuadraticAccessor<LA, EA>;
    using QLSF = ShapeField<numComp, QuadraticTriangleShape, QA>;
    QLSF field(meshInfo, {vtxField, edgeField});

    // f(x,y) = 2x + y ; set exact analytic values at vertices and edge
    // midpoints of a right triangle: v0=(0,0) v1=(1,0) v2=(0,1)
    // edge0 = midpoint(v0,v1) = (0.5,0)   -> f=1.0
    // edge1 = midpoint(v1,v2) = (0.5,0.5) -> f=1.5
    // edge2 = midpoint(v2,v0) = (0,0.5)   -> f=0.5
    Kokkos::parallel_for(
        "setVtx", 1, KOKKOS_LAMBDA(const int) {
          field(0, 0, 0, MeshField::Vertex) = 0.0; // v0 (0,0)
          field(1, 0, 0, MeshField::Vertex) = 2.0; // v1 (1,0)
          field(2, 0, 0, MeshField::Vertex) = 1.0; // v2 (0,1)
          field(0, 0, 0, MeshField::Edge) = 1.0;   // edge0 mid
          field(1, 0, 0, MeshField::Edge) = 1.5;   // edge1 mid
          field(2, 0, 0, MeshField::Edge) = 0.5;   // edge2 mid
        });
    Kokkos::fence();

    MeshField::FieldElement f(meshInfo.numTri, field,
                              MeshField::QuadraticTriangleShape(),
                              QuadraticTriangleToField());

    // evaluate at centroid (1/3, 1/3) -> expected f(x,y) at physical
    // centroid of (0,0),(1,0),(0,1) = (1/3,1/3): 2*(1/3)+(1/3) = 1.0
    Kokkos::View<MeshField::Real[1][2]> lc("localCoords");
    Kokkos::deep_copy(lc, 1.0 / 3);
    auto result = MeshField::evaluate(f, lc);

    MeshField::LO numErrors = 0;
    Kokkos::parallel_reduce(
        "check", 1,
        KOKKOS_LAMBDA(const int &i, MeshField::LO &lerrors) {
          const auto computed = result(i, 0);
          const auto expected = 1.0;
          if (Kokkos::fabs(computed - expected) > 1e-9) {
            Kokkos::printf("MISMATCH: expected %f computed %f\n", expected,
                          computed);
            lerrors += 1;
          } else {
            Kokkos::printf("OK: expected %f computed %f\n", expected,
                          computed);
          }
        },
        numErrors);
    if (numErrors > 0) {
      Kokkos::finalize();
      return 1;
    }
  }
  Kokkos::finalize();
  return 0;
}
