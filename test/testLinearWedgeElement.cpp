#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Field.hpp"
#include "MeshField_For.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <Omega_h_vector.hpp>
#include <array>
#include <cassert>
#include <iostream>

// Checks MeshField::LinearWedgeShape inside a MeshField::FieldElement: field
// evaluation (MeshField::evaluate) and the Jacobian (MeshField::getJacobians,
// MeshField::getJacobianDeterminants) for two wedges, compared against the
// geometric map x(xi) differentiated by hand. Failures are counted explicitly
// so the test also fails in builds where assert() is disabled.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;
using Vec3 = Omega_h::Vector<3>;

constexpr int numWedges = 2;
constexpr int numNodes = MeshField::LinearWedgeShape::numNodes;

bool isClose(Real a, Real b) {
  return Omega_h::are_close(a, b, 1e-12, 1e-12);
}

// Each wedge has its own six vertices: node n of wedge w is vertex 6*w + n.
// Nodes 0-2 are the zeta = -1 triangle and nodes 3-5 the zeta = +1 triangle,
// matching the LinearWedgeShape node ordering.
struct WedgeToVertexField {
  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Wedge};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO wedgeNodeIdx, MeshField::LO wedgeCompIdx,
             MeshField::LO wedge, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Wedge);
    const MeshField::LO vtx = numNodes * wedge + wedgeNodeIdx;
    return {0, wedgeCompIdx, vtx, MeshField::Vertex};
  }
};

// clang-format off
// wedge 0: right prism, triangle (0,0),(2,0),(0,3) swept by height 4
//   x(xi) = (2 xi0, 3 xi1, 2 (1 + zeta)), J = diag(2, 3, 2), det J = 12
// wedge 1: the top triangle is shifted, scaled and tilted, so J varies with
//   xi0, xi1 and zeta
const Vec3 wedgeCoords[numWedges][numNodes] = {
  {{0, 0, 0},     {2, 0, 0},     {0, 3, 0},
   {0, 0, 4},     {2, 0, 4},     {0, 3, 4}},
  {{0, 0, 0},     {1, 0, 0},     {0, 1, 0},
   {0.2, 0.1, 1}, {1.7, 0.3, 1.2}, {0.1, 1.4, 0.9}}};
const char *wedgeNames[numWedges] = {"right prism", "skewed wedge"};

// the six nodes, the centroid, and interior points
const Vec3 localPts[] = {
  {0, 0, -1}, {1, 0, -1}, {0, 1, -1}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1},
  {1. / 3, 1. / 3, 0},
  {0.2, 0.3, -0.4}, {0.6, 0.1, 0.7}, {0.05, 0.9, 0.95}};
// clang-format on
constexpr int numPts = static_cast<int>(sizeof(localPts) / sizeof(localPts[0]));
constexpr int centroidPt = 6;

// x(xi) = sum_a L_a(xi0, xi1) [ (1-zeta)/2 B_a + (1+zeta)/2 T_a ] with B_a the
// bottom (nodes 0-2) and T_a the top (nodes 3-5) vertex coordinates
Vec3 expectedX(const Vec3 (&nodes)[numNodes], Vec3 xi) {
  const Real L[3] = {1 - xi[0] - xi[1], xi[0], xi[1]};
  Vec3 x = Omega_h::zero_vector<3>();
  for (int a = 0; a < 3; ++a) {
    x += L[a] * (1 - xi[2]) / 2 * nodes[a];
    x += L[a] * (1 + xi[2]) / 2 * nodes[a + 3];
  }
  return x;
}

// rows of the MeshField Jacobian, J(i,j) = dx_j/dxi_i, from differentiating
// x(xi) directly:
//   dx/dxi0  = (1-zeta)/2 (B1 - B0) + (1+zeta)/2 (T1 - T0)
//   dx/dxi1  = (1-zeta)/2 (B2 - B0) + (1+zeta)/2 (T2 - T0)
//   dx/dzeta = 1/2 sum_a L_a (T_a - B_a)
std::array<Vec3, 3> expectedJ(const Vec3 (&nodes)[numNodes], Vec3 xi) {
  const Real down = (1 - xi[2]) / 2;
  const Real up = (1 + xi[2]) / 2;
  const Real L[3] = {1 - xi[0] - xi[1], xi[0], xi[1]};
  const Vec3 dXi0 = down * (nodes[1] - nodes[0]) + up * (nodes[4] - nodes[3]);
  const Vec3 dXi1 = down * (nodes[2] - nodes[0]) + up * (nodes[5] - nodes[3]);
  Vec3 dZeta = Omega_h::zero_vector<3>();
  for (int a = 0; a < 3; ++a)
    dZeta += L[a] / 2 * (nodes[a + 3] - nodes[a]);
  return {dXi0, dXi1, dZeta};
}

// Writes the wedge vertex coordinates into the coordinate field.
template <typename CoordField>
void setVtxCoords(CoordField field) {
  const int numVtx = numWedges * numNodes;
  Kokkos::View<Real *[3], Kokkos::HostSpace> coords_h("coords_h", numVtx);
  for (int w = 0; w < numWedges; ++w)
    for (int n = 0; n < numNodes; ++n)
      for (int d = 0; d < 3; ++d)
        coords_h(numNodes * w + n, d) = wedgeCoords[w][n][d];
  auto coords = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), coords_h);
  auto setCoordField = KOKKOS_LAMBDA(const int &vtx) {
    for (int d = 0; d < 3; ++d)
      field(vtx, 0, d, MeshField::Vertex) = coords(vtx, d);
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {numVtx}, setCoordField,
                          "setCoordField");
}

// Compares the evaluated positions, Jacobians and determinants of wedge w with
// the hand-differentiated map at every local point. Returns the failure count.
template <class XView, class JView, class DetView>
int checkWedge(int w, const XView &x_h, const JView &J_h,
               const DetView &detJ_h) {
  const auto &nodes = wedgeCoords[w];
  int evalFailed = 0, jacFailed = 0, detFailed = 0;
  for (int p = 0; p < numPts; ++p) {
    const int row = w * numPts + p;
    // at a node, x is that node; at the centroid, x is the mean of the six
    // vertices; everywhere else x follows the map
    Vec3 xExpected = expectedX(nodes, localPts[p]);
    if (p < numNodes)
      xExpected = nodes[p];
    if (p == centroidPt) {
      xExpected = Omega_h::zero_vector<3>();
      for (int n = 0; n < numNodes; ++n)
        xExpected += nodes[n] / Real(numNodes);
    }
    for (int d = 0; d < 3; ++d)
      evalFailed += !isClose(x_h(row, d), xExpected[d]);

    const auto jExpected = expectedJ(nodes, localPts[p]);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        jacFailed += !isClose(J_h(row, i, j), jExpected[i][j]);
    const Real detExpected =
        jExpected[0] * Omega_h::cross(jExpected[1], jExpected[2]);
    detFailed += !isClose(detJ_h(row), detExpected);
  }
  std::cout << "LinearWedgeShape FieldElement, " << wedgeNames[w]
            << ": evaluate failures " << evalFailed << ", Jacobian failures "
            << jacFailed << ", det J failures " << detFailed << "\n";
  return evalFailed + jacFailed + detFailed;
}

// Builds the coordinate field and the FieldElement, runs the evaluation,
// Jacobian and determinant, and checks every wedge. Must be called between
// Kokkos::initialize and Kokkos::finalize.
int runLinearWedgeElementChecks() {
  const MeshField::MeshInfo meshInfo{
      .numVtx = numWedges * numNodes, .numWedge = numWedges, .dim = 3};
  auto coordField = MeshField::CreateCoordinateField<
      ExecutionSpace, MeshField::KokkosController, 3>(meshInfo);
  setVtxCoords(coordField.field);

  MeshField::FieldElement fe(numWedges, coordField.field,
                             MeshField::LinearWedgeShape(),
                             WedgeToVertexField());

  // the same local points in every wedge, stored element-major
  Kokkos::View<Real **> lc("localCoords", numWedges * numPts, 3);
  auto lc_h = Kokkos::create_mirror_view(lc);
  for (int w = 0; w < numWedges; ++w)
    for (int p = 0; p < numPts; ++p)
      for (int d = 0; d < 3; ++d)
        lc_h(w * numPts + p, d) = localPts[p][d];
  Kokkos::deep_copy(lc, lc_h);

  const auto x = MeshField::evaluate(fe, lc, numPts);
  const auto J = MeshField::getJacobians(fe, lc, numPts);
  const auto detJ = MeshField::getJacobianDeterminants(fe, J);
  const auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  const auto J_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
  const auto detJ_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), detJ);

  int numFailed = 0;
  for (int w = 0; w < numWedges; ++w)
    numFailed += checkWedge(w, x_h, J_h, detJ_h);
  return numFailed;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  const int numFailed = runLinearWedgeElementChecks();
  Kokkos::finalize();
  return numFailed ? 1 : 0;
}
