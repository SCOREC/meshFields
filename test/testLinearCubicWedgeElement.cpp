#include "MeshField_Defines.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Shape.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <Omega_h_vector.hpp>
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

// Checks MeshField::LinearCubicWedgeShape inside a MeshField::FieldElement:
// field evaluation (MeshField::evaluate) and the Jacobian
// (MeshField::getJacobians, MeshField::getJacobianDeterminants) for two cubic
// wedges. The coordinates are stored in a field from
// MeshField::CreateLagrangeField with LinearCubicWedgeShape: one DOF per vertex
// and two per vertical edge.
// The nodes are placed by geometric maps that are linear in (xi0, xi1)
// and cubic in zeta, so the element map reproduces them exactly and the
// expected values are the analytic maps and their derivatives. Failures are
// counted explicitly so the test also fails in builds where assert() is
// disabled.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;
using Vec3 = Omega_h::Vector<3>;

namespace {

constexpr int numWedges = 2;
constexpr int numNodes = MeshField::LinearCubicWedgeShape::numNodes;
constexpr int numVtxPerWedge = 6;
constexpr int numEdgesPerWedge = 3;

bool isClose(Real a, Real b) { return Omega_h::are_close(a, b, 1e-12, 1e-12); }

// Local node n of wedge w: nodes 0-5 are vertex 6*w + n; nodes 6+2a and 7+2a
// are the two DOFs on vertical edge 3*w + a, at zeta = -1/3 and +1/3.
struct CubicWedgeToField {
  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Wedge};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO wedgeNodeIdx, MeshField::LO wedgeCompIdx,
             MeshField::LO wedge, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Wedge);
    if (wedgeNodeIdx < numVtxPerWedge) {
      const MeshField::LO vtx = numVtxPerWedge * wedge + wedgeNodeIdx;
      return {0, wedgeCompIdx, vtx, MeshField::Vertex};
    }
    const MeshField::LO a = (wedgeNodeIdx - numVtxPerWedge) / 2;
    const MeshField::LO dof = (wedgeNodeIdx - numVtxPerWedge) % 2;
    const MeshField::LO edge = numEdgesPerWedge * wedge + a;
    return {dof, wedgeCompIdx, edge, MeshField::Edge};
  }
};

// Geometric maps that place the nodes of each wedge. Both are linear in
// (xi0, xi1) and cubic in zeta, so they lie in the element's function space.
Vec3 mapX(int w, Vec3 xi) {
  const Real x0 = xi[0], x1 = xi[1], z = xi[2];
  if (w == 0)
    return {2 * x0 + 0.3 * z * z * x1 + 0.2 * z * z * z,
            3 * x1 + 0.1 * z * z * z * x0 - 0.4 * z,
            2 + 2 * z + 0.25 * z * z * z};
  return {x0 + 0.5 * x1 + 0.3 * z * z * z * x0 + 0.1 * z,
          x1 + 0.2 * z * z * x1 + 0.4 * z * z,
          1.5 + z + 0.2 * z * z * z * x1 + 0.1 * z * z * z};
}

// Analytic derivatives of mapX. Row i is dx/dxi_i, the same layout as the
// MeshField Jacobian: J(i, j) = dx_j / dxi_i.
std::array<Vec3, 3> mapJ(int w, Vec3 xi) {
  const Real x0 = xi[0], x1 = xi[1], z = xi[2];
  if (w == 0)
    return {Vec3{2, 0.1 * z * z * z, 0}, Vec3{0.3 * z * z, 3, 0},
            Vec3{0.6 * z * x1 + 0.6 * z * z, 0.3 * z * z * x0 - 0.4,
                 2 + 0.75 * z * z}};
  return {Vec3{1 + 0.3 * z * z * z, 0, 0},
          Vec3{0.5, 1 + 0.2 * z * z, 0.2 * z * z * z},
          Vec3{0.9 * z * z * x0 + 0.1, 0.4 * z * x1 + 0.8 * z,
               1 + 0.6 * z * z * x1 + 0.3 * z * z}};
}

// Reference coordinates of the 12 nodes: six vertices, then two edge nodes on
// each vertical edge at zeta = -1/3 and +1/3.
const Vec3 nodeRef[numNodes] = {
    {0, 0, -1},      {1, 0, -1},     {0, 1, -1},      {0, 0, 1},
    {1, 0, 1},       {0, 1, 1},      {0, 0, -1. / 3}, {0, 0, 1. / 3},
    {1, 0, -1. / 3}, {1, 0, 1. / 3}, {0, 1, -1. / 3}, {0, 1, 1. / 3}};

// The 12 nodes, then interior points.
std::vector<Vec3> makeLocalPoints() {
  std::vector<Vec3> pts(nodeRef, nodeRef + numNodes);
  pts.push_back({1. / 3, 1. / 3, 0});
  pts.push_back({0.2, 0.3, -0.4});
  pts.push_back({0.6, 0.1, 0.7});
  pts.push_back({0.05, 0.9, 0.95});
  return pts;
}

// Writes the node positions from mapX into the coordinate field: vertex
// nodes to the vertex DOFs, and the zeta = -1/3 and +1/3 nodes to DOFs 0 and 1
// of their vertical edge.
template <typename CoordField> void setCubicWedgeCoords(CoordField field) {
  const int numVtx = numWedges * numVtxPerWedge;
  const int numEdges = numWedges * numEdgesPerWedge;
  Kokkos::View<Real *[3], Kokkos::HostSpace> vtx_h("vtx_h", numVtx);
  Kokkos::View<Real *[2][3], Kokkos::HostSpace> edge_h("edge_h", numEdges);

  for (int w = 0; w < numWedges; ++w)
    for (int n = 0; n < numNodes; ++n) {
      const Vec3 X = mapX(w, nodeRef[n]);
      for (int d = 0; d < 3; ++d) {
        if (n < numVtxPerWedge) {
          vtx_h(numVtxPerWedge * w + n, d) = X[d];
        } else {
          const int a = (n - numVtxPerWedge) / 2;
          const int dof = (n - numVtxPerWedge) % 2;
          edge_h(numEdgesPerWedge * w + a, dof, d) = X[d];
        }
      }
    }

  auto vtx = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), vtx_h);
  auto edge = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), edge_h);
  Kokkos::parallel_for(
      "setVtxCoords", Kokkos::RangePolicy<ExecutionSpace>(0, numVtx),
      KOKKOS_LAMBDA(const int v) {
        for (int d = 0; d < 3; ++d)
          field(v, 0, d, MeshField::Vertex) = vtx(v, d);
      });
  Kokkos::parallel_for(
      "setEdgeCoords", Kokkos::RangePolicy<ExecutionSpace>(0, numEdges),
      KOKKOS_LAMBDA(const int e) {
        for (int dof = 0; dof < 2; ++dof)
          for (int d = 0; d < 3; ++d)
            field(e, dof, d, MeshField::Edge) = edge(e, dof, d);
      });
}

// Compares the evaluated positions, Jacobians and determinants of wedge w
// with the analytic maps at every point in pts. Returns the failure count.
template <typename XView, typename JView, typename DetView>
int checkWedge(int w, const std::vector<Vec3> &pts, const XView &x_h,
               const JView &J_h, const DetView &detJ_h) {
  const int numPts = static_cast<int>(pts.size());
  int evalFailed = 0, jacFailed = 0, detFailed = 0;
  for (int p = 0; p < numPts; ++p) {
    const int row = w * numPts + p;
    const Vec3 xi = pts[p];
    const Vec3 xExpected = mapX(w, xi);
    for (int d = 0; d < 3; ++d)
      evalFailed += !isClose(x_h(row, d), xExpected[d]);
    const auto jExpected = mapJ(w, xi);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        jacFailed += !isClose(J_h(row, i, j), jExpected[i][j]);
    const Real detExpected =
        jExpected[0] * Omega_h::cross(jExpected[1], jExpected[2]);
    detFailed += !isClose(detJ_h(row), detExpected);
  }
  std::cout << "LinearCubicWedgeShape FieldElement, cubic wedge " << w
            << ": evaluate failures " << evalFailed << ", Jacobian failures "
            << jacFailed << ", det J failures " << detFailed << "\n";
  return evalFailed + jacFailed + detFailed;
}

// Builds the field and the FieldElement, runs the evaluation, Jacobian and
// determinant, and checks every wedge.
int runCubicWedgeElementChecks() {
  const auto pts = makeLocalPoints();
  const int numPts = static_cast<int>(pts.size());
  // numEdge counts only the vertical edges, which hold the zeta = +-1/3 DOFs
  const MeshField::MeshInfo meshInfo{.numVtx = numWedges * numVtxPerWedge,
                                     .numEdge = numWedges * numEdgesPerWedge,
                                     .numWedge = numWedges,
                                     .dim = 3};
  auto coordField =
      MeshField::CreateLagrangeField<ExecutionSpace,
                                     MeshField::KokkosController, Real, 3, 3, 3,
                                     MeshField::LinearCubicWedgeShape>(
          meshInfo);
  setCubicWedgeCoords(coordField.field);
  MeshField::FieldElement fe(numWedges, coordField.field,
                             MeshField::LinearCubicWedgeShape(),
                             CubicWedgeToField());
  // the same local points in every wedge, stored element-major
  Kokkos::View<Real **> lc("localCoords", numWedges * numPts, 3);
  auto lc_h = Kokkos::create_mirror_view(lc);
  for (int w = 0; w < numWedges; ++w)
    for (int p = 0; p < numPts; ++p)
      for (int d = 0; d < 3; ++d)
        lc_h(w * numPts + p, d) = pts[p][d];
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
    numFailed += checkWedge(w, pts, x_h, J_h, detJ_h);
  return numFailed;
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  const int numFailed = runCubicWedgeElementChecks();
  Kokkos::finalize();
  return numFailed ? 1 : 0;
}
