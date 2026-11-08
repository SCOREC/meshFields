#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_For.hpp"
#include "MeshField_Integrate.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_library.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

// Checks the extruded wedge element maps, getExtrudedWedgeElement and
// getExtrudedWedgeMeshInfo, on a 2D triangle mesh stacked into numLayers
// layers:
// 1. the DOF holder of every wedge node, for open and periodic stacks, linear
//    and cubic, with and without reverseOrientation, and the MeshInfo counts,
// 2. the field positions at the nodes and interior points of every wedge,
// 3. the volume from the Integrator,
// 4. the sign of det J with (R, phi, Z) = (x, s, y).
// The coordinates come from analytic maps X(x, y, s) that the elements
// reproduce exactly. Checks 2-4 use open stacks: in a periodic stack the last
// layer closes onto plane 0, which a single map X cannot describe. Failures
// are counted explicitly so the test also fails in builds where assert() is
// disabled.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::LO;
using MeshField::Real;
using Vec3 = std::array<Real, 3>;

namespace {

constexpr int numLayers = 3;
constexpr Real H = 1.5;           // height of the stack
constexpr Real h = H / numLayers; // height of one layer
constexpr Real A = 1.0;           // area of the 2D mesh

bool isClose(Real a, Real b, Real tol) {
  return std::fabs(a - b) <= tol * (1 + std::fabs(b));
}

// Analytic maps X(x, y, s) of the point (x, y) of the 2D mesh at height s
using Map = Vec3 (*)(Real, Real, Real);
Vec3 straightMap(Real x, Real y, Real s) { return {x, y, s}; }
Vec3 taperedMap(Real x, Real y, Real s) { return {x * (1 + 0.1 * s), y, s}; }
Vec3 curvedMap(Real x, Real y, Real s) {
  return {x * (1 + 0.1 * s) + 0.02 * s * s * s, y, s + 0.05 * s * s * s};
}
// R = x, phi = s, Z = y
Vec3 cylindricalMap(Real x, Real y, Real s) { return {x, s, y}; }

// Host copies of the 2D mesh
struct HostMesh2d {
  HostMesh2d(Omega_h::Mesh &mesh2d)
      : triVerts(mesh2d.ask_elem_verts()), coords(mesh2d.coords()),
        nVtx2d(mesh2d.nverts()), nTri2d(mesh2d.nfaces()) {}
  Omega_h::HostRead<Omega_h::LO> triVerts;
  Omega_h::HostRead<Omega_h::Real> coords;
  int nVtx2d;
  int nTri2d;

  // the 2D vertex of wedge corner a above triangle t
  LO cornerVtx(LO t, int a, bool reverse) const {
    const int corner[2][3] = {{0, 1, 2}, {0, 2, 1}};
    return triVerts[t * 3 + corner[reverse][a]];
  }

  Real signedArea(LO t) const {
    Real x[3], y[3];
    for (int a = 0; a < 3; ++a) {
      const LO v = triVerts[t * 3 + a];
      x[a] = coords[2 * v];
      y[a] = coords[2 * v + 1];
    }
    return 0.5 *
           ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]));
  }

  // X at the local point xi of wedge w in an open stack
  Vec3 position(Map X, LO w, Vec3 const &xi, bool reverse) const {
    const LO k = w / nTri2d, t = w % nTri2d;
    const Real L[3] = {1 - xi[0] - xi[1], xi[0], xi[1]};
    Real x = 0, y = 0;
    for (int a = 0; a < 3; ++a) {
      const LO v = cornerVtx(t, a, reverse);
      x += L[a] * coords[2 * v];
      y += L[a] * coords[2 * v + 1];
    }
    return X(x, y, k * h + h * (1 + xi[2]) / 2);
  }
};

template <int ShapeOrder>
auto createCoordField(const MeshField::MeshInfo &info) {
  if constexpr (ShapeOrder == 1) {
    return MeshField::CreateCoordinateField<ExecutionSpace,
                                            MeshField::KokkosController, 3>(
        info);
  } else {
    return MeshField::CreateLagrangeField<
        ExecutionSpace, MeshField::KokkosController, Real, 3, 3, 3,
        MeshField::LinearCubicWedgeShape>(info);
  }
}

// Writes X into the coordinate field of an open stack: vertex (p, v) at
// X(x_v, y_v, s_p) and, for the cubic field, DOFs 0 and 1 of edge (k, v) at
// X(x_v, y_v, s_k + h/3) and X(x_v, y_v, s_k + 2h/3).
template <int ShapeOrder, typename CoordField>
void setExtrudedCoords(CoordField field, const HostMesh2d &m, Map X) {
  const int numVtx = (numLayers + 1) * m.nVtx2d;
  const int numEdges = numLayers * m.nVtx2d;
  Kokkos::View<Real *[3], Kokkos::HostSpace> vtx_h("vtx_h", numVtx);
  Kokkos::View<Real *[2][3], Kokkos::HostSpace> edge_h("edge_h", numEdges);
  for (int v = 0; v < m.nVtx2d; ++v) {
    const Real x = m.coords[2 * v], y = m.coords[2 * v + 1];
    for (int p = 0; p <= numLayers; ++p) {
      const Vec3 pos = X(x, y, p * h);
      for (int d = 0; d < 3; ++d)
        vtx_h(p * m.nVtx2d + v, d) = pos[d];
    }
    for (int k = 0; k < numLayers; ++k)
      for (int dof = 0; dof < 2; ++dof) {
        const Vec3 pos = X(x, y, k * h + (dof + 1) * h / 3);
        for (int d = 0; d < 3; ++d)
          edge_h(k * m.nVtx2d + v, dof, d) = pos[d];
      }
  }
  auto vtx = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), vtx_h);
  auto setVtxCoords = KOKKOS_LAMBDA(const int &i) {
    for (int d = 0; d < 3; ++d)
      field(i, 0, d, MeshField::Vertex) = vtx(i, d);
  };
  MeshField::parallel_for<typename CoordField::Ctrlr>(
      {0}, {numVtx}, setVtxCoords, "setVtxCoords");
  if constexpr (ShapeOrder == 3) {
    auto edge = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), edge_h);
    auto setEdgeCoords = KOKKOS_LAMBDA(const int &e) {
      for (int dof = 0; dof < 2; ++dof)
        for (int d = 0; d < 3; ++d)
          field(e, dof, d, MeshField::Edge) = edge(e, dof, d);
    };
    MeshField::parallel_for<typename CoordField::Ctrlr>(
        {0}, {numEdges}, setEdgeCoords, "setEdgeCoords");
  }
}

const std::vector<Vec3> interiorPts = {
    {1. / 3, 1. / 3, 0}, {0.2, 0.3, -0.4}, {0.6, 0.1, 0.7}, {0.05, 0.9, 0.95}};

// The nodes of the shape, then interior points
template <typename Shape> std::vector<Vec3> nodesAndInteriorPoints() {
  const auto xi = Shape().getNodeParametricCoords();
  std::vector<Vec3> pts;
  for (size_t n = 0; n < Shape::numNodes; ++n)
    pts.push_back({xi[3 * n], xi[3 * n + 1], xi[3 * n + 2]});
  pts.insert(pts.end(), interiorPts.begin(), interiorPts.end());
  return pts;
}

// the same local points in every wedge, stored element-major
Kokkos::View<Real **> makeLocalCoords(int numWedges,
                                      const std::vector<Vec3> &pts) {
  const int numPts = static_cast<int>(pts.size());
  Kokkos::View<Real **> lc("localCoords", numWedges * numPts, 3);
  auto lc_h = Kokkos::create_mirror_view(lc);
  for (int w = 0; w < numWedges; ++w)
    for (int p = 0; p < numPts; ++p)
      for (int d = 0; d < 3; ++d)
        lc_h(w * numPts + p, d) = pts[p][d];
  Kokkos::deep_copy(lc, lc_h);
  return lc;
}

const char *shapeName(int shapeOrder) {
  return shapeOrder == 1 ? "linear" : "cubic";
}

// Gets the DOF holder of every (wedge, node) from the map on the device and
// compares it with the layout computed on the host. Also checks that the ids
// are in range, that the top of layer k is the bottom of layer k + 1 (and the
// top of the last layer is the bottom of layer 0 when periodic), and the
// MeshInfo counts.
template <int ShapeOrder>
int checkIds(Omega_h::Mesh &mesh2d, const HostMesh2d &m, bool periodic,
             bool reverse) {
  const auto elem = MeshField::Omegah::getExtrudedWedgeElement<ShapeOrder>(
      mesh2d, numLayers, periodic, reverse);
  const auto map = elem.map;
  const int numNodes = decltype(elem.shp)::numNodes;
  const int numWedges = numLayers * m.nTri2d;
  // node, entity, topology, and 1 if the component is passed through
  Kokkos::View<LO **> holders("holders", numWedges * numNodes, 4);
  Kokkos::parallel_for(
      "getHolders",
      Kokkos::RangePolicy<ExecutionSpace>(0, numWedges * numNodes),
      KOKKOS_LAMBDA(const int i) {
        const LO w = i / numNodes, n = i % numNodes;
        const auto holder = map(n, 0, w, MeshField::Wedge);
        LO compOk = holder.component == 0;
        for (LO c = 1; c < 3; ++c)
          compOk = compOk && map(n, c, w, MeshField::Wedge).component == c;
        holders(i, 0) = holder.node;
        holders(i, 1) = holder.entity;
        holders(i, 2) = holder.topo;
        holders(i, 3) = compOk;
      });
  const auto holders_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), holders);

  const auto info =
      MeshField::Omegah::getExtrudedWedgeMeshInfo(mesh2d, numLayers, periodic);
  const int numPlanes = periodic ? numLayers : numLayers + 1;
  int idFailed = 0, rangeFailed = 0, stackFailed = 0, countFailed = 0;
  for (LO w = 0; w < numWedges; ++w) {
    const LO k = w / m.nTri2d, t = w % m.nTri2d;
    const LO top = periodic ? (k + 1) % numLayers : k + 1;
    for (int n = 0; n < numNodes; ++n) {
      const int i = w * numNodes + n;
      const LO dof = holders_h(i, 0), ent = holders_h(i, 1),
               topo = holders_h(i, 2);
      LO expDof = 0, expEnt = 0, expTopo = MeshField::Vertex;
      if (n < 6) {
        const LO plane = n < 3 ? k : top;
        expEnt = plane * m.nVtx2d + m.cornerVtx(t, n % 3, reverse);
      } else {
        expDof = (n - 6) % 2;
        expEnt = k * m.nVtx2d + m.cornerVtx(t, (n - 6) / 2, reverse);
        expTopo = MeshField::Edge;
      }
      idFailed += !(dof == expDof && ent == expEnt && topo == expTopo &&
                    holders_h(i, 3) == 1);
      if (topo == MeshField::Vertex)
        rangeFailed += !(dof == 0 && ent >= 0 && ent < info.numVtx);
      else
        rangeFailed += !(topo == MeshField::Edge && (dof == 0 || dof == 1) &&
                         ent >= 0 && ent < info.numEdge);
    }
  }
  for (LO k = 0; k < numLayers; ++k) {
    if (k + 1 == numLayers && !periodic)
      continue;
    const LO next = (k + 1) % numLayers;
    for (LO t = 0; t < m.nTri2d; ++t)
      for (int a = 0; a < 3; ++a)
        stackFailed += holders_h((k * m.nTri2d + t) * numNodes + a + 3, 1) !=
                       holders_h((next * m.nTri2d + t) * numNodes + a, 1);
  }
  countFailed += info.dim != 3;
  countFailed += info.numVtx != numPlanes * m.nVtx2d;
  countFailed += info.numEdge != numLayers * m.nVtx2d;
  countFailed += info.numWedge != numWedges;
  countFailed += (info.numTri || info.numQuad || info.numTet || info.numHex ||
                  info.numPrism || info.numPyramid);
  std::cout << "ids, " << shapeName(ShapeOrder) << ", "
            << (periodic ? "periodic" : "open") << ", reverseOrientation "
            << reverse << ": id failures " << idFailed << ", range failures "
            << rangeFailed << ", stacking failures " << stackFailed
            << ", MeshInfo failures " << countFailed << "\n";
  return idFailed + rangeFailed + stackFailed + countFailed;
}

// Evaluates the field at the nodes and interior points of every wedge of an
// open stack and compares with X.
template <int ShapeOrder>
int checkPositions(Omega_h::Mesh &mesh2d, const HostMesh2d &m, Map X,
                   bool reverse, const char *mapName) {
  const auto info =
      MeshField::Omegah::getExtrudedWedgeMeshInfo(mesh2d, numLayers);
  auto coordField = createCoordField<ShapeOrder>(info);
  setExtrudedCoords<ShapeOrder>(coordField.field, m, X);
  const auto elem = MeshField::Omegah::getExtrudedWedgeElement<ShapeOrder>(
      mesh2d, numLayers, false, reverse);
  MeshField::FieldElement fe(info.numWedge, coordField.field, elem.shp,
                             elem.map);
  const auto pts = nodesAndInteriorPoints<decltype(elem.shp)>();
  const int numPts = static_cast<int>(pts.size());
  const auto lc = makeLocalCoords(info.numWedge, pts);
  const auto x = MeshField::evaluate(fe, lc, numPts);
  const auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  int failed = 0;
  for (int w = 0; w < info.numWedge; ++w)
    for (int p = 0; p < numPts; ++p) {
      const Vec3 expected = m.position(X, w, pts[p], reverse);
      for (int d = 0; d < 3; ++d)
        failed += !isClose(x_h(w * numPts + p, d), expected[d], 1e-12);
    }
  std::cout << "positions, " << shapeName(ShapeOrder) << ", " << mapName
            << ", reverseOrientation " << reverse << ": failures " << failed
            << "\n";
  return failed;
}

// sums w * dV over every point of every wedge
class VolumeIntegrator : public MeshField::Integrator {
public:
  VolumeIntegrator(int triOrder, int zetaOrder)
      : MeshField::Integrator(triOrder, zetaOrder) {}
  void atPoints(Kokkos::View<Real **>, Kokkos::View<Real *> w,
                Kokkos::View<Real *> dV) override {
    // w holds one weight per point of a single element; dV holds one entry
    // per point of every element
    const int numPtsPerElem = w.extent(0);
    Real sum = 0;
    Kokkos::parallel_reduce(
        "volume", dV.extent(0),
        KOKKOS_LAMBDA(const int i, Real &s) {
          s += w(i % numPtsPerElem) * dV(i);
        },
        sum);
    volume = sum;
  }
  Real volume = 0;
};

// dV is signed, so X must give positively oriented wedges
template <int ShapeOrder>
int checkVolume(Omega_h::Mesh &mesh2d, const HostMesh2d &m, Map X,
                Real expected, const char *mapName) {
  const auto info =
      MeshField::Omegah::getExtrudedWedgeMeshInfo(mesh2d, numLayers);
  auto coordField = createCoordField<ShapeOrder>(info);
  setExtrudedCoords<ShapeOrder>(coordField.field, m, X);
  const auto elem =
      MeshField::Omegah::getExtrudedWedgeElement<ShapeOrder>(mesh2d, numLayers);
  MeshField::FieldElement fe(info.numWedge, coordField.field, elem.shp,
                             elem.map);
  VolumeIntegrator vi(3, 11);
  vi.process(fe);
  const bool ok = isClose(vi.volume, expected, 1e-12);
  std::cout << "volume, " << shapeName(ShapeOrder) << ", " << mapName << ": "
            << vi.volume << ", expected " << expected << (ok ? "" : " FAIL")
            << "\n";
  return !ok;
}

// With (R, phi, Z) = (x, s, y), det J has the sign -sign(area) of the 2D
// triangle, and +sign(area) with reverseOrientation. Uses getJacobians, not
// the Integrator, since Debug builds assert dV >= 0 there.
template <int ShapeOrder>
int checkOrientation(Omega_h::Mesh &mesh2d, const HostMesh2d &m, bool reverse) {
  const auto info =
      MeshField::Omegah::getExtrudedWedgeMeshInfo(mesh2d, numLayers);
  auto coordField = createCoordField<ShapeOrder>(info);
  setExtrudedCoords<ShapeOrder>(coordField.field, m, cylindricalMap);
  const auto elem = MeshField::Omegah::getExtrudedWedgeElement<ShapeOrder>(
      mesh2d, numLayers, false, reverse);
  MeshField::FieldElement fe(info.numWedge, coordField.field, elem.shp,
                             elem.map);
  const int numPts = static_cast<int>(interiorPts.size());
  const auto lc = makeLocalCoords(info.numWedge, interiorPts);
  const auto J = MeshField::getJacobians(fe, lc, numPts);
  const auto detJ = MeshField::getJacobianDeterminants(fe, J);
  const auto detJ_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), detJ);
  int failed = 0;
  for (int w = 0; w < info.numWedge; ++w) {
    const Real area = m.signedArea(w % m.nTri2d);
    const Real areaSign = (area > 0) - (area < 0);
    const Real expectedSign = reverse ? areaSign : -areaSign;
    for (int p = 0; p < numPts; ++p)
      failed += !(detJ_h(w * numPts + p) * expectedSign > 0);
  }
  std::cout << "orientation, " << shapeName(ShapeOrder)
            << ", reverseOrientation " << reverse << ": wrong signs " << failed
            << " of " << info.numWedge * numPts << "\n";
  return failed;
}

int runExtrudedWedgeChecks(Omega_h::Mesh &mesh2d) {
  const HostMesh2d m(mesh2d);
  int failed = 0;
  for (bool periodic : {false, true})
    for (bool reverse : {false, true}) {
      failed += checkIds<1>(mesh2d, m, periodic, reverse);
      failed += checkIds<3>(mesh2d, m, periodic, reverse);
    }
  for (bool reverse : {false, true}) {
    failed += checkPositions<1>(mesh2d, m, taperedMap, reverse, "tapered");
    failed += checkPositions<3>(mesh2d, m, curvedMap, reverse, "curved");
  }
  std::cout.precision(16);
  failed += checkVolume<1>(mesh2d, m, straightMap, A * H, "straight");
  failed +=
      checkVolume<1>(mesh2d, m, taperedMap, A * (H + 0.05 * H * H), "tapered");
  failed += checkVolume<3>(
      mesh2d, m, curvedMap,
      A * (H + 0.05 * H * H + 0.05 * H * H * H + 0.00375 * H * H * H * H),
      "sheared and curved");
  for (bool reverse : {false, true}) {
    failed += checkOrientation<1>(mesh2d, m, reverse);
    failed += checkOrientation<3>(mesh2d, m, reverse);
  }
  return failed;
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int numFailed = 0;
  {
    auto lib = Omega_h::Library(&argc, &argv);
    auto mesh2d = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0,
                                     0.0, 2, 2, 0);
    numFailed = runExtrudedWedgeChecks(mesh2d);
  }
  Kokkos::finalize();
  return numFailed ? 1 : 0;
}
