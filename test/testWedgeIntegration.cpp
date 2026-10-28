#include "KokkosController.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_For.hpp"
#include "MeshField_Integrate.hpp"
#include "MeshField_ShapeField.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>

// Checks the wedge quadrature:
// 1. every EdgeIntegration rule has positive weights and points inside
//    (-1, 1), and integrates zeta^c exactly for c up to its accuracy,
// 2. getWedgeIntegrationPoints(triOrder, zetaOrder) has the expected number
//    of points, positive weights and points inside the reference wedge, and
//    integrates xi0^a xi1^b zeta^c exactly for a + b <= triOrder and
//    c <= zetaOrder,
// 3. Integrator::process with LinearWedgeShape gives the exact volume of a
//    right prism and a tapered wedge, and a zeta order that is too low gives
//    the wrong volume for the tapered wedge.
// Failures are counted explicitly so the test also fails in builds where
// assert() is disabled.

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MeshField::Real;

bool isClose(Real a, Real b, Real tol) {
  return std::fabs(a - b) <= tol * (1 + std::fabs(b));
}

Real factorial(int n) { return n <= 1 ? 1.0 : n * factorial(n - 1); }

// exact integral of xi0^a xi1^b over the reference triangle
Real triangleMonomial(int a, int b) {
  return factorial(a) * factorial(b) / factorial(a + b + 2);
}

// exact integral of zeta^c over [-1, 1]
Real lineMonomial(int c) { return (c % 2 == 1) ? 0.0 : 2.0 / (c + 1); }

// A rule of accuracy p must integrate zeta^c exactly for c <= p. The c = 0
// case checks that the weights sum to 2, the length of the reference edge.
int checkEdgeRule(MeshField::Integration<1> const &rule) {
  const auto pts = rule.getPoints();
  const int p = rule.getAccuracy();
  int failed = (static_cast<int>(pts.size()) != rule.countPoints());
  for (auto const &pt : pts)
    failed += !(pt.weight > 0 && pt.param[0] > -1 && pt.param[0] < 1);
  for (int c = 0; c <= p; ++c) {
    Real sum = 0;
    for (auto const &pt : pts)
      sum += pt.weight * std::pow(pt.param[0], c);
    failed += !isClose(sum, lineMonomial(c), 1e-14);
  }
  std::cout << "edge rule, accuracy " << p << ", " << pts.size()
            << " points: failures " << failed << "\n";
  return failed;
}

int checkWedgeRule(int triOrder, int zetaOrder, int numTriPts,
                   int numGaussPts) {
  const auto pts = MeshField::getWedgeIntegrationPoints(triOrder, zetaOrder);
  int failed = (static_cast<int>(pts.size()) != numTriPts * numGaussPts);
  for (auto const &pt : pts) {
    const Real x0 = pt.param[0], x1 = pt.param[1], z = pt.param[2];
    failed += !(pt.weight > 0 && x0 >= 0 && x1 >= 0 && x0 + x1 <= 1 && z > -1 &&
                z < 1);
  }
  // a = b = c = 0 checks that the weights sum to 1
  for (int a = 0; a <= triOrder; ++a)
    for (int b = 0; a + b <= triOrder; ++b)
      for (int c = 0; c <= zetaOrder; ++c) {
        Real sum = 0;
        for (auto const &pt : pts)
          sum += pt.weight * std::pow(pt.param[0], a) *
                 std::pow(pt.param[1], b) * std::pow(pt.param[2], c);
        failed +=
            !isClose(sum, triangleMonomial(a, b) * lineMonomial(c), 1e-14);
      }
  std::cout << "wedge rule (" << triOrder << ", " << zetaOrder << "), "
            << pts.size() << " points: failures " << failed << "\n";
  return failed;
}

// node n of wedge w is vertex 6*w + n
struct WedgeToVertexField {
  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Wedge};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO wedgeNodeIdx, MeshField::LO wedgeCompIdx,
             MeshField::LO wedge, MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Wedge);
    return {0, wedgeCompIdx, 6 * wedge + wedgeNodeIdx, MeshField::Vertex};
  }
};

// sums w * |det J| over every point of every wedge
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

// clang-format off
// wedge 0: right prism, triangle (0,0),(2,0),(0,3) times height 4, volume 12
// wedge 1: tapered wedge, bottom triangle (0,0),(2,0),(0,2) at z = 0 and top
//   triangle (0,0),(1,0),(0,1) at z = 1, a frustum with volume
//   h/3 (A0 + A1 + sqrt(A0 A1)) = (2 + 1/2 + 1)/3 = 7/6
constexpr int numWedges = 2;
const Real wedgeCoords[numWedges][6][3] = {
  {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {0, 0, 4}, {2, 0, 4}, {0, 3, 4}},
  {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}};
// clang-format on
const Real exactVolume = 12.0 + 7.0 / 6.0;

template <typename CoordField> void setVtxCoords(CoordField field) {
  const int numVtx = 6 * numWedges;
  Kokkos::View<Real *[3], Kokkos::HostSpace> coords_h("coords_h", numVtx);
  for (int w = 0; w < numWedges; ++w)
    for (int n = 0; n < 6; ++n)
      for (int d = 0; d < 3; ++d)
        coords_h(6 * w + n, d) = wedgeCoords[w][n][d];
  auto coords = Kokkos::create_mirror_view_and_copy(ExecutionSpace(), coords_h);
  auto setCoordField = KOKKOS_LAMBDA(const int &vtx) {
    for (int d = 0; d < 3; ++d)
      field(vtx, 0, d, MeshField::Vertex) = coords(vtx, d);
  };
  MeshField::parallel_for<typename CoordField::Ctrlr>(
      {0}, {numVtx}, setCoordField, "setCoordField");
}

int checkIntegrator() {
  const MeshField::MeshInfo meshInfo{
      .numVtx = 6 * numWedges, .numWedge = numWedges, .dim = 3};
  auto coordField = MeshField::CreateCoordinateField<
      ExecutionSpace, MeshField::KokkosController, 3>(meshInfo);
  setVtxCoords(coordField.field);
  MeshField::FieldElement fe(numWedges, coordField.field,
                             MeshField::LinearWedgeShape(),
                             WedgeToVertexField());
  // det J of a linear wedge is degree 1 in (xi0, xi1) and degree 2 in zeta
  const int cases[][2] = {{1, 2}, {2, 2}, {3, 11}};
  int failed = 0;
  std::cout.precision(16);
  for (auto const &c : cases) {
    VolumeIntegrator vi(c[0], c[1]);
    vi.process(fe);
    const bool ok = isClose(vi.volume, exactVolume, 1e-13);
    failed += !ok;
    std::cout << "Integrator (" << c[0] << ", " << c[1] << "): volume "
              << vi.volume << ", expected " << exactVolume
              << (ok ? "" : " FAIL") << "\n";
  }
  // one Gauss point in zeta is not enough for the tapered wedge
  VolumeIntegrator low(1, 1);
  low.process(fe);
  const bool lowDiffers = !isClose(low.volume, exactVolume, 1e-6);
  failed += !lowDiffers;
  std::cout << "Integrator (1, 1): volume " << low.volume
            << (lowDiffers ? " (not exact, as expected)" : " FAIL") << "\n";
  return failed;
}

int runWedgeIntegrationChecks() {
  int failed = checkEdgeRule(MeshField::EdgeIntegration::N1());
  failed += checkEdgeRule(MeshField::EdgeIntegration::N2());
  failed += checkEdgeRule(MeshField::EdgeIntegration::N3());
  failed += checkEdgeRule(MeshField::EdgeIntegration::N4());
  failed += checkEdgeRule(MeshField::EdgeIntegration::N5());
  failed += checkEdgeRule(MeshField::EdgeIntegration::N6());
  failed += checkWedgeRule(1, 1, 1, 1);
  failed += checkWedgeRule(2, 2, 3, 2);
  failed += checkWedgeRule(2, 5, 3, 3);
  failed += checkWedgeRule(4, 4, 6, 3);
  failed += checkWedgeRule(3, 11, 6, 6);
  failed += checkIntegrator();
  return failed;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  const int numFailed = runWedgeIntegrationChecks();
  Kokkos::finalize();
  return numFailed ? 1 : 0;
}
