#include "MeshField_Bell.hpp"
#include "MeshField_MappedElement.hpp"
#include "MeshField_TestHarness.hpp"
#include <string>

// Tests for the Bell element.
//
// 1. reference duality: the 21 functionals of the extended Bell reference
//    element (vertex values, gradients, and Hessians, and the edge moments of
//    the normal derivative against the degree 4 Legendre polynomial) applied
//    to the 21 reference basis functions give the identity.  Hessians are
//    computed by Richardson-extrapolated central differences of the
//    gradients, which is exact for these polynomials up to rounding.
// 2. physical duality, on single triangles of several shapes: setting stored
//    dof i to 1 and the rest to 0 gives the i-th physical basis function,
//    whose physical dofs (vertex values, gradients, and Hessians) are e_i and
//    whose normal derivative is cubic on every edge (its physical edge
//    moments vanish).
// 3. reproduction: the Bell space contains the quartics, so a quartic with
//    exact dofs is reproduced in value and gradient; x^5 is not.

using MeshField::LO;
using MeshField::Real;
using MeshField::Vector2;

// coordinate d of reference triangle vertex v, matching BellReferenceShape
KOKKOS_INLINE_FUNCTION Real refVert(int v, int d) {
  constexpr Real verts[3][2] = {{0, 0}, {1, 0}, {0, 1}};
  return verts[v][d];
}

// vertex i (0 or 1) of reference edge e, matching BellReferenceShape
KOKKOS_INLINE_FUNCTION int refEdge(int e, int i) {
  constexpr int edges[3][2] = {{1, 2}, {0, 2}, {0, 1}};
  return edges[e][i];
}

KOKKOS_INLINE_FUNCTION Real legendre4(Real u) {
  return (35 * u * u * u * u - 30 * u * u + 3) / 8;
}

// moment of the normal derivative against the degree 4 Legendre polynomial
// along edge e, which runs from reference vertex a to b; T is the edge vector
// in the frame of the gradients returned by gradAt(xi) at parametric xi
template <typename GradAt>
KOKKOS_INLINE_FUNCTION Real edgeNormalMoment(int e, const Real T[2],
                                             GradAt gradAt) {
  // 5 point Gauss-Legendre rule on [-1, 1]; exact to degree 9
  constexpr int numGaussPts = 5;
  constexpr Real gaussPts[numGaussPts] = {
      -0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831,
      0.9061798459386640};
  constexpr Real gaussWts[numGaussPts] = {
      0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
      0.4786286704993665, 0.2369268850561891};
  const int a = refEdge(e, 0), b = refEdge(e, 1);
  const Real len = Kokkos::sqrt(T[0] * T[0] + T[1] * T[1]);
  const Real n[2] = {T[1] / len, -T[0] / len};
  Real moment = 0;
  for (int q = 0; q < numGaussPts; ++q) {
    const Real u = gaussPts[q];
    const Real s = (u + 1) / 2;
    const auto g =
        gradAt(Vector2{refVert(a, 0) + s * (refVert(b, 0) - refVert(a, 0)),
                       refVert(a, 1) + s * (refVert(b, 1) - refVert(a, 1))});
    moment += gaussWts[q] / 2 * legendre4(u) * (n[0] * g[0] + n[1] * g[1]);
  }
  return moment;
}

// the functionals applied to basis function k: 18 vertex functionals, then 3
// edge moments
KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 21> applyFunctionals(int k) {
  const MeshField::BellReferenceShape shape;
  const Real h = 1e-3; // difference step for the Hessian
  Kokkos::Array<Real, 21> f;
  for (int v = 0; v < 3; ++v) {
    const Vector2 x{refVert(v, 0), refVert(v, 1)};
    const auto grad = shape.getLocalGradients(x);
    f[v * 6] = shape.getValues(x)[k];
    f[v * 6 + 1] = grad[k * 2];
    f[v * 6 + 2] = grad[k * 2 + 1];
    // central differences of the gradient with steps h and 2h, combined to
    // cancel the h^2 error term; exact for the quartic gradients here
    const auto d = [&](int dir, int comp, Real step) {
      Vector2 p = x, m = x;
      p[dir] += step;
      m[dir] -= step;
      return (shape.getLocalGradients(p)[k * 2 + comp] -
              shape.getLocalGradients(m)[k * 2 + comp]) /
             (2 * step);
    };
    const auto hess = [&](int dir, int comp) {
      return (4 * d(dir, comp, h) - d(dir, comp, 2 * h)) / 3;
    };
    f[v * 6 + 3] = hess(0, 0);
    f[v * 6 + 4] = hess(1, 0);
    f[v * 6 + 5] = hess(1, 1);
  }
  for (int e = 0; e < 3; ++e) {
    const int a = refEdge(e, 0), b = refEdge(e, 1);
    const Real T[2] = {refVert(b, 0) - refVert(a, 0),
                       refVert(b, 1) - refVert(a, 1)};
    f[18 + e] = edgeNormalMoment(e, T, [&](Vector2 xi) {
      const auto g = shape.getLocalGradients(xi);
      return Kokkos::Array<Real, 2>{g[k * 2], g[k * 2 + 1]};
    });
  }
  return f;
}

// 1. the reference functionals applied to the reference basis are the identity
void testReferenceDuality() {
  LO errors = 0;
  Kokkos::parallel_reduce(
      "checkReferenceDuality", Kokkos::RangePolicy<ExecutionSpace>(0, 21),
      KOKKOS_LAMBDA(const int k, LO &lerr) {
        const auto f = applyFunctionals(k);
        for (int i = 0; i < 21; ++i) {
          const bool hessian = i < 18 && i % 6 >= 3;
          const Real tol = hessian ? 1e-8 : MeshField::Epsilon;
          const Real expected = i == k ? 1.0 : 0.0;
          if (!(Kokkos::fabs(f[i] - expected) <= tol)) {
            Kokkos::printf("functional %d of basis function %d: %.3e\n", i, k,
                           f[i]);
            lerr++;
          }
        }
      },
      errors);
  report("testReferenceDuality", errors);
}

// ---------------------------------------------------------------------------
// single triangle elements

// triangle 0 with nodes 0, 1, 2 at vertices 0, 1, 2
struct TriangleToVertexField {
  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Triangle};
  }
  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(LO node, LO comp, LO, MeshField::Mesh_Topology) const {
    return {0, comp, node, MeshField::Vertex};
  }
};

// Bell dofs of triangle 0: element node n is node n % 6 of vertex n / 6
struct BellTriangleToVertexField {
  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Triangle};
  }
  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(LO node, LO comp, LO, MeshField::Mesh_Topology) const {
    constexpr LO nv = MeshField::BellTriangleShape::nodesPerVertex;
    return {node % nv, comp, node / nv, MeshField::Vertex};
  }
};

// the vertex coordinates of a triangle; kept apart from Triangle::name so
// that device code can capture them
struct TriangleCoords {
  Real xy[3][2];
};

struct Triangle {
  TriangleCoords coords;
  std::string name;
};

// a Bell field and P1 coordinates on one triangle
template <template <typename...> typename Controller> struct SingleTriangle {
  const MeshField::MeshInfo meshInfo{
      .numVtx = 3, .numEdge = 3, .numTri = 1, .dim = 2};
  decltype(MeshField::CreateBellField<ExecutionSpace, Controller, Real, 1>(
      MeshField::MeshInfo{})) bell;
  decltype(MeshField::CreateLagrangeField<ExecutionSpace, Controller, Real, 1,
                                          2, 2>(MeshField::MeshInfo{})) coords;
  const TriangleCoords t;

  SingleTriangle(TriangleCoords const &tIn)
      : bell(MeshField::CreateBellField<ExecutionSpace, Controller, Real, 1>(
            meshInfo)),
        coords(MeshField::CreateLagrangeField<ExecutionSpace, Controller, Real,
                                              1, 2, 2>(meshInfo)),
        t(tIn) {
    setCoords();
  }

  // a member function, since nvcc does not allow device lambdas in
  // constructors
  void setCoords() {
    auto cf = coords.field;
    const TriangleCoords tc = t;
    Kokkos::parallel_for(
        "setCoords", Kokkos::RangePolicy<ExecutionSpace>(0, 3),
        KOKKOS_LAMBDA(const int v) {
          cf(v, 0, 0, MeshField::Vertex) = tc.xy[v][0];
          cf(v, 0, 1, MeshField::Vertex) = tc.xy[v][1];
        });
  }

  auto element() const {
    MeshField::FieldElement fieldElm(1, bell.field,
                                     MeshField::BellTriangleShape(),
                                     BellTriangleToVertexField());
    MeshField::FieldElement geomElm(1, coords.field,
                                    MeshField::LinearTriangleShape(),
                                    TriangleToVertexField());
    return MeshField::MappedElement(fieldElm, geomElm,
                                    MeshField::BellDofTransformation());
  }

  // dofs from a function with value, grad, and hess members
  template <typename Func> void setDofs(Func f) {
    auto bf = bell.field;
    const TriangleCoords tc = t;
    Kokkos::parallel_for(
        "setDofs", Kokkos::RangePolicy<ExecutionSpace>(0, 3),
        KOKKOS_LAMBDA(const int v) {
          const Real x = tc.xy[v][0], y = tc.xy[v][1];
          const auto g = f.grad(x, y);
          const auto h = f.hess(x, y);
          bf(v, 0, 0, MeshField::Vertex) = f.value(x, y);
          bf(v, 1, 0, MeshField::Vertex) = g[0];
          bf(v, 2, 0, MeshField::Vertex) = g[1];
          bf(v, 3, 0, MeshField::Vertex) = h[0];
          bf(v, 4, 0, MeshField::Vertex) = h[1];
          bf(v, 5, 0, MeshField::Vertex) = h[2];
        });
  }

  // stored dof i is 1, the rest 0
  void setUnitDof(int i) {
    auto bf = bell.field;
    Kokkos::parallel_for(
        "setUnitDof", Kokkos::RangePolicy<ExecutionSpace>(0, 18),
        KOKKOS_LAMBDA(const int n) {
          bf(n / 6, n % 6, 0, MeshField::Vertex) = n == i ? 1.0 : 0.0;
        });
  }
};

// physical Hessian at xi from Richardson-extrapolated central differences of
// the physical gradient along xi: d(grad)/dxi_i = sum_k J[i][k] d(grad)/dx_k,
// so H = J^{-1} (d grad / d xi)
template <typename Elm, typename C, typename G>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3>
physicalHessian(Elm const &elm, C const &c, G const &geo, Vector2 xi) {
  const Real h = 1e-3;
  const auto d = [&](int dir, int comp, Real step) {
    Vector2 p = xi, m = xi;
    p[dir] += step;
    m[dir] -= step;
    return (elm.gradientFromCoefficients(c, geo, p)[comp] -
            elm.gradientFromCoefficients(c, geo, m)[comp]) /
           (2 * step);
  };
  Real dG[2][2]; // dG[i][j] = d grad_j / d xi_i
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      dG[i][j] = (4 * d(i, j, h) - d(i, j, 2 * h)) / 3;
  const auto Jinv = MeshField::invert<2>(geo);
  Real H[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      H[i][j] = Jinv[i * 2] * dG[0][j] + Jinv[i * 2 + 1] * dG[1][j];
  return {H[0][0], H[0][1], H[1][1]};
}

// 2. physical duality and cubic normal derivatives
template <template <typename...> typename Controller>
void testPhysicalDuality(Triangle const &tri) {
  SingleTriangle<Controller> st(tri.coords);
  const auto elm = st.element();
  const TriangleCoords t = tri.coords;
  LO errors = 0;
  for (int i = 0; i < 18; ++i) {
    st.setUnitDof(i);
    LO lerrors = 0;
    Kokkos::parallel_reduce(
        "checkPhysicalDuality", Kokkos::RangePolicy<ExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int, LO &lerr) {
          const auto c = elm.coefficients(0);
          const auto geo = elm.geometry(0);
          Real f[18];
          for (int v = 0; v < 3; ++v) {
            const Vector2 xi{refVert(v, 0), refVert(v, 1)};
            const auto g = elm.gradientFromCoefficients(c, geo, xi);
            const auto H = physicalHessian(elm, c, geo, xi);
            f[v * 6] = elm.valueFromCoefficients(c, xi)[0];
            f[v * 6 + 1] = g[0];
            f[v * 6 + 2] = g[1];
            f[v * 6 + 3] = H[0];
            f[v * 6 + 4] = H[1];
            f[v * 6 + 5] = H[2];
          }
          for (int j = 0; j < 18; ++j) {
            const Real tol = j % 6 >= 3 ? 1e-7 : MeshField::Epsilon;
            if (!(Kokkos::fabs(f[j] - (i == j ? 1.0 : 0.0)) <= tol)) {
              Kokkos::printf("basis %d dof %d: %.3e\n", i, j, f[j]);
              lerr++;
            }
          }
          // physical edge moments of the normal derivative
          for (int e = 0; e < 3; ++e) {
            const int a = refEdge(e, 0), b = refEdge(e, 1);
            const Real T[2] = {t.xy[b][0] - t.xy[a][0],
                               t.xy[b][1] - t.xy[a][1]};
            const Real moment = edgeNormalMoment(e, T, [&](Vector2 xi) {
              return elm.gradientFromCoefficients(c, geo, xi);
            });
            if (!(Kokkos::fabs(moment) <= MeshField::Epsilon)) {
              Kokkos::printf("basis %d edge %d normal moment %.3e\n", i, e,
                             moment);
              lerr++;
            }
          }
        },
        lerrors);
    errors += lerrors;
  }
  report("testPhysicalDuality(" + tri.name + ")", errors);
}

// a quartic polynomial with all terms
struct Quartic {
  KOKKOS_INLINE_FUNCTION Real value(Real x, Real y) const {
    return 1 + x - 2 * y + x * x + 3 * x * y - y * y + x * x * x - x * y * y +
           2 * x * x * y + y * y * y + x * x * x * x - 2 * x * x * x * y +
           x * x * y * y + x * y * y * y - y * y * y * y;
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 2> grad(Real x, Real y) const {
    return {1 + 2 * x + 3 * y + 3 * x * x - y * y + 4 * x * y + 4 * x * x * x -
                6 * x * x * y + 2 * x * y * y + y * y * y,
            -2 + 3 * x - 2 * y - 2 * x * y + 2 * x * x + 3 * y * y -
                2 * x * x * x + 2 * x * x * y + 3 * x * y * y - 4 * y * y * y};
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> hess(Real x, Real y) const {
    return {2 + 6 * x + 4 * y + 12 * x * x - 12 * x * y + 2 * y * y,
            3 - 2 * y + 4 * x - 6 * x * x + 4 * x * y + 3 * y * y,
            -2 - 2 * x + 6 * y + 2 * x * x + 6 * x * y - 12 * y * y};
  }
};

struct Quintic { // x^5
  KOKKOS_INLINE_FUNCTION Real value(Real x, Real) const {
    return x * x * x * x * x;
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 2> grad(Real x, Real) const {
    return {5 * x * x * x * x, 0};
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> hess(Real x, Real) const {
    return {20 * x * x * x, 0, 0};
  }
};

// the largest error in value and gradient at interior points
template <template <typename...> typename Controller, typename Func>
Real maxReproductionError(SingleTriangle<Controller> &st, Func f) {
  st.setDofs(f);
  const auto elm = st.element();
  const TriangleCoords t = st.t;
  Real err = 0;
  Kokkos::parallel_reduce(
      "checkReproduction", Kokkos::RangePolicy<ExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(const int, Real &lerr) {
        const auto c = elm.coefficients(0);
        const auto geo = elm.geometry(0);
        const Real pts[4][2] = {
            {1.0 / 3, 1.0 / 3}, {0.1, 0.2}, {0.6, 0.3}, {0.05, 0.85}};
        for (int p = 0; p < 4; ++p) {
          const Vector2 xi{pts[p][0], pts[p][1]};
          const Real l0 = 1 - xi[0] - xi[1];
          const Real x =
              l0 * t.xy[0][0] + xi[0] * t.xy[1][0] + xi[1] * t.xy[2][0];
          const Real y =
              l0 * t.xy[0][1] + xi[0] * t.xy[1][1] + xi[1] * t.xy[2][1];
          const auto g = elm.gradientFromCoefficients(c, geo, xi);
          const auto gf = f.grad(x, y);
          lerr = Kokkos::max(lerr,
                             Kokkos::fabs(elm.valueFromCoefficients(c, xi)[0] -
                                          f.value(x, y)));
          lerr = Kokkos::max(lerr, Kokkos::fabs(g[0] - gf[0]));
          lerr = Kokkos::max(lerr, Kokkos::fabs(g[1] - gf[1]));
        }
      },
      Kokkos::Max<Real>(err));
  return err;
}

// 3. quartics are reproduced, x^5 is not
template <template <typename...> typename Controller>
void testReproduction(Triangle const &tri) {
  SingleTriangle<Controller> st(tri.coords);
  const Real quarticErr = maxReproductionError(st, Quartic());
  const Real quinticErr = maxReproductionError(st, Quintic());
  const bool pass = quarticErr <= 1e-10 && quinticErr > 1e-6;
  if (!report("testReproduction(" + tri.name + ")", !pass))
    std::cout << "  quartic error " << quarticErr << ", x^5 error "
              << quinticErr << "\n";
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  testReferenceDuality();
  const Real c30 = Kokkos::cos(M_PI / 6), s30 = Kokkos::sin(M_PI / 6);
  const Triangle triangles[] = {
      {{{{0, 0}, {1, 0}, {0, 1}}}, "reference"},
      {{{{1, 2}, {3, 2}, {1, 2.5}}}, "translated and scaled"},
      {{{{0, 0}, {3, 0.5}, {1, 2}}}, "skewed"},
      {{{{0, 0},
         {3 * c30 - 0.5 * s30, 3 * s30 + 0.5 * c30},
         {c30 - 2 * s30, s30 + 2 * c30}}},
       "rotated skewed"},
      {{{{0, 0}, {0, 1}, {1, 0}}}, "clockwise"}};
  t.forEachController([&]<template <typename...> typename Controller>() {
    for (auto const &tri : triangles) {
      testPhysicalDuality<Controller>(tri);
      testReproduction<Controller>(tri);
    }
  });
  return t.exitCode();
}
