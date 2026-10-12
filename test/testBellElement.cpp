#include "MeshField_Bell.hpp"
#include "MeshField_MappedElement.hpp"
#include "MeshField_TestHarness.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// Tests for the Bell element.
//
// 1. reference duality: the 21 functionals of the extended Bell reference
//    element (vertex values, gradients, and Hessians, and the edge moments of
//    the normal derivative against the degree 4 Legendre polynomial) applied
//    to the 21 reference basis functions give the identity.
// 2. physical duality, on single triangles of several shapes: setting stored
//    dof i to 1 and the rest to 0 gives the i-th physical basis function,
//    whose physical dofs (vertex values, gradients, and Hessians) are e_i and
//    whose normal derivative is cubic on every edge (its physical edge
//    moments vanish).
// 3. reproduction: the Bell space contains the quartics, so a quartic with
//    exact dofs is reproduced in value, gradient, and Hessian; x^5 is not.
// 4. C1 continuity on an Omega_h mesh with warped vertex coordinates and
//    random dofs: along every interior edge, the value and gradient from the
//    two adjacent triangles agree.
// 5. convergence of interpolation on refined warped meshes: for
//    f = sin(pi x) sin(2 pi y) with exact vertex dofs, the maximum error at
//    interior points converges at rate 5 for values, 4 for gradients, and 3
//    for Hessians.
// 6. IdentityMap::hessian maps the packed Hessians of two components,
//    indexed [entry * numComp + component], to H_x = J^{-1} H_xi J^{-T}.
// 7. evaluate() on the 3x3 box mesh, as in PR #99's test: 2x+y and x^2+2y
//    with exact dofs are reproduced at the centroid, an interior point, a
//    vertex, and all vertices of every triangle, for fields with 1-3
//    components (component c holds (c+1)*f).

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

// TODO: replace this with Edge integration in MeshFields
// 5 point Gauss-Legendre rule on [-1, 1]; exact to degree 9
constexpr int numGaussPts = 5;
KOKKOS_INLINE_FUNCTION Real gaussPt(int q) {
  constexpr Real pts[numGaussPts] = {-0.9061798459386640, -0.5384693101056831,
                                     0.0, 0.5384693101056831,
                                     0.9061798459386640};
  return pts[q];
}
KOKKOS_INLINE_FUNCTION Real gaussWt(int q) {
  constexpr Real wts[numGaussPts] = {0.2369268850561891, 0.4786286704993665,
                                     0.5688888888888889, 0.4786286704993665,
                                     0.2369268850561891};
  return wts[q];
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
  const int a = refEdge(e, 0), b = refEdge(e, 1);
  const Real len = Kokkos::sqrt(T[0] * T[0] + T[1] * T[1]);
  const Real n[2] = {T[1] / len, -T[0] / len};
  Real moment = 0;
  for (int q = 0; q < numGaussPts; ++q) {
    const Real u = gaussPt(q);
    const Real s = (u + 1) / 2;
    const auto g =
        gradAt(Vector2{refVert(a, 0) + s * (refVert(b, 0) - refVert(a, 0)),
                       refVert(a, 1) + s * (refVert(b, 1) - refVert(a, 1))});
    moment += gaussWt(q) / 2 * legendre4(u) * (n[0] * g[0] + n[1] * g[1]);
  }
  return moment;
}

// the functionals applied to basis function k: 18 vertex functionals, then 3
// edge moments
KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 21> applyFunctionals(int k) {
  const MeshField::BellReferenceShape shape;
  Kokkos::Array<Real, 21> f;
  for (int v = 0; v < 3; ++v) {
    const Vector2 x{refVert(v, 0), refVert(v, 1)};
    const auto grad = shape.getLocalGradients(x);
    f[v * 6] = shape.getValues(x)[k];
    f[v * 6 + 1] = grad[k * 2];
    f[v * 6 + 2] = grad[k * 2 + 1];
    const auto hess = shape.getLocalHessians(x);
    f[v * 6 + 3] = hess[k * 3];
    f[v * 6 + 4] = hess[k * 3 + 1];
    f[v * 6 + 5] = hess[k * 3 + 2];
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
          const Real tol = MeshField::Epsilon;
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
            const auto H = elm.hessianFromCoefficients(c, geo, xi);
            f[v * 6] = elm.valueFromCoefficients(c, xi)[0];
            f[v * 6 + 1] = g[0];
            f[v * 6 + 2] = g[1];
            f[v * 6 + 3] = H[0];
            f[v * 6 + 4] = H[1];
            f[v * 6 + 5] = H[2];
          }
          for (int j = 0; j < 18; ++j) {
            const Real tol = MeshField::Epsilon;
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

// the largest error in value, gradient, and Hessian at interior points
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
          const auto H = elm.hessianFromCoefficients(c, geo, xi);
          const auto gf = f.grad(x, y);
          const auto Hf = f.hess(x, y);
          for (int k = 0; k < 3; ++k)
            lerr = Kokkos::max(lerr, Kokkos::fabs(H[k] - Hf[k]));
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

// ---------------------------------------------------------------------------
// Omega_h meshes

// modify the coordinates so the triangles have different shapes
struct Warp {
  KOKKOS_INLINE_FUNCTION Vector2 operator()(Vector2 x) const {
    return {x[0] + 0.05 * Kokkos::sin(2 * M_PI * x[1]),
            x[1] + 0.04 * Kokkos::sin(2 * M_PI * x[0])};
  }
};

// 4. value and gradient agree across interior edges
template <template <typename...> typename Controller>
void testContinuity(Omega_h::Library &lib) {
  auto mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 4, 4, 0);
  MeshField::OmegahMeshField<ExecutionSpace, 2, Controller> omf(mesh);
  auto cfWc = omf.template CreateLagrangeCoordinateField<1>(Warp());
  auto bellWc = omf.template CreateBellField<1>();
  auto bf = bellWc.field;
  Kokkos::parallel_for(
      "randomDofs", mesh.nverts(), KOKKOS_LAMBDA(const int v) {
        for (int k = 0; k < 6; ++k) {
          uint64_t h = (uint64_t(v) * 6 + k + 1) * 0x9E3779B97F4A7C15ULL;
          h ^= h >> 29;
          bf(v, k, 0, MeshField::Vertex) = double(h % 2000) / 1000 - 1;
        }
      });
  const auto elm = omf.CreateBellElement(
      bellWc.field, omf.CreateLagrangeElement(cfWc.field));
  const MeshField::Omegah::LinearTriangleToVertexField triVerts(mesh);
  const auto edgeVerts = mesh.ask_verts_of(Omega_h::EDGE);
  const auto edgeFaces = mesh.ask_up(Omega_h::EDGE, Omega_h::FACE);
  const auto e2f = edgeFaces.a2ab;
  const auto faces = edgeFaces.ab2b;
  LO numInterior = 0;
  LO errors = 0;
  Kokkos::parallel_reduce(
      "checkContinuity", mesh.nedges(),
      KOKKOS_LAMBDA(const int e, LO &lerr, LO &linterior) {
        if (e2f[e + 1] - e2f[e] != 2)
          return;
        linterior++;
        const int va = edgeVerts[e * 2], vb = edgeVerts[e * 2 + 1];
        for (int q = 1; q <= 3; ++q) {
          const Real s = 0.25 * q;
          Real v[2];
          Real g[2][2];
          for (int side = 0; side < 2; ++side) {
            const int f = faces[e2f[e] + side];
            // barycentric coordinates of the point in the element's ordering
            Real L[3] = {0, 0, 0};
            for (int k = 0; k < 3; ++k) {
              const auto vtx = triVerts(k, 0, f, MeshField::Triangle).entity;
              if (vtx == va)
                L[k] = 1 - s;
              if (vtx == vb)
                L[k] = s;
            }
            const Vector2 xi{L[1], L[2]};
            const auto c = elm.coefficients(f);
            const auto geo = elm.geometry(f);
            const auto grad = elm.gradientFromCoefficients(c, geo, xi);
            v[side] = elm.valueFromCoefficients(c, xi)[0];
            g[side][0] = grad[0];
            g[side][1] = grad[1];
          }
          if (!(Kokkos::fabs(v[0] - v[1]) <= MeshField::Epsilon) ||
              !(Kokkos::fabs(g[0][0] - g[1][0]) <= 1e-10) ||
              !(Kokkos::fabs(g[0][1] - g[1][1]) <= 1e-10)) {
            Kokkos::printf("edge %d s %.2f: value %.3e grad %.3e %.3e\n", e, s,
                           v[0] - v[1], g[0][0] - g[1][0], g[0][1] - g[1][1]);
            lerr++;
          }
        }
      },
      errors, numInterior);
  // a mesh without interior edges would check nothing
  report("testContinuity(" + std::to_string(numInterior) + " interior edges)",
         errors + (numInterior == 0));
}

struct SinSin { // sin(pi x) sin(2 pi y)
  KOKKOS_INLINE_FUNCTION Real value(Real x, Real y) const {
    return Kokkos::sin(M_PI * x) * Kokkos::sin(2 * M_PI * y);
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 2> grad(Real x, Real y) const {
    return {M_PI * Kokkos::cos(M_PI * x) * Kokkos::sin(2 * M_PI * y),
            2 * M_PI * Kokkos::sin(M_PI * x) * Kokkos::cos(2 * M_PI * y)};
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> hess(Real x, Real y) const {
    const Real sx = Kokkos::sin(M_PI * x), cx = Kokkos::cos(M_PI * x);
    const Real sy = Kokkos::sin(2 * M_PI * y), cy = Kokkos::cos(2 * M_PI * y);
    return {-M_PI * M_PI * sx * sy, 2 * M_PI * M_PI * cx * cy,
            -4 * M_PI * M_PI * sx * sy};
  }
};

// maximum interpolation errors in value, gradient, and Hessian on an n x n
// mesh
template <template <typename...> typename Controller>
Kokkos::Array<Real, 3> interpolationError(Omega_h::Library &lib, int n) {
  auto mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, n, n, 0);
  MeshField::OmegahMeshField<ExecutionSpace, 2, Controller> omf(mesh);
  auto cfWc = omf.template CreateLagrangeCoordinateField<1>(Warp());
  auto cf = cfWc.field;
  auto bellWc = omf.template CreateBellField<1>();
  auto bf = bellWc.field;
  const SinSin f;
  Kokkos::parallel_for(
      "interpolate", mesh.nverts(), KOKKOS_LAMBDA(const int v) {
        const Real x = cf(v, 0, 0, MeshField::Vertex);
        const Real y = cf(v, 0, 1, MeshField::Vertex);
        const auto g = f.grad(x, y);
        const auto h = f.hess(x, y);
        bf(v, 0, 0, MeshField::Vertex) = f.value(x, y);
        bf(v, 1, 0, MeshField::Vertex) = g[0];
        bf(v, 2, 0, MeshField::Vertex) = g[1];
        bf(v, 3, 0, MeshField::Vertex) = h[0];
        bf(v, 4, 0, MeshField::Vertex) = h[1];
        bf(v, 5, 0, MeshField::Vertex) = h[2];
      });
  const auto geomElm = omf.CreateLagrangeElement(cf);
  const auto elm = omf.CreateBellElement(bellWc.field, geomElm);
  Real valueErr = 0, gradErr = 0, hessErr = 0;
  Kokkos::parallel_reduce(
      "interpolationError", mesh.nfaces(),
      KOKKOS_LAMBDA(const int ent, Real &lv, Real &lg, Real &lh) {
        const auto c = elm.coefficients(ent);
        const auto geo = elm.geometry(ent);
        const auto cx = geomElm.coefficients(ent);
        // Duffy-mapped tensor Gauss points in the triangle
        for (int i = 0; i < numGaussPts; ++i) {
          for (int j = 0; j < numGaussPts; ++j) {
            const Real u = (gaussPt(i) + 1) / 2;
            const Real w = (gaussPt(j) + 1) / 2;
            const Vector2 xi{u, w * (1 - u)};
            const auto x = geomElm.valueFromCoefficients(cx, xi);
            const auto g = elm.gradientFromCoefficients(c, geo, xi);
            const auto gf = f.grad(x[0], x[1]);
            lv = Kokkos::max(lv,
                             Kokkos::fabs(elm.valueFromCoefficients(c, xi)[0] -
                                          f.value(x[0], x[1])));
            lg = Kokkos::max(lg, Kokkos::sqrt((g[0] - gf[0]) * (g[0] - gf[0]) +
                                              (g[1] - gf[1]) * (g[1] - gf[1])));
            const auto H = elm.hessianFromCoefficients(c, geo, xi);
            const auto Hf = f.hess(x[0], x[1]);
            for (int k = 0; k < 3; ++k)
              lh = Kokkos::max(lh, Kokkos::fabs(H[k] - Hf[k]));
          }
        }
      },
      Kokkos::Max<Real>(valueErr), Kokkos::Max<Real>(gradErr),
      Kokkos::Max<Real>(hessErr));
  return {valueErr, gradErr, hessErr};
}

// 5. interpolation converges at rates 5 (values), 4 (gradients), and 3
// (Hessians)
template <template <typename...> typename Controller>
void testConvergence(Omega_h::Library &lib) {
  std::vector<Kokkos::Array<Real, 3>> err;
  const int sizes[] = {4, 8, 16, 32};
  for (int n : sizes)
    err.push_back(interpolationError<Controller>(lib, n));
  Real valueRate = 0, gradRate = 0, hessRate = 0;
  for (size_t i = 1; i < err.size(); ++i) {
    valueRate = std::log2(err[i - 1][0] / err[i][0]);
    gradRate = std::log2(err[i - 1][1] / err[i][1]);
    hessRate = std::log2(err[i - 1][2] / err[i][2]);
    std::cout << "  n=" << sizes[i] << ": value error " << err[i][0]
              << " (rate " << valueRate << "), gradient error " << err[i][1]
              << " (rate " << gradRate << "), Hessian error " << err[i][2]
              << " (rate " << hessRate << ")\n";
  }
  report("testConvergence",
         !(valueRate >= 4.7 && gradRate >= 3.7 && hessRate >= 2.7));
}

struct Linear2xy { // 2x + y
  KOKKOS_INLINE_FUNCTION Real value(Real x, Real y) const { return 2 * x + y; }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 2> grad(Real, Real) const {
    return {2, 1};
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> hess(Real, Real) const {
    return {0, 0, 0};
  }
};

struct QuadraticX2y { // x^2 + 2y
  KOKKOS_INLINE_FUNCTION Real value(Real x, Real y) const {
    return x * x + 2 * y;
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 2> grad(Real x, Real) const {
    return {2 * x, 2};
  }
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> hess(Real, Real) const {
    return {2, 0, 0};
  }
};

// 7. evaluate() reproduces f at every point of each point set
template <template <typename...> typename Controller, size_t numComp,
          typename Func>
void testMeshEvaluate(Omega_h::Mesh &mesh, Func f, std::string const &name) {
  MeshField::OmegahMeshField<ExecutionSpace, 2, Controller> omf(mesh);
  auto cfWc = omf.template CreateLagrangeCoordinateField<1>();
  auto bellWc = omf.template CreateBellField<numComp>();
  auto cf = cfWc.field;
  auto bf = bellWc.field;
  Kokkos::parallel_for(
      "setDofs", mesh.nverts(), KOKKOS_LAMBDA(const int v) {
        const Real x = cf(v, 0, 0, MeshField::Vertex);
        const Real y = cf(v, 0, 1, MeshField::Vertex);
        const auto g = f.grad(x, y);
        const auto h = f.hess(x, y);
        for (size_t c = 0; c < numComp; ++c) {
          const Real s = c + 1;
          bf(v, 0, c, MeshField::Vertex) = s * f.value(x, y);
          bf(v, 1, c, MeshField::Vertex) = s * g[0];
          bf(v, 2, c, MeshField::Vertex) = s * g[1];
          bf(v, 3, c, MeshField::Vertex) = s * h[0];
          bf(v, 4, c, MeshField::Vertex) = s * h[1];
          bf(v, 5, c, MeshField::Vertex) = s * h[2];
        }
      });
  auto geomElm = omf.CreateLagrangeElement(cf);
  auto elm = omf.CreateBellElement(bf, geomElm);

  struct PointSet {
    std::vector<Vector2> xi;
    std::string name;
  };
  const std::vector<PointSet> pointSets = {
      {{{1.0 / 3, 1.0 / 3}}, "centroid"},
      {{{0.1, 0.4}}, "interior"},
      {{{0, 0}}, "vertex"},
      {{{0, 0}, {1, 0}, {0, 1}}, "all vertices"}};
  const size_t numElms = mesh.nfaces();
  for (auto const &ps : pointSets) {
    const size_t numPts = ps.xi.size();
    Kokkos::View<Real **> lc("localCoords", numElms * numPts, 2);
    auto lc_h = Kokkos::create_mirror_view(lc);
    for (size_t e = 0; e < numElms; ++e)
      for (size_t p = 0; p < numPts; ++p)
        for (int d = 0; d < 2; ++d)
          lc_h(e * numPts + p, d) = ps.xi[p][d];
    Kokkos::deep_copy(lc, lc_h);
    const auto vals = MeshField::evaluate(elm, lc, numPts);
    // positions from the coordinate element directly, independent of
    // evaluate(), so an error in evaluate() cannot cancel out
    LO errors = 0;
    Kokkos::parallel_reduce(
        "checkMeshEvaluate", numElms,
        KOKKOS_LAMBDA(const int e, LO &lerr) {
          const auto cx = geomElm.coefficients(e);
          for (size_t p = 0; p < numPts; ++p) {
            const int i = e * numPts + p;
            const auto x =
                geomElm.valueFromCoefficients(cx, Vector2{lc(i, 0), lc(i, 1)});
            const Real expected = f.value(x[0], x[1]);
            for (size_t c = 0; c < numComp; ++c) {
              if (!(Kokkos::fabs(vals(i, c) - (c + 1) * expected) <=
                    MeshField::Epsilon)) {
                Kokkos::printf("point %d component %zu: %.15f expected %.15f\n",
                               i, c, vals(i, c), (c + 1) * expected);
                lerr++;
              }
            }
          }
        },
        errors);
    report("testMeshEvaluate(" + name + ", " + std::to_string(numComp) +
               " components, " + ps.name + ")",
           errors);
  }
}

// 6. physical Hessians of two components from J H_x J^T
void testHessianMap() {
  constexpr int numComp = 2;
  const Kokkos::Array<Real, 4> J = {2.0, 0.5, -0.3, 1.5};
  // physical Hessians (xx, xy, yy) of each component
  const Real Hx[numComp][3] = {{1.0, -2.0, 3.0}, {-4.0, 0.5, 2.5}};
  // H_xi = J H_x J^T, packed entry-major
  Kokkos::Array<Real, 3 * numComp> refHess;
  for (int c = 0; c < numComp; ++c) {
    const Real H[2][2] = {{Hx[c][0], Hx[c][1]}, {Hx[c][1], Hx[c][2]}};
    Real JHJt[2][2];
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 2; ++k) {
        JHJt[i][k] = 0;
        for (int j = 0; j < 2; ++j)
          for (int l = 0; l < 2; ++l)
            JHJt[i][k] += J[i * 2 + j] * H[j][l] * J[k * 2 + l];
      }
    refHess[0 * numComp + c] = JHJt[0][0];
    refHess[1 * numComp + c] = JHJt[0][1];
    refHess[2 * numComp + c] = JHJt[1][1];
  }
  const auto h = MeshField::IdentityMap{}.hessian(refHess, J);
  LO errors = 0;
  for (int c = 0; c < numComp; ++c)
    for (int e = 0; e < 3; ++e)
      errors += !(std::fabs(h[e * numComp + c] - Hx[c][e]) <= 1e-13);
  report("testHessianMap", errors);
}

int main(int argc, char **argv) {
  TestRunner t(argc, argv);
  testReferenceDuality();
  testHessianMap();
  const Real c30 = Kokkos::cos(M_PI / 6), s30 = Kokkos::sin(M_PI / 6);
  const Triangle triangles[] = {
      {{{{0, 0}, {1, 0}, {0, 1}}}, "reference"},
      {{{{1, 2}, {3, 2}, {1, 2.5}}}, "translated and scaled"},
      {{{{0, 0}, {3, 0.5}, {1, 2}}}, "skewed"},
      {{{{0, 0},
         {3 * c30 - 0.5 * s30, 3 * s30 + 0.5 * c30},
         {c30 - 2 * s30, s30 + 2 * c30}}},
       "rotated skewed"},
      {{{{0, 0}, {0, 1}, {1, 0}}}, "clockwise"},
      // the two triangles of PR #99's single triangle tests
      {{{{0, 0}, {4, 0}, {0, 3}}}, "PR #99 right"},
      {{{{1, 1}, {5, 1}, {2, 4}}}, "PR #99 general"}};
  t.forEachController([&]<template <typename...> typename Controller>() {
    for (auto const &tri : triangles) {
      testPhysicalDuality<Controller>(tri);
      testReproduction<Controller>(tri);
    }
    testContinuity<Controller>(t.library());
    testConvergence<Controller>(t.library());
    t.forEachMesh([&]<size_t dim>(Omega_h::Mesh &mesh) {
      if constexpr (dim == 2) {
        testMeshEvaluate<Controller, 1>(mesh, Linear2xy(), "2x+y");
        testMeshEvaluate<Controller, 1>(mesh, QuadraticX2y(), "x^2+2y");
        testMeshEvaluate<Controller, 2>(mesh, Linear2xy(), "2x+y");
        testMeshEvaluate<Controller, 3>(mesh, QuadraticX2y(), "x^2+2y");
      }
    });
  });
  return t.exitCode();
}
