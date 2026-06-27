#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"
#include "MeshField_Fail.hpp"
#include "MeshField_For.hpp"
#include "MeshField_ShapeField.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_simplex.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <sstream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

struct LinearFunction {
  KOKKOS_INLINE_FUNCTION
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y) const {
    return 2.0 * x + y;
  }
  // ∂f/∂x = 2.0, ∂f/∂y = 1.0
  // ∂²f/∂x² = 0.0, ∂²f/∂xy = 0.0, ∂²f/∂y² = 0.0
  static constexpr MeshField::Real dfdx = 2.0;
  static constexpr MeshField::Real dfdy = 1.0;
  static constexpr MeshField::Real d2fdx2 = 0.0;
  static constexpr MeshField::Real d2fdxy = 0.0;
  static constexpr MeshField::Real d2fdy2 = 0.0;
};

struct QuadraticFunction {
  KOKKOS_INLINE_FUNCTION
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y) const {
    return (x * x) + (2.0 * y);
  }
  // ∂f/∂x = 2x, ∂f/∂y = 2.0
  // ∂²f/∂x² = 2.0, ∂²f/∂xy = 0.0, ∂²f/∂y² = 0.0
};

Omega_h::Mesh createMeshTri18(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  return Omega_h::build_box(world, family, len, len, 0.0, 3, 3, 0);
}

struct TestCoords {
  Kokkos::View<MeshField::Real *[3]> coords;
  size_t NumPtsPerElem;
  std::string name;
};

template <typename Result, typename CoordField, typename AnalyticFunction>
bool checkResult(Omega_h::Mesh &mesh, Result &result, CoordField coordField,
                 TestCoords testCase, AnalyticFunction func, size_t numComp) {
  // Create explicit local copy to ensure proper capture in KOKKOS_LAMBDA
  const size_t npts_per_elem = testCase.NumPtsPerElem;
  const size_t num_comp = numComp;
  
  MeshField::FieldElement fcoords(
      mesh.nfaces(), coordField, MeshField::LinearTriangleCoordinateShape(),
      MeshField::Omegah::LinearTriangleToVertexField(mesh));
  auto globalCoords =
      MeshField::evaluate(fcoords, testCase.coords, npts_per_elem);

  // Host-side debugging output
  std::cout << "\n=== DEBUG checkResult for " << testCase.name << " ===" << std::endl;
  std::cout << "  numPtsPerElem = " << npts_per_elem << std::endl;
  std::cout << "  numComponents = " << num_comp << std::endl;
  std::cout << "  mesh.nfaces() = " << mesh.nfaces() << std::endl;
  std::cout << "  result.extent(0) = " << result.extent(0) << std::endl;
  std::cout << "  result.extent(1) = " << result.extent(1) << std::endl;
  std::cout << "  globalCoords.extent(0) = " << globalCoords.extent(0) << std::endl;
  std::cout << "  Expected total points = " << mesh.nfaces() * npts_per_elem << std::endl;
  std::cout << "  MachinePrecision = " << MeshField::MachinePrecision << std::endl;

  MeshField::LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkResult", mesh.nfaces(),
      KOKKOS_LAMBDA(const int &ent, MeshField::LO &lerrors) {
        const auto first = ent * npts_per_elem;
        const auto last = first + npts_per_elem;
        
        // Bounds checking
        if (first >= result.extent(0) || last > result.extent(0)) {
          Kokkos::printf("ERROR: ent=%d, first=%d, last=%d exceeds result.extent(0)=%d\n",
                         ent, (int)first, (int)last, (int)result.extent(0));
          lerrors += 1;
          return;
        }
        
        if (first >= globalCoords.extent(0) || last > globalCoords.extent(0)) {
          Kokkos::printf("ERROR: ent=%d, first=%d, last=%d exceeds globalCoords.extent(0)=%d\n",
                         ent, (int)first, (int)last, (int)globalCoords.extent(0));
          lerrors += 1;
          return;
        }
        
        for (auto pt = first; pt < last; pt++) {
          const auto x = globalCoords(pt, 0);
          const auto y = globalCoords(pt, 1);
          const auto expected = func(x, y);
          
          // Debug output for first few elements
          if (ent < 3 && pt == first) {
            Kokkos::printf("Element %d: pt=%d, x=%.6f, y=%.6f, expected=%.6f, result(pt,0)=%.6f\n",
                           ent, (int)pt, x, y, expected, result(pt, 0));
          }
          
          for (size_t i = 0; i < num_comp; ++i) {
            const auto computed = result(pt, i);
            const auto diff = Kokkos::fabs(computed - expected);
            MeshField::LO isError = 0;
            if (diff > 10 * MeshField::MachinePrecision) {
              isError = 1;
              Kokkos::printf(
                  "result for elm %d, pt %d (local %d), comp %d: expected "
                  "%.15f (x=%.6f, y=%.6f) computed %.15f, diff=%.15e, tol=%.15e\n",
                  ent, (int)pt, (int)(pt - first), (int)i, expected, x, y, computed, diff, MeshField::MachinePrecision);
            }
            lerrors += isError;
          }
        }
      },
      numErrors);
  
  std::cout << "  Total errors found: " << numErrors << std::endl;
  return (numErrors > 0);
}

template <typename AnalyticFunction, typename ShapeField>
void setReducedQuinticDOFs(Omega_h::Mesh &mesh, AnalyticFunction func,
                           ShapeField field) {
  const auto MeshDim = mesh.dim();
  auto coords = mesh.coords();
  auto setFieldAtVertices = KOKKOS_LAMBDA(const int &vtx) {
    const auto x = coords[vtx * MeshDim];
    const auto y = coords[vtx * MeshDim + 1];
    // value: f(x,y)
    field(vtx, 0, 0, MeshField::Vertex) = func(x, y);
    // first derivatives
    if constexpr (std::is_same_v<AnalyticFunction, LinearFunction>) {
      field(vtx, 0, 1, MeshField::Vertex) = AnalyticFunction::dfdx;
      field(vtx, 0, 2, MeshField::Vertex) = AnalyticFunction::dfdy;
      field(vtx, 0, 3, MeshField::Vertex) = AnalyticFunction::d2fdx2;
      field(vtx, 0, 4, MeshField::Vertex) = AnalyticFunction::d2fdxy;
      field(vtx, 0, 5, MeshField::Vertex) = AnalyticFunction::d2fdy2;
    } else {
      // QuadraticFunction: f(x,y) = x^2 + 2y
      // ∂f/∂x = 2x, ∂f/∂y = 2
      // ∂²f/∂x² = 2, ∂²f/∂xy = 0, ∂²f/∂y² = 0
      field(vtx, 0, 1, MeshField::Vertex) = 2.0 * x;
      field(vtx, 0, 2, MeshField::Vertex) = 2.0;
      field(vtx, 0, 3, MeshField::Vertex) = 2.0;
      field(vtx, 0, 4, MeshField::Vertex) = 0.0;
      field(vtx, 0, 5, MeshField::Vertex) = 0.0;
    }
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nverts()},
                          setFieldAtVertices, "setReducedQuinticDOFs");
}

template <size_t NumPtsPerElem>
Kokkos::View<MeshField::Real *[3]>
createElmAreaCoords(size_t numElements,
                    Kokkos::Array<MeshField::Real, 3 * NumPtsPerElem> coords) {
  Kokkos::View<MeshField::Real *[3]> lc("localCoords",
                                        numElements * NumPtsPerElem);
  Kokkos::parallel_for(
      "setLocalCoords", numElements, KOKKOS_LAMBDA(const int &elm) {
        for (size_t pt = 0; pt < NumPtsPerElem; pt++) {
          lc(elm * NumPtsPerElem + pt, 0) = coords[pt * 3 + 0];
          lc(elm * NumPtsPerElem + pt, 1) = coords[pt * 3 + 1];
          lc(elm * NumPtsPerElem + pt, 2) = coords[pt * 3 + 2];
        }
      });
  return lc;
}

void doFail(std::string_view order, std::string_view function,
            std::string_view location, std::string_view numComp) {
  std::stringstream ss;
  ss << order << " field evaluation with " << numComp << " components and "
     << function << " analytic function at " << location << " points failed\n";
  std::string msg = ss.str();
  MeshField::fail(msg);
}

template <size_t numComponents, template <typename...> typename Controller,
          typename TestCaseType, typename FunctionType>
void runTest(Omega_h::Mesh &mesh,
             MeshField::OmegahMeshField<ExecutionSpace, 2, Controller> &omf,
             TestCaseType testCase, FunctionType function) {
  using functionType = decltype(function);
  using ViewType = decltype(testCase.coords);
  auto field = omf.template CreateLagrangeField<MeshField::Real, 1, 6>();
  using FieldType = decltype(field);
  setReducedQuinticDOFs(mesh, function, field);
  // verify field
  Kokkos::parallel_for(
      "printField", mesh.nverts(),
      KOKKOS_LAMBDA(const int &vtx) {
        Kokkos::printf("vtx %d: value %f, dfdx %f, dfdy %f, d2fdx2 %f, d2fdxy %f, "
               "d2fdy2 %f\n",
               vtx, field(vtx, 0, 0, MeshField::Vertex),
               field(vtx, 0, 1, MeshField::Vertex),
               field(vtx, 0, 2, MeshField::Vertex),
               field(vtx, 0, 3, MeshField::Vertex),
               field(vtx, 0, 4, MeshField::Vertex),
               field(vtx, 0, 5, MeshField::Vertex));
      });
  auto result = omf.template triangleReducedQuinticEval<ViewType, FieldType>(
      testCase.coords, testCase.NumPtsPerElem, field);
  auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                            decltype(function){}, numComponents);
  if (failed) {
    std::string functionErr;
    if constexpr (std::is_same_v<functionType, LinearFunction>) {
      functionErr = "linear";
    } else {
      functionErr = "quadratic";
    }
    doFail("reducedQuintic", functionErr, testCase.name, std::to_string(numComponents));
  }
}

template <template <typename...> typename Controller>
void doRun(Omega_h::Mesh &mesh,
           MeshField::OmegahMeshField<ExecutionSpace, 2, Controller> &omf) {

  // setup field with values from the analytic function
  static const size_t OnePtPerElem = 1;
  static const size_t ThreePtsPerElem = 3;
  auto centroids = createElmAreaCoords<OnePtPerElem>(
      mesh.nfaces(), {1 / 3.0, 1 / 3.0, 1 / 3.0});
  auto interior =
      createElmAreaCoords<OnePtPerElem>(mesh.nfaces(), {0.1, 0.4, 0.5});
  auto vertex =
      createElmAreaCoords<OnePtPerElem>(mesh.nfaces(), {0.0, 0.0, 1.0});
  // clang-format off
    auto allVertices = createElmAreaCoords<ThreePtsPerElem>(mesh.nfaces(),
        {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0});
    const auto cases = {TestCoords{centroids, OnePtPerElem, "centroids"},
                        TestCoords{interior, OnePtPerElem, "interior"},
                        TestCoords{vertex, OnePtPerElem, "vertex"},
                        TestCoords{allVertices, ThreePtsPerElem, "allVertices"}};
  // clang-format on

  for (auto testCase : cases) {
    using ViewType = decltype(testCase.coords);
    {
      const auto numComponents = 1;
      runTest<numComponents>(mesh, omf, testCase, LinearFunction());
    }

    {
      const auto numComponents = 1;
      runTest<numComponents>(mesh, omf, testCase, QuadraticFunction());
    }

    {
      const auto numComponents = 2;
      runTest<numComponents>(mesh, omf, testCase, LinearFunction());
    }

    {
      const auto numComponents = 3;
      runTest<numComponents>(mesh, omf, testCase, LinearFunction());
    }

    {
      const auto numComponents = 2;
      runTest<numComponents>(mesh, omf, testCase, QuadraticFunction());
    }

    {
      const auto numComponents = 3;
      runTest<numComponents>(mesh, omf, testCase, QuadraticFunction());
    }
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  MeshField::Debug = true;
#ifdef MESHFIELDS_ENABLE_CABANA
  {
    auto mesh = createMeshTri18(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::CabanaController>
        omf(mesh);
    doRun<MeshField::CabanaController>(mesh, omf);
  }
#endif
  {
    auto mesh = createMeshTri18(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 2, MeshField::KokkosController>
        omf(mesh);
    doRun<MeshField::KokkosController>(mesh, omf);
  }
  Kokkos::finalize();
  return 0;
}