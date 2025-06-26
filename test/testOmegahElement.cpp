#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"    //remove?
#include "MeshField_Fail.hpp"       //remove?
#include "MeshField_For.hpp"        //remove?
#include "MeshField_ShapeField.hpp" //remove?
#ifdef MESHFIELDS_ENABLE_CABANA
#include "CabanaController.hpp"
#endif
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
};

struct QuadraticFunction {
  KOKKOS_INLINE_FUNCTION
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y) const {
    return (x * x) + (2.0 * y);
  }
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
  const auto numPtsPerElem = testCase.NumPtsPerElem;
  MeshField::FieldElement fcoords(
      mesh.nfaces(), coordField, MeshField::LinearTriangleCoordinateShape(),
      MeshField::Omegah::LinearTriangleToVertexField(mesh));
  auto globalCoords =
      MeshField::evaluate(fcoords, testCase.coords, numPtsPerElem);

  MeshField::LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkResult", mesh.nfaces(),
      KOKKOS_LAMBDA(const int &ent, MeshField::LO &lerrors) {
        const auto first = ent * numPtsPerElem;
        const auto last = first + numPtsPerElem;
        for (auto pt = first; pt < last; pt++) {
          const auto x = globalCoords(pt, 0);
          const auto y = globalCoords(pt, 1);
          const auto expected = func(x, y);
          for (int i = 0; i < numComp; ++i) {
            const auto computed = result(pt, i);
            MeshField::LO isError = 0;
            if (Kokkos::fabs(computed - expected) >
                MeshField::MachinePrecision) {
              isError = 1;
              Kokkos::printf(
                  "result for elm %d, pt %d, does not match: expected "
                  "%f computed %f\n",
                  ent, pt, expected, computed);
            }
            lerrors += isError;
          }
        }
      },
      numErrors);
  return (numErrors > 0);
}

template <typename AnalyticFunction, typename ShapeField>
void setVertices(Omega_h::Mesh &mesh, AnalyticFunction func, ShapeField field) {
  const auto MeshDim = mesh.dim();
  auto coords = mesh.coords();
  auto setFieldAtVertices = KOKKOS_LAMBDA(const int &vtx) {
    // get dofholder position at the midpoint of edge
    // - TODO should be encoded in the field?
    const auto x = coords[vtx * MeshDim];
    const auto y = coords[vtx * MeshDim + 1];
    for (int i = 0; i < field.numComp; ++i) {
      field(vtx, 0, i, MeshField::Vertex) = func(x, y);
    }
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nverts()},
                          setFieldAtVertices, "setFieldAtVertices");
}

template <typename AnalyticFunction, typename ShapeField>
void setEdges(Omega_h::Mesh &mesh, AnalyticFunction func, ShapeField field) {
  const auto MeshDim = mesh.dim();
  const auto edgeDim = 1;
  const auto vtxDim = 0;
  const auto edge2vtx = mesh.ask_down(edgeDim, vtxDim).ab2b;
  auto coords = mesh.coords();
  auto setFieldAtEdges = KOKKOS_LAMBDA(const int &edge) {
    // get dofholder position at the midpoint of edge
    // - TODO should be encoded in the field?
    const auto left = edge2vtx[edge * 2];
    const auto right = edge2vtx[edge * 2 + 1];
    const auto x = (coords[left * MeshDim] + coords[right * MeshDim]) / 2.0;
    const auto y =
        (coords[left * MeshDim + 1] + coords[right * MeshDim + 1]) / 2.0;
    for (int i = 0; i < field.numComp; ++i) {
      field(edge, 0, 0, MeshField::Edge) = func(x, y);
    }
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nedges()},
                          setFieldAtEdges, "setFieldAtEdges");
}

template <size_t NumPtsPerElem>
Kokkos::View<MeshField::Real *[3]>
createElmAreaCoords(size_t numElements,
                    Kokkos::Array<MeshField::Real, 3 * NumPtsPerElem> coords) {
  Kokkos::View<MeshField::Real *[3]> lc("localCoords",
                                        numElements * NumPtsPerElem);
  Kokkos::parallel_for(
      "setLocalCoords", numElements, KOKKOS_LAMBDA(const int &elm) {
        for (int pt = 0; pt < NumPtsPerElem; pt++) {
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

  auto coords = mesh.coords();
  for (auto testCase : cases) {
    using ViewType = decltype(testCase.coords);
    {
      const auto ShapeOrder = 1;
      const auto numComponents = 1;
      auto field = omf.template CreateLagrangeField<MeshField::Real, ShapeOrder,
                                                    numComponents>();
      LinearFunction func = LinearFunction();
      setVertices(mesh, func, field);
      using FieldType = decltype(field);
      auto result = omf.template triangleLocalPointEval<ViewType, FieldType>(
          testCase.coords, testCase.NumPtsPerElem, field);
      auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                                LinearFunction{}, numComponents);
      if (failed)
        doFail("linear", "linear", testCase.name,
               std::to_string(numComponents));
    }

    {
      const auto ShapeOrder = 2;
      const auto numComponents = 1;
      auto field = omf.template CreateLagrangeField<MeshField::Real, ShapeOrder,
                                                    numComponents>();
      auto func = QuadraticFunction();
      setVertices(mesh, func, field);
      setEdges(mesh, func, field);
      using FieldType = decltype(field);
      auto result = omf.template triangleLocalPointEval<ViewType, FieldType>(
          testCase.coords, testCase.NumPtsPerElem, field);
      auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                                QuadraticFunction{}, numComponents);
      if (failed)
        doFail("quadratic", "quadratic", testCase.name,
               std::to_string(numComponents));
    }

    {
      const auto ShapeOrder = 2;
      const auto numComponents = 1;
      auto field = omf.template CreateLagrangeField<MeshField::Real, ShapeOrder,
                                                    numComponents>();
      auto func = LinearFunction();
      setVertices(mesh, func, field);
      setEdges(mesh, func, field);
      using FieldType = decltype(field);
      auto result = omf.template triangleLocalPointEval<ViewType, FieldType>(
          testCase.coords, testCase.NumPtsPerElem, field);
      auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                                LinearFunction{}, numComponents);
      if (failed)
        doFail("quadratic", "linear", testCase.name,
               std::to_string(numComponents));
    }

    {
      const auto ShapeOrder = 1;
      const auto numComponents = 2;
      auto field = omf.template CreateLagrangeField<MeshField::Real, ShapeOrder,
                                                    numComponents>();
      LinearFunction func = LinearFunction();
      setVertices(mesh, func, field);
      using FieldType = decltype(field);
      auto result = omf.template triangleLocalPointEval<ViewType, FieldType>(
          testCase.coords, testCase.NumPtsPerElem, field);
      auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                                LinearFunction{}, numComponents);
      if (failed)
        doFail("linear", "linear", testCase.name,
               std::to_string(numComponents));
    }
    {
      const auto ShapeOrder = 1;
      const auto numComponents = 3;
      auto field = omf.template CreateLagrangeField<MeshField::Real, ShapeOrder,
                                                    numComponents>();
      LinearFunction func = LinearFunction();
      setVertices(mesh, func, field);
      using FieldType = decltype(field);
      auto result = omf.template triangleLocalPointEval<ViewType, FieldType>(
          testCase.coords, testCase.NumPtsPerElem, field);
      auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                                LinearFunction{}, numComponents);
      if (failed)
        doFail("linear", "linear", testCase.name,
               std::to_string(numComponents));
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
