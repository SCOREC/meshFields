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
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y,
                             MeshField::Real z) const {
    return 2.0 * x + y + z;
  }
};

struct QuadraticFunction {
  KOKKOS_INLINE_FUNCTION
  MeshField::Real operator()(MeshField::Real x, MeshField::Real y,
                             MeshField::Real z) const {
    return (x * x) + (2.0 * y) + z;
  }
};

Omega_h::Mesh createMeshTet(Omega_h::Library &lib) {
  auto world = lib.world();
  const auto family = OMEGA_H_SIMPLEX;
  auto len = 1.0;
  return Omega_h::build_box(world, family, len, len, len, 3, 3, 3);
}

struct TestCoords {
  Kokkos::View<MeshField::Real *[4]> coords;
  size_t NumPtsPerElem;
  std::string name;
};

template <typename Result, typename CoordField, typename AnalyticFunction>
bool checkResult(Omega_h::Mesh &mesh, Result &result, CoordField coordField,
                 TestCoords testCase, AnalyticFunction func, size_t numComp) {
  const auto numPtsPerElem = testCase.NumPtsPerElem;
  MeshField::FieldElement fcoords(
      mesh.nregions(), coordField, MeshField::LinearTetrahedronShape(),
      MeshField::Omegah::LinearTetrahedronToVertexField(mesh));
  auto globalCoords =
      MeshField::evaluate(fcoords, testCase.coords, numPtsPerElem);

  MeshField::LO numErrors = 0;
  Kokkos::parallel_reduce(
      "checkResult", mesh.nregions(),
      KOKKOS_LAMBDA(const int &ent, MeshField::LO &lerrors) {
        const auto first = ent * numPtsPerElem;
        const auto last = first + numPtsPerElem;
        for (auto pt = first; pt < last; pt++) {
          const auto x = globalCoords(pt, 0);
          const auto y = globalCoords(pt, 1);
          const auto z = globalCoords(pt, 2);
          const auto expected = func(x, y, z);
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
    const auto z = coords[vtx * MeshDim + 2];
    for (int i = 0; i < field.numComp; ++i) {
      field(vtx, 0, i, MeshField::Vertex) = func(x, y, z);
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
    const auto z =
        (coords[left * MeshDim + 2] + coords[right * MeshDim + 2]) / 2.0;
    for (int i = 0; i < field.numComp; ++i) {
      field(edge, 0, i, MeshField::Edge) = func(x, y, z);
    }
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nedges()},
                          setFieldAtEdges, "setFieldAtEdges");
}

template <size_t NumPtsPerElem>
Kokkos::View<MeshField::Real *[4]>
createElmAreaCoords(size_t numElements,
                    Kokkos::Array<MeshField::Real, 4 * NumPtsPerElem> coords) {
  Kokkos::View<MeshField::Real *[4]> lc("localCoords",
                                        numElements * NumPtsPerElem);
  Kokkos::parallel_for(
      "setLocalCoords", numElements, KOKKOS_LAMBDA(const int &elm) {
        for (int pt = 0; pt < NumPtsPerElem; pt++) {
          lc(elm * NumPtsPerElem + pt, 0) = coords[pt * 4 + 0];
          lc(elm * NumPtsPerElem + pt, 1) = coords[pt * 4 + 1];
          lc(elm * NumPtsPerElem + pt, 2) = coords[pt * 4 + 2];
          lc(elm * NumPtsPerElem + pt, 3) = coords[pt * 4 + 3];
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
template <size_t ShapeOrder, size_t numComponents,
          template <typename...> typename Controller>
void runTest(Omega_h::Mesh &mesh,
             MeshField::OmegahMeshField<ExecutionSpace, 3, Controller> &omf,
             auto testCase, auto function) {
  using functionType = decltype(function);
  using ViewType = decltype(testCase.coords);
  auto field = omf.template CreateLagrangeField<MeshField::Real, ShapeOrder,
                                                numComponents>();
  using FieldType = decltype(field);
  setVertices(mesh, function, field);
  if constexpr (ShapeOrder == 2) {
    setEdges(mesh, function, field);
  }
  auto result = omf.template tetrahedronLocalPointEval<ViewType, FieldType>(
      testCase.coords, testCase.NumPtsPerElem, field);
  auto failed = checkResult(mesh, result, omf.getCoordField(), testCase,
                            decltype(function){}, numComponents);
  if (failed) {
    std::string fieldErr = ShapeOrder == 1 ? "linear" : "quadratic";
    std::string functionErr;
    if constexpr (std::is_same_v<functionType, LinearFunction>) {
      functionErr = "linear";
    } else {
      functionErr = "quadratic";
    }
    doFail(fieldErr, functionErr, testCase.name, std::to_string(numComponents));
  }
}

template <template <typename...> typename Controller>
void doRun(Omega_h::Mesh &mesh,
           MeshField::OmegahMeshField<ExecutionSpace, 3, Controller> &omf) {

  // setup field with values from the analytic function
  static const size_t OnePtPerElem = 1;
  static const size_t FourPtsPerElem = 4;
  auto centroids = createElmAreaCoords<OnePtPerElem>(
      mesh.nregions(), {1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0});
  auto interior =
      createElmAreaCoords<OnePtPerElem>(mesh.nregions(), {0.1, 0.4, 0.3, 0.2});
  auto vertex =
      createElmAreaCoords<OnePtPerElem>(mesh.nregions(), {0.0, 0.0, 1.0, 0.0});
  // clang-format off
    auto allVertices = createElmAreaCoords<FourPtsPerElem>(mesh.nregions(),
        {1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0});
    const auto cases = {TestCoords{centroids, OnePtPerElem, "centroids"},
                        TestCoords{interior, OnePtPerElem, "interior"},
                        TestCoords{vertex, OnePtPerElem, "vertex"},
                        TestCoords{allVertices, FourPtsPerElem, "allVertices"}};
  // clang-format on

  auto coords = mesh.coords();
  for (auto testCase : cases) {
    using ViewType = decltype(testCase.coords);
    {
      const auto ShapeOrder = 1;
      const auto numComponents = 1;
      runTest<ShapeOrder, numComponents>(mesh, omf, testCase, LinearFunction());
    }

    {
      const auto ShapeOrder = 2;
      const auto numComponents = 1;
      runTest<ShapeOrder, numComponents>(mesh, omf, testCase,
                                         QuadraticFunction());
    }

    {
      const auto ShapeOrder = 2;
      const auto numComponents = 1;
      runTest<ShapeOrder, numComponents>(mesh, omf, testCase, LinearFunction());
    }
    {
      const auto ShapeOrder = 1;
      const auto numComponents = 2;
      runTest<ShapeOrder, numComponents>(mesh, omf, testCase, LinearFunction());
    }
    {
      const auto ShapeOrder = 1;
      const auto numComponents = 3;
      runTest<ShapeOrder, numComponents>(mesh, omf, testCase, LinearFunction());
    }
    {
      const auto ShapeOrder = 2;
      const auto numComponents = 2;
      runTest<ShapeOrder, numComponents>(mesh, omf, testCase,
                                         QuadraticFunction());
    }

    {
      const auto ShapeOrder = 2;
      const auto numComponents = 3;
      runTest<ShapeOrder, numComponents>(mesh, omf, testCase, LinearFunction());
    }
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  MeshField::Debug = true;
#ifdef MESHFIELDS_ENABLE_CABANA
  {
    auto mesh = createMeshTet(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 3, MeshField::CabanaController>
        omf(mesh);
    doRun<MeshField::CabanaController>(mesh, omf);
  }
#endif
  {
    auto mesh = createMeshTet(lib);
    MeshField::OmegahMeshField<ExecutionSpace, 3, MeshField::KokkosController>
        omf(mesh);
    doRun<MeshField::KokkosController>(mesh, omf);
  }
  Kokkos::finalize();
  return 0;
}
