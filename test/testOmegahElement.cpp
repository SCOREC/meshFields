#include "KokkosController.hpp"
#include "MeshField.hpp"
#include "MeshField_Element.hpp"    //remove?
#include "MeshField_Fail.hpp"       //remove?
#include "MeshField_For.hpp"        //remove?
#include "MeshField_ShapeField.hpp" //remove?
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

// void checkResult(MeshField::MeshInfo meshInfo) { //FIXME
//
//    MeshField::FieldElement fcoords(meshInfo.numTri, coordField,
//                                    MeshField::LinearTriangleCoordinateShape(),
//                                    LinearTriangleToVertexField(mesh));
//    auto globalCoords = MeshField::evaluate(fcoords, localCoords, offsets);
//
//   // check the result
//   MeshField::LO numErrors = 0;
//   Kokkos::parallel_reduce(
//       "checkResult", meshInfo.numTri,
//       KOKKOS_LAMBDA(const int &ent, MeshField::LO &lerrors) {
//         for (auto pt = offsets(ent); pt < offsets(ent + 1); pt++) {
//           const auto x = globalCoords(pt, 0);
//           const auto y = globalCoords(pt, 1);
//           const auto expected = func(x, y);
//           const auto computed = eval(pt, 0);
//           MeshField::LO isError = 0;
//           if (Kokkos::fabs(computed - expected) >
//           MeshField::MachinePrecision) {
//             isError = 1;
//             Kokkos::printf("result for elm %d, pt %d, does not match:
//             expected "
//                            "%f computed %f\n",
//                            ent, pt, expected, computed);
//           }
//           lerrors += isError;
//         }
//       },
//       numErrors);
//   return (numErrors > 0);
// }

struct TestCoords {
  Kokkos::View<MeshField::Real *[3]> coords;
  size_t NumPtsPerElem;
  std::string name;
};

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
            std::string_view location) {
  std::stringstream ss;
  ss << order << " field evaluation with " << function
     << " analytic function at " << location << " points failed\n";
  std::string msg = ss.str();
  MeshField::fail(msg);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  auto lib = Omega_h::Library(&argc, &argv);
  MeshField::Debug = true;
  {
    auto mesh = createMeshTri18(lib);
    {
      const auto dim = mesh.dim();
      if (dim != 2) {
        MeshField::fail("0.1 input mesh must be 2d, meshdim is %d\n", dim);
      }
    }
    MeshField::OmegahMeshField<ExecutionSpace, MeshField::KokkosController> omf(
        mesh);

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
    // clang-format on

    // clang-format off
    const auto cases = {TestCoords{centroids, OnePtPerElem, "centroids"},
                        TestCoords{interior, OnePtPerElem, "interior"},
                        TestCoords{vertex, OnePtPerElem, "vertex"},
                        TestCoords{allVertices, ThreePtsPerElem, "allVertices"}};
    // clang-format on

    static const size_t LinearField = 1;
    static const size_t QuadraticField = 2;
    for (auto testCase : cases) {
      {
        {
          const auto dim = mesh.dim();
          if (dim != 2) {
            MeshField::fail("0.2 input mesh must be 2d, meshdim is %d\n", dim);
          }
        }
        const auto ShapeOrder = 1;
        const auto MeshDim = 2;
        auto coords = mesh.coords();
        auto field =
            omf.CreateLagrangeField<MeshField::Real, ShapeOrder, MeshDim>();
        auto func = LinearFunction();
        auto setField = KOKKOS_LAMBDA(const int &i) {
          const auto x = coords[i * MeshDim];
          const auto y = coords[i * MeshDim + 1];
          field(0, 0, i, MeshField::Vertex) = func(x, y);
        };
        MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nverts()},
                                setField, "setField");
        using ViewType = decltype(testCase.coords);
        using FieldType = decltype(field);
        auto result =
            omf.triangleLocalPointEval<LinearFunction, ViewType, FieldType>(
                testCase.coords, testCase.NumPtsPerElem, LinearFunction{},
                field);
        // FIXME
        //  auto failed = checkResult(...):
        //  if (failed)
        //    doFail("linear", "linear", testCase.name);
      }

      //      failed = omf.triangleLocalPointEval<QuadraticFunction,
      //      QuadraticField>(
      //          mesh, testCase.coords, testCase.NumPtsPerElem,
      //          QuadraticFunction{});
      //      if (failed)
      //        doFail("quadratic", "quadratic", testCase.name);
      //
      //      failed = omf.triangleLocalPointEval<LinearFunction,
      //      QuadraticField>(
      //          mesh, testCase.coords, testCase.NumPtsPerElem,
      //          LinearFunction{});
      //      if (failed)
      //        doFail("quadratic", "linear", testCase.name);
    }
  }
  Kokkos::finalize();
  return 0;
}
