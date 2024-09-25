#ifndef MESHFIELD_SHAPEFIELD_HPP
#define MESHFIELD_SHAPEFIELD_HPP

#include "MeshField.hpp"
#include "MeshField_Shape.hpp"
#include "MeshField_logging.hpp"
#include "KokkosController.hpp"

namespace MeshField {

struct MeshInfo {
   int numVtx;    // entDim = 0
   int numEdge;   // entDim = 1
   int numTri;    // entDim = 2
   int numQuad;   // entDim = 2
   int numTet;    // entDim = 3
   int numHex;    // entDim = 3
   int numPrism;  // entDim = 3
   int numPyramid;// entDim = 3
};

template <typename MeshFieldType, typename Shape>
struct ShapeField {
  MeshFieldType meshField;
  Shape shape;
  MeshInfo meshInfo;
  ShapeField(MeshFieldType& meshFieldIn, Shape shapeIn, MeshInfo meshInfoIn) :
    meshField(meshFieldIn),
    shape(shapeIn),
    meshInfo(meshInfoIn) {};
};
//prototype that provides access operator to underlying fields
//https://godbolt.org/z/3c8bzrPca
//https://godbolt.org/z/7xK9cEsx7

template <typename ExecutionSpace, size_t order>
auto CreateLagrangeField(MeshInfo& meshInfo) { //assumes 2d or 3d mesh, do we want to support 1d meshes?
  static_assert((order == 1 || order == 2),
    "CreateLagrangeField only supports linear and quadratic fields\n");
  using MemorySpace = typename ExecutionSpace::memory_space;
  if constexpr (order == 1) {
    assert(meshInfo.numVtx > 0);
    using Ctrlr =
        Controller::KokkosController<MemorySpace, ExecutionSpace, double ***>;
    Ctrlr kk_ctrl({/*field 0*/ 1, 1, meshInfo.numVtx}); //1 dof with 1 component per vtx
    MeshField<Ctrlr> kokkosMeshField(kk_ctrl);
    return ShapeField(kokkosMeshField, LinearTriangleShape(), meshInfo);
  } else if constexpr (order == 2) {
    assert(meshInfo.numVtx > 0);
    assert(meshInfo.numEdge > 0);
    using Ctrlr =
        Controller::KokkosController<MemorySpace, ExecutionSpace, double ***, double ***>;
    Ctrlr kk_ctrl({/*field 0*/ 1, 1, meshInfo.numVtx,   //1 dof with 1 component per vtx
                   /*field 1*/ 1, 1, meshInfo.numEdge}); //1 dof with 1 component per edge
    MeshField<Ctrlr> kokkosMeshField(kk_ctrl);
    return ShapeField(kokkosMeshField, QuadraticTriangleShape(), meshInfo);
  } else {
    return nullptr; //silence compiler warning
  }
};

} // namespace MeshField

#endif

