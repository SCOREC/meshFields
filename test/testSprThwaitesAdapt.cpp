#include <iostream>

#include "Omega_h_adapt.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_class.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_recover.hpp" //project_by_fit
#include "Omega_h_shape.hpp"
#include "Omega_h_timer.hpp"
#include <Omega_h_atomics.hpp> //Omega_h::atomic_fetch_add
#include <Omega_h_dbg.hpp>
#include <Omega_h_file.hpp> //Omega_h::binary
#include <Omega_h_for.hpp>
#include <iomanip> //precision
#include <sstream> //ostringstream
#include <string_view>

#include "MeshField_SPR_ErrorEstimator.hpp"

// detect floating point exceptions
#include <fenv.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

using namespace Omega_h;

void setupFieldTransfer(AdaptOpts &opts) {
  opts.xfer_opts.type_map["solution_1"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["solution_2"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["ice_thickness"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["surface_height"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["bed_topography"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["mu_log"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["prescribed_velocity_1"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["prescribed_velocity_2"] = OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["observed_surface_velocity_rms"] =
      OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["observed_surface_velocity_1"] =
      OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["observed_surface_velocity_2"] =
      OMEGA_H_LINEAR_INTERP;
  opts.xfer_opts.type_map["stiffening_factor_log"] = OMEGA_H_LINEAR_INTERP;
  const int numLayers = 11;
  for (int i = 1; i <= numLayers; i++) {
    std::stringstream ss;
    ss << "temperature_" << std::setfill('0') << std::setw(2) << i;
    opts.xfer_opts.type_map[ss.str()] = OMEGA_H_LINEAR_INTERP;
  }
}

/**
 * retrieve the effective strain rate from the mesh
 *
 * despite the name being 'effective_stress' it is the effective strain:
 * the frobenius norm of  the strain tensor
 */
Reals getEffectiveStrainRate(Mesh &mesh) {
  return mesh.get_array<Real>(2, "effective_stress");
}

Reals recoverLinearStrain(Mesh &mesh, Reals effectiveStrain) {
  return project_by_fit(&mesh, effectiveStrain);
}

Reals getElementCentroids(Mesh &mesh, Reals vtxCoords) {
  assert(mesh.dim() == 2);
  Write<Real> centroids(mesh.dim() * mesh.nelems(), 0,
                        "stores coordinates of cell centroid of each element");

  const auto &faces2nodes = mesh.ask_down(FACE, VERT).ab2b;

  Kokkos::parallel_for(
      "calculate the centroid in each tri element", mesh.nelems(),
      OMEGA_H_LAMBDA(const LO id) {
        const auto current_el_verts = gather_verts<3>(faces2nodes, id);
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            gather_vectors<3, 2>(vtxCoords, current_el_verts);
        auto centroid = average(current_el_vert_coords);
        int index = 2 * id;
        centroids[index] = centroid[0];
        centroids[index + 1] = centroid[1];
      });

  return read(centroids);
}

Reals min_max_normalization_coordinates(const Reals &coordinates, int dim = 2) {
  int num_points = coordinates.size() / dim;

  int coords_size = coordinates.size();

  Write<Real> x_coordinates(num_points, 0, "x coordinates");

  Write<Real> y_coordinates(num_points, 0, "y coordinates");

  parallel_for(
      "separates x and y coordinates", num_points, KOKKOS_LAMBDA(const int id) {
        int index = id * dim;
        x_coordinates[id] = coordinates[index];
        y_coordinates[id] = coordinates[index + 1];
      });

  const auto min_x = Omega_h::get_min(read(x_coordinates));
  const auto min_y = Omega_h::get_min(read(y_coordinates));

  const auto max_x = Omega_h::get_max(read(x_coordinates));
  const auto max_y = Omega_h::get_max(read(y_coordinates));

  const auto del_x = max_x - min_x;
  const auto del_y = max_y - min_y;

  Write<Real> normalized_coordinates(coords_size, 0,
                                     "stores scaled coordinates");

  parallel_for(
      "scale coordinates", num_points, KOKKOS_LAMBDA(const int id) {
        int index = id * dim;
        normalized_coordinates[index] = (x_coordinates[id] - min_x) / del_x;
        normalized_coordinates[index + 1] = (y_coordinates[id] - min_y) / del_y;
      });

  return read(normalized_coordinates);
}

template <typename ShapeField>
void setFieldAtVertices(Omega_h::Mesh &mesh, Reals recoveredStrain,
                        ShapeField field) {
  const auto MeshDim = mesh.dim();
  auto setFieldAtVertices = KOKKOS_LAMBDA(const int &vtx) {
    field(vtx, 0, 0, MeshField::Vertex) = recoveredStrain[vtx];
  };
  MeshField::parallel_for(ExecutionSpace(), {0}, {mesh.nverts()},
                          setFieldAtVertices, "setFieldAtVertices");
}

void printTriCount(Mesh *mesh, std::string_view prefix) {
  const auto nTri = mesh->nglobal_ents(2);
  if (!mesh->comm()->rank())
    std::cout << prefix << " nTri: " << nTri << "\n";
}

int main(int argc, char **argv) {
  feenableexcept(
      FE_ALL_EXCEPT &
      ~FE_INEXACT); // Enable all floating point exceptions but FE_INEXACT
  auto lib = Library(&argc, &argv);
  if (argc != 6) {
    fprintf(stderr,
            "Usage: %s inputMesh.osh outputMeshPrefix adaptRatio min_size "
            "max_size\n",
            argv[0]);
    exit(EXIT_FAILURE);
  }
  auto world = lib.world();
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(argv[1], world, &mesh);
  const auto prefix = std::string(argv[2]);
  // adaptRatio = 0.1 is used in scorec/core:cws/sprThwaites test/spr_test.cc
  const MeshField::Real adaptRatio = std::stof(argv[3]);
  const MeshField::Real min_size =
      std::stof(argv[4]); // the smallest edge length in initial mesh is 1
  const MeshField::Real max_size = std::stof(
      argv[5]); // a value of 8 or 16 roughly maintains the boundary shape
                // in the downstream parts of the domain
                // where there is significant coarsening
  std::cout << "input mesh: " << argv[1] << " outputMeshPrefix: " << prefix
            << " adaptRatio: " << adaptRatio << " max_size: " << max_size
            << " min_size: " << min_size << "\n";
  const auto outname = prefix + "_adaptRatio_" + std::string(argv[3]) +
                       "_maxSz_" + std::to_string(max_size) + "_minSz_" +
                       std::to_string(min_size);

  auto effectiveStrain = getEffectiveStrainRate(mesh);
  auto recoveredStrain = recoverLinearStrain(mesh, effectiveStrain);
  mesh.add_tag<Real>(VERT, "recoveredStrain", 1, recoveredStrain, false,
                     Omega_h::ArrayType::VectorND);

  const auto MeshDim = 2;
  const auto ShapeOrder = 1;
  MeshField::OmegahMeshField<ExecutionSpace, MeshDim,
                             MeshField::KokkosController>
      omf(mesh);
  auto recoveredStrainField =
      omf.CreateLagrangeField<Real, ShapeOrder, MeshDim>();
  setFieldAtVertices(mesh, recoveredStrain, recoveredStrainField);

  auto coordField = omf.getCoordField();
  const auto [shp, map] =
      MeshField::Omegah::getTriangleElement<ShapeOrder>(mesh);
  MeshField::FieldElement coordFe(mesh.nelems(), coordField, shp, map);

  auto estimation = MeshField::SPR::Estimation(
      mesh, effectiveStrain, recoveredStrainField, adaptRatio);

  const auto [tgtLength, error] =
      MeshField::SPR::getSprSizeField(estimation, omf, coordFe);
  std::cout << "Error: " << error << '\n';
  Omega_h::Write<MeshField::Real> tgtLength_oh(tgtLength);
  mesh.add_tag<Real>(VERT, "tgtLength", 1, tgtLength_oh, false,
                     Omega_h::ArrayType::VectorND);

  { // write vtk
    const std::string vtkFileName = "beforeAdapt" + outname + ".vtk";
    Omega_h::vtk::write_parallel(vtkFileName, &mesh, 2);
    const std::string vtkFileName_edges =
        "beforeAdapt" + outname + "_edges.vtk";
    Omega_h::vtk::write_parallel(vtkFileName_edges, &mesh, 1);
  }

  // adapt
  auto opts = Omega_h::AdaptOpts(&mesh);
  setupFieldTransfer(opts);
  printTriCount(&mesh, "beforeAdapt");

  auto verbose = false;
  const auto isos = Omega_h::isos_from_lengths(tgtLength_oh);
  auto metric = clamp_metrics(mesh.nverts(), isos, min_size, max_size);
  Omega_h::grade_fix_adapt(&mesh, opts, metric, verbose);

  printTriCount(&mesh, "afterAdapt");

  const auto sol = mesh.get_array<Omega_h::Real>(VERT, "solution_1");
  const auto sol_min = Omega_h::get_min(sol);
  const auto sol_max = Omega_h::get_max(sol);
  std::cout << "solution_1 min " << sol_min << " max " << sol_max << '\n';

  { // write vtk and osh for adapted mesh
    const std::string outfilename = "afterAdapt" + outname;
    Omega_h::vtk::write_parallel(outfilename + ".vtk", &mesh, 2);
    Omega_h::binary::write(outfilename + ".osh", &mesh);
    std::cout << "wrote adapted mesh: " << outfilename + ".osh"
              << "\n";
  }

  return 0;
}
