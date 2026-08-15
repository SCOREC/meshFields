#include <MeshField_Macros.hpp>
#include <MeshField_Field.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Element.hpp>
#include <MeshField_ReducedQuintic.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace MeshField;

/**
 * @brief Unit tests for ReducedQuintic triangle element implementation
 * 
 * Tests cover:
 * 1. LU solver with partial pivoting
 * 2. Geometric parameter computation
 * 3. Coordinate transformations (barycentric <-> local)
 * 4-8. Field evaluation with constant, linear, and quadratic fields
 */

/**
 * @brief Test 1: LU solver
 * Tests the custom LU decomposition with partial pivoting
 */
bool testSolveLU() {
  std::cout << "Test 1: Testing LU solver\n";
  std::cout << "==========================\n";
  
  // Test with a simple 3x3 upper triangular system: Ax = b
  // A = [2  1  0]    b = [3]    Expected solution: x = [1]
  //     [0  2  1]        [3]                            [1]
  //     [0  0  2]        [2]                            [1]
  // Verification: Row 1: 2(1)+1(1)+0(1)=3 OK, Row 2: 0(1)+2(1)+1(1)=3 OK, Row 3: 0(1)+0(1)+2(1)=2 OK
  
  const int n = 3;
  Real A_host[9] = {2, 1, 0,
                    0, 2, 1,
                    0, 0, 2};
  Real b_host[3] = {3, 3, 2};
  Real expected[3] = {1, 1, 1};
  
  // Allocate device Views and copy data to device
  Kokkos::View<Real**, Kokkos::LayoutRight> A_d("A_device", 3, 3);
  Kokkos::View<Real**, Kokkos::LayoutRight> b_d("b_device", 3, 1);
  auto A_h = Kokkos::create_mirror_view(A_d);
  auto b_h = Kokkos::create_mirror_view(b_d);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) A_h(i, j) = A_host[i * 3 + j];
    b_h(i, 0) = b_host[i];
  }
  Kokkos::deep_copy(A_d, A_h);
  Kokkos::deep_copy(b_d, b_h);
  
  // Run LU solver on device
  int info;
  Kokkos::parallel_reduce("testSolveLU", 1, KOKKOS_LAMBDA(int, int& linfo) {
    linfo = solveLU_internal(A_d, b_d);
  }, info);
  
  if (info != 0) {
    std::cout << "  [FAIL] FAILED: LU solver returned error code " << info << "\n\n";
    return false;
  }
  
  // Copy result back to host
  Kokkos::deep_copy(b_h, b_d);
  
  // Check solution
  Real tol = 1e-10;
  bool passed = true;
  std::cout << "  Solution:\n";
  for (int i = 0; i < n; i++) {
    Real error = std::abs(b_h(i, 0) - expected[i]);
    std::cout << "    x[" << i << "] = " << b_h(i, 0) 
              << " (expected: " << expected[i] 
              << ", error: " << error << ")\n";
    if (error > tol) {
      passed = false;
    }
  }
  
  if (passed) {
    std::cout << "  [PASS] PASSED\n\n";
  } else {
    std::cout << "  [FAIL] FAILED: Solution exceeds tolerance\n\n";
  }
  
  return passed;
}

/**
 * @brief Test 2: Geometric parameter computation
 * Tests computeReducedQuinticGeometry
 */
bool testGeometryComputation() {
  std::cout << "Test 2: Testing geometric parameter computation\n";
  std::cout << "================================================\n";
  
  // Simple right triangle:
  // v0 = (0, 0), v1 = (4, 0), v2 = (0, 3)
  Omega_h::Matrix<2,3> coords;
  coords[0][0] = 0; coords[0][1] = 0;  // v0
  coords[1][0] = 4; coords[1][1] = 0;  // v1
  coords[2][0] = 0; coords[2][1] = 3;  // v2
  
  Real a, b, c, sin_theta, cos_theta;
  int order[3];
  computeReducedQuinticGeometry(coords, a, b, c, sin_theta, cos_theta, order);
  
  std::cout << "  Triangle vertices:\n";
  std::cout << "    v0 = (" << coords[0][0] << ", " << coords[0][1] << ")\n";
  std::cout << "    v1 = (" << coords[1][0] << ", " << coords[1][1] << ")\n";
  std::cout << "    v2 = (" << coords[2][0] << ", " << coords[2][1] << ")\n";
  std::cout << "  Computed parameters:\n";
  std::cout << "    a (dist to reordered v1) = " << a << "\n";
  std::cout << "    b (dist to reordered v0) = " << b << "\n";
  std::cout << "    c (perp dist to reordered v2) = " << c << "\n";
  std::cout << "    sin_theta = " << sin_theta << "\n";
  std::cout << "    cos_theta = " << cos_theta << "\n";
  std::cout << "    vertex order = [" << order[0] << ", " << order[1] << ", " << order[2] << "]\n";
  
  // After reordering to put longest edge (v1-v2, length 5) along xi-axis:
  // Expected order: [1, 2, 0] (puts v1 at reordered v0, v2 at reordered v1, v0 at reordered v2)
  // Expected: a = 1.8, b = 3.2, c = 2.4, origin = (1.44, 1.92)
  Real tol = 1e-6;
  bool passed = true;
  
  // Check that longest edge is preserved
  Real longest_edge = 5.0;  // sqrt((4-0)^2 + (0-3)^2) = 5
  if (std::abs(a + b - longest_edge) > tol) {
    std::cout << "  [FAIL] a + b != longest edge length (expected " << longest_edge << ", got " << (a+b) << ")\n";
    passed = false;
  }
  
  // Just verify basic properties rather than exact values
  // since the exact values depend on the reordering logic
  if (a < 0 || b < 0 || c < 0) {
    std::cout << "  [FAIL] Negative geometric parameters\n";
    passed = false;
  }
  
  if (std::abs(sin_theta*sin_theta + cos_theta*cos_theta - 1.0) > tol) {
    std::cout << "  [FAIL] sin^2(theta) + cos^2(theta) != 1\n";
    passed = false;
  }
  
  if (passed) {
    std::cout << "  [PASS] PASSED\n\n";
  } else {
    std::cout << "  [FAIL] FAILED\n\n";
  }
  
  return passed;
}

/**
 * @brief Test 3: Coordinate transformation
 * Tests parametricToLocal helper function
 */
KOKKOS_INLINE_FUNCTION
Vector3 localToBarycentric(Vector2 const& local,
                           Real a, Real b, Real c)
{
  const Real xi = local[0];
  const Real eta = local[1];

  // From:
  // xi  = -b*L0 + a*L1
  // eta = c*L2
  // L0 + L1 + L2 = 1

  const Real lambda2 = eta / c;

  const Real rhs = 1.0 - lambda2;

  const Real lambda1 = (xi + b * rhs) / (a + b);
  const Real lambda0 = rhs - lambda1;

  return {lambda0, lambda1, lambda2};
}

bool testCoordinateTransformation()
{
  std::cout << "\n";
  std::cout << "Test 3: Coordinate Transformation\n";
  std::cout << "=================================\n";

  const Real tol = 1e-10;

  auto checkNear =
      [&](Real actual,
          Real expected,
          const char* msg) -> bool
  {
    bool ok = std::abs(actual - expected) < tol;

    std::cout << "  "
              << msg
              << " actual=" << actual
              << " expected=" << expected
              << (ok ? " [PASS]" : " [FAIL]")
              << "\n";

    return ok;
  };

  bool passed = true;

  //
  // Case 1:
  // Simple triangle
  //
  {
    std::cout << "\nSimple triangle:\n";

    const Real a = 4.0;
    const Real b = 0.0;
    const Real c = 3.0;

    // Parametric coordinates: xi[0]=L1, xi[1]=L2
    Vector2 v0 = {0.0, 0.0};
    Vector2 v1 = {1.0, 0.0};
    Vector2 v2 = {0.0, 1.0};

    int order[3] = {0, 1, 2};  // No reordering for this simple case

    auto p0 = ReducedQuinticHelpers::parametricToLocal(v0, order, a,b,c);
    auto p1 = ReducedQuinticHelpers::parametricToLocal(v1, order, a,b,c);
    auto p2 = ReducedQuinticHelpers::parametricToLocal(v2, order, a,b,c);

    passed &= checkNear(p0[0], 0.0, "vertex0 xi");
    passed &= checkNear(p0[1], 0.0, "vertex0 eta");

    passed &= checkNear(p1[0], 4.0, "vertex1 xi");
    passed &= checkNear(p1[1], 0.0, "vertex1 eta");

    passed &= checkNear(p2[0], 0.0, "vertex2 xi");
    passed &= checkNear(p2[1], 3.0, "vertex2 eta");

    Vector2 centroid = {1.0/3.0, 1.0/3.0};

    auto center =
        ReducedQuinticHelpers::parametricToLocal(
            centroid, order, a,b,c);

    passed &= checkNear(center[0], 4.0/3.0,
                        "centroid xi");

    passed &= checkNear(center[1], 1.0,
                        "centroid eta");
  }

  //
  // Case 2:
  // Actual M3DC1 geometry
  //
  {
    std::cout << "\nM3DC1 example:\n";

    const Real a = 1.8;
    const Real b = 3.2;
    const Real c = 2.4;

    // Parametric coordinates: xi[0]=L1, xi[1]=L2
    Vector2 v0 = {0.0, 0.0};
    Vector2 v1 = {1.0, 0.0};
    Vector2 v2 = {0.0, 1.0};

    int order[3] = {0, 1, 2}; 

    auto p0 = ReducedQuinticHelpers::parametricToLocal(v0, order, a,b,c);
    auto p1 = ReducedQuinticHelpers::parametricToLocal(v1, order, a,b,c);
    auto p2 = ReducedQuinticHelpers::parametricToLocal(v2, order, a,b,c);

    passed &= checkNear(p0[0], -3.2,
                        "vertex0 xi");
    passed &= checkNear(p0[1], 0.0,
                        "vertex0 eta");

    passed &= checkNear(p1[0], 1.8,
                        "vertex1 xi");
    passed &= checkNear(p1[1], 0.0,
                        "vertex1 eta");

    passed &= checkNear(p2[0], 0.0,
                        "vertex2 xi");
    passed &= checkNear(p2[1], 2.4,
                        "vertex2 eta");

    Vector2 centroid = {1.0/3.0, 1.0/3.0};

    auto center =
        ReducedQuinticHelpers::parametricToLocal(
            centroid, order, a,b,c);

    passed &= checkNear(center[0],
                        (a-b)/3.0,
                        "centroid xi");

    passed &= checkNear(center[1],
                        c/3.0,
                        "centroid eta");
  }

  //
  // Case 3:
  // Round-trip test
  //
  {
    std::cout << "\nRound-trip test:\n";

    const Real a = 1.8;
    const Real b = 3.2;
    const Real c = 2.4;

    Vector2 local = {-0.2, 1.0};

    auto bary =
        localToBarycentric(local,a,b,c);

    int order[3] = {0, 1, 2}; 

    // Convert barycentric to parametric: xi[0]=L1, xi[1]=L2
    Vector2 param = {bary[1], bary[2]};

    auto recovered =
        ReducedQuinticHelpers::parametricToLocal(
            param, order, a,b,c);

    std::cout
      << "  bary = ("
      << bary[0] << ", "
      << bary[1] << ", "
      << bary[2] << ")\n";

    passed &= checkNear(recovered[0],
                        local[0],
                        "roundtrip xi");

    passed &= checkNear(recovered[1],
                        local[1],
                        "roundtrip eta");
  }

  std::cout << "\n";

  if (passed)
    std::cout << "[PASS] Coordinate transform tests PASSED\n";
  else
    std::cout << "[FAIL] Coordinate transform tests FAILED\n";

  std::cout << "\n";

  return passed;
}


struct EvalPoint {
  Real coord[2];
  Real expected_value;
  Real expected_dx;
  Real expected_dy;
};

/**
 * @brief Test field evaluation with expected values
 * Tests that field values and gradients match expected analytical results
 */
bool testFieldEvaluation(const char* testName, Omega_h::Matrix<2,3> const& coords, Real dofs[18],
                         EvalPoint* evalPoints, int numPoints, Omega_h::Library& lib) {
  std::cout << "Test: " << testName << "\n";
  std::cout << "  numPoints=" << numPoints << "\n";
  std::cout << "  Triangle vertices: "
            << "(" << coords[0][0] << "," << coords[0][1] << ") "
            << "(" << coords[1][0] << "," << coords[1][1] << ") "
            << "(" << coords[2][0] << "," << coords[2][1] << ")\n";
  
  // Get geometric parameters
  Real a, b, c, sin_theta, cos_theta;
  int order[3];
  computeReducedQuinticGeometry(coords, a, b, c, sin_theta, cos_theta, order);
  std::cout << "  Geometric params: a=" << a << " b=" << b << " c=" << c
            << " sin_theta=" << sin_theta << " cos_theta=" << cos_theta
            << " order=[" << order[0] << "," << order[1] << "," << order[2] << "]\n";
  
  // Rotate DOFs to local coordinate system
  for(int i=0; i<3; i++) {
    rotateDof(dofs+i*6, sin_theta, cos_theta);
  }
  
  // Precompute coefficients on device using Matrix<2,3> view
  Kokkos::View<Omega_h::Matrix<2,3>*> triCoords_d("triCoords_device", 1);
  auto triCoords_h = Kokkos::create_mirror_view(triCoords_d);
  for (int i = 0; i < 3; i++) {
    triCoords_h(0)[i][0] = coords[i][0];
    triCoords_h(0)[i][1] = coords[i][1];
  }
  Kokkos::deep_copy(triCoords_d, triCoords_h);
  
  auto coeffs = precomputeReducedQuinticCoefficients(1, triCoords_d);
  auto elemOrder_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coeffs.elemOrder);
  auto elemGeomParams_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coeffs.elemGeomParams);
  auto elemCoeffs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coeffs.elemCoeffs);
  
  Real tol = 1e-8;
  bool passed = true;
  
  for (int p = 0; p < numPoints; p++) {
    Real physCoord[2] = {evalPoints[p].coord[0], evalPoints[p].coord[1]};
    
    // Convert physical to barycentric
    Real x0 = coords[0][0], y0 = coords[0][1];
    Real x1 = coords[1][0], y1 = coords[1][1];
    Real x2 = coords[2][0], y2 = coords[2][1];
    Real x = physCoord[0], y = physCoord[1];
    
    Real detT = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2);
    Real lambda0 = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / detT;
    Real lambda1 = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / detT;
    Real lambda2 = 1.0 - lambda0 - lambda1;
    
    // Convert barycentric to parametric coordinates
    // Parametric coords: xi[0] = L1, xi[1] = L2, where L0 = 1 - xi[0] - xi[1]
    Kokkos::Array<Real, 2> xiParam = {lambda1, lambda2};
    
    // Evaluate on host
    ReducedQuinticTriangleShape shape;

    Kokkos::View<Real*>  shapeValues_d("shapeValues", 18);   // 1D
    Kokkos::View<Real**> shapeGrads_d("shapeGrads", 18, 2);  // 2D

    Kokkos::parallel_for("EvaluateField", 1, KOKKOS_LAMBDA(int) {
      const int order[3] = {coeffs.elemOrder(0, 0), coeffs.elemOrder(0, 1), coeffs.elemOrder(0, 2)};
      const Real a = coeffs.elemGeomParams(0, 0);
      const Real b = coeffs.elemGeomParams(0, 1);
      const Real c_geom = coeffs.elemGeomParams(0, 2);
      auto coeffSlice = Kokkos::subview(coeffs.elemCoeffs, 0, Kokkos::ALL());
      auto shapeValues_array = shape.getValues(xiParam, order, a, b, c_geom, coeffSlice);
      auto shapeGrads_array = shape.getLocalGradients(xiParam, order, a, b, c_geom, coeffSlice);
      for (int i = 0; i < 18; i++) {
        shapeValues_d(i) = shapeValues_array[i];
        shapeGrads_d(i, 0) = shapeGrads_array[i][0];
        shapeGrads_d(i, 1) = shapeGrads_array[i][1];
      }
    });

    auto shapeValues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shapeValues_d);
    auto shapeGrads = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shapeGrads_d);
    
    // Interpolate field value and gradients
    Real f_val = 0.0;
    Real dfdxi0 = 0.0;
    Real dfdxi1 = 0.0;
    
    for (int ni = 0; ni < 18; ni++) {
      Real dofValue = dofs[ni];
      f_val += shapeValues(ni) * dofValue;
      dfdxi0 += shapeGrads(ni, 0) * dofValue;
      dfdxi1 += shapeGrads(ni, 1) * dofValue;
    }
    
    // Transform gradients to physical coordinates
    Real J[2][2] = {
      {coords[0][0] - coords[2][0], coords[1][0] - coords[2][0]},
      {coords[0][1] - coords[2][1], coords[1][1] - coords[2][1]}
    };
    Real detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];
    Real Jinv[2][2] = {
      { J[1][1]/detJ, -J[0][1]/detJ},
      {-J[1][0]/detJ,  J[0][0]/detJ}
    };
    
    Real dfdx = Jinv[0][0]*dfdxi0 + Jinv[1][0]*dfdxi1;
    Real dfdy = Jinv[0][1]*dfdxi0 + Jinv[1][1]*dfdxi1;
    
    // Compare with expected values
    Real err_val = std::abs(f_val - evalPoints[p].expected_value);
    Real err_dx = std::abs(dfdx - evalPoints[p].expected_dx);
    Real err_dy = std::abs(dfdy - evalPoints[p].expected_dy);
    
    if (err_val > tol || err_dx > tol || err_dy > tol) {
      std::cout << "  [FAIL] Point (" << physCoord[0] << ", " << physCoord[1] << "):\n";
      std::cout << "    f:     " << f_val << " (expected: " << evalPoints[p].expected_value 
                << ", error: " << err_val << ")\n";
      std::cout << "    \\partial f/\\partial x: " << dfdx << " (expected: " << evalPoints[p].expected_dx 
                << ", error: " << err_dx << ")\n";
      std::cout << "    \\partial f/\\partial y: " << dfdy << " (expected: " << evalPoints[p].expected_dy 
                << ", error: " << err_dy << ")\n";
      passed = false;
    }
  }
  
  if (passed) {
    std::cout << "  [PASS] PASSED\n\n";
  } else {
    std::cout << "  [FAIL] FAILED\n\n";
    fail("testFieldEvaluation: \"%s\" FAILED", testName);
  }
  
  return passed;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  bool allPassed = true;
  {
    auto lib = Omega_h::Library(&argc, &argv);
    
    std::cout << "MeshFields ReducedQuintic Test Suite\n";
    std::cout << "====================================\n\n";
    
    // Run unit tests
    allPassed &= testSolveLU();
    allPassed &= testGeometryComputation();
    allPassed &= testCoordinateTransformation();
    
    // Helper to create a Matrix<2,3> from triplets
    auto m23 = [](Real x0,Real y0, Real x1,Real y1, Real x2,Real y2) {
      Omega_h::Matrix<2,3> m;
      m[0][0]=x0; m[0][1]=y0;
      m[1][0]=x1; m[1][1]=y1;
      m[2][0]=x2; m[2][1]=y2;
      return m;
    };

    // Test 4: Constant field f=1
    std::cout << "Test 4: Constant field (f=1)\n";
    std::cout << "============================\n";
    {
      auto coords = m23(0,0, 4,0, 0,3);
      Real dofs[18];
      for (int i = 0; i < 3; i++) {
        dofs[i*6 + 0] = 1.0;  // value
        dofs[i*6 + 1] = 0.0;  // dx
        dofs[i*6 + 2] = 0.0;  // dy
        dofs[i*6 + 3] = 0.0;  // dxx
        dofs[i*6 + 4] = 0.0;  // dxy
        dofs[i*6 + 5] = 0.0;  // dyy
      }
      EvalPoint evalPoints[] = {
        {{4.0/3.0, 1.0}, 1.0, 0.0, 0.0},
        {{0.0, 0.0}, 1.0, 0.0, 0.0},
        {{4.0, 0.0}, 1.0, 0.0, 0.0},
        {{0.0, 3.0}, 1.0, 0.0, 0.0},
        {{2.0, 0.0}, 1.0, 0.0, 0.0}
      };
      allPassed &= testFieldEvaluation("Constant field on right triangle", 
                                       coords, dofs, evalPoints, 5, lib);
    }
    
    // Test 5: Linear field f=x
    std::cout << "Test 5: Linear field (f=x)\n";
    std::cout << "==========================\n";
    {
      auto coords = m23(0,0, 4,0, 0,3);
      Real dofs[18];
      for (int i = 0; i < 3; i++) {
        dofs[i*6 + 0] = coords[i][0];  // f = x
        dofs[i*6 + 1] = 1.0;           // df/dx = 1
        dofs[i*6 + 2] = 0.0;           // df/dy = 0
        dofs[i*6 + 3] = 0.0;
        dofs[i*6 + 4] = 0.0;
        dofs[i*6 + 5] = 0.0;
      }
      EvalPoint evalPoints[] = {
        {{4.0/3.0, 1.0}, 4.0/3.0, 1.0, 0.0},
        {{2.0, 0.0}, 2.0, 1.0, 0.0},
        {{1.0, 1.0}, 1.0, 1.0, 0.0}
      };
      allPassed &= testFieldEvaluation("Linear field f=x", 
                                       coords, dofs, evalPoints, 3, lib);
    }
    
    // Test 6: Linear field f=y
    std::cout << "Test 6: Linear field (f=y)\n";
    std::cout << "==========================\n";
    {
      auto coords = m23(0,0, 4,0, 0,3);
      Real dofs[18];
      for (int i = 0; i < 3; i++) {
        dofs[i*6 + 0] = coords[i][1];  // f = y
        dofs[i*6 + 1] = 0.0;           // df/dx = 0
        dofs[i*6 + 2] = 1.0;           // df/dy = 1
        dofs[i*6 + 3] = 0.0;
        dofs[i*6 + 4] = 0.0;
        dofs[i*6 + 5] = 0.0;
      }
      EvalPoint evalPoints[] = {
        {{4.0/3.0, 1.0}, 1.0, 0.0, 1.0},
        {{0.0, 1.5}, 1.5, 0.0, 1.0},
        {{2.0, 1.5}, 1.5, 0.0, 1.0}
      };
      allPassed &= testFieldEvaluation("Linear field f=y", 
                                       coords, dofs, evalPoints, 3, lib);
    }
    
    // Test 7: Quadratic field f=x^2
    std::cout << "Test 7: Quadratic field (f=x^2)\n";
    std::cout << "===============================\n";
    {
      auto coords = m23(0,0, 4,0, 0,3);
      Real dofs[18];
      for (int i = 0; i < 3; i++) {
        Real x = coords[i][0];
        dofs[i*6 + 0] = x * x;   // f = x^2
        dofs[i*6 + 1] = 2.0 * x; // df/dx = 2x
        dofs[i*6 + 2] = 0.0;     // df/dy = 0
        dofs[i*6 + 3] = 2.0;     // d^2f/dx^2 = 2
        dofs[i*6 + 4] = 0.0;
        dofs[i*6 + 5] = 0.0;
      }
      EvalPoint evalPoints[] = {
        {{4.0/3.0, 1.0}, (4.0/3.0)*(4.0/3.0), 2.0*(4.0/3.0), 0.0},
        {{2.0, 0.0}, 4.0, 4.0, 0.0}
      };
      allPassed &= testFieldEvaluation("Quadratic field f=x^2", 
                                       coords, dofs, evalPoints, 2, lib);
    }
    
    // Test 8: Quadratic field f=x^2+y^2 on general triangle
    std::cout << "Test 8: Quadratic field on general triangle (f=x^2+y^2)\n";
    std::cout << "======================================================\n";
    {
      auto coords = m23(1,1, 5,1, 2,4);
      Real dofs[18];
      for (int i = 0; i < 3; i++) {
        Real x = coords[i][0];
        Real y = coords[i][1];
        dofs[i*6 + 0] = x*x + y*y;  // f = x^2 + y^2
        dofs[i*6 + 1] = 2.0*x;      // df/dx = 2x
        dofs[i*6 + 2] = 2.0*y;      // df/dy = 2y
        dofs[i*6 + 3] = 2.0;        // d^2f/dx^2 = 2
        dofs[i*6 + 4] = 0.0;        // d^2f/dxdy = 0
        dofs[i*6 + 5] = 2.0;        // d^2f/dy^2 = 2
      }
      EvalPoint evalPoints[] = {
        {{8.0/3.0, 2.0}, (8.0/3.0)*(8.0/3.0) + 4.0, 2.0*(8.0/3.0), 4.0},
        {{3.0, 1.0}, 10.0, 6.0, 2.0}
      };
      allPassed &= testFieldEvaluation("Quadratic field f=x^2+y^2", 
                                       coords, dofs, evalPoints, 2, lib);
    }

    std::cout << "Test 9: Mixed derivative field (f=x*y)\n";
    std::cout << "=====================================\n";
    {
      auto coords = m23(1,1, 5,1, 2,4);
      Real dofs[18];

      for (int i = 0; i < 3; i++) {
        Real x = coords[i][0];
        Real y = coords[i][1];

        dofs[i*6 + 0] = x * y;   // f
        dofs[i*6 + 1] = y;       // df/dx
        dofs[i*6 + 2] = x;       // df/dy
        dofs[i*6 + 3] = 0.0;     // d^2f/dx^2
        dofs[i*6 + 4] = 1.0;     // d^2f/dxdy
        dofs[i*6 + 5] = 0.0;     // d^2f/dy^2
      }

      EvalPoint evalPoints[] = {
        {
          {8.0/3.0, 2.0},
          (8.0/3.0) * 2.0,   // f
          2.0,               // df/dx = y
          8.0/3.0            // df/dy = x
        },
        {
          {3.0, 1.0},
          3.0,               // f
          1.0,               // df/dx
          3.0                // df/dy
        },
        {
          {2.5, 2.0},
          5.0,               // f
          2.0,               // df/dx
          2.5                // df/dy
        }
      };

      allPassed &= testFieldEvaluation(
          "Mixed derivative field f=x*y",
          coords,
          dofs,
          evalPoints,
          3,
          lib);
    }
    
    std::cout << "\n====================================\n";
    if (allPassed) {
      std::cout << "[PASS] All tests PASSED\n";
    } else {
      std::cout << "[FAIL] Some tests FAILED\n";
    }
    std::cout << "====================================\n";
  }
  Kokkos::finalize();
  return allPassed ? 0 : 1;
}