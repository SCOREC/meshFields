#ifndef MESHFIELD_REDUCEDQUINTIC_HPP
#define MESHFIELD_REDUCEDQUINTIC_HPP

#include <MeshField_Defines.hpp>
#include <MeshField_Fail.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>

namespace MeshField {

/**
 * @brief Device-compatible LU decomposition with partial pivoting (internal helper)
 * 
 * Solves the linear system A*X = B where A is n×n and B is n×nrhs.
 * This replaces the need for LAPACK's dgesv for our 20×20 system.
 * 
 * @param n Dimension of the matrix (20 for reduced quintic)
 * @param nrhs Number of right-hand sides (18 for reduced quintic)
 * @param A Input matrix (destroyed on output), row-major format
 * @param lda Leading dimension of A
 * @param B Input/output: RHS on input, solution on output, row-major format
 * @param ldb Leading dimension of B
 * @return 0 on success, positive value if singular
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
int solveLU_internal(int n, int nrhs, Real* A, int lda, Real* B, int ldb) {
  // Use fixed-size array for pivoting (n=20 maximum, which fits all our use cases)
  // We use a stack array since this is device-compatible
  int pivot[20];
  const Real eps = 1e-14;
  
  // LU decomposition with partial pivoting
  for (int k = 0; k < n; k++) {
    // Find pivot
    int maxRow = k;
    Real maxVal = Kokkos::fabs(A[k * lda + k]);
    for (int i = k + 1; i < n; i++) {
      Real val = Kokkos::fabs(A[i * lda + k]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }
    
    if (maxVal < eps) {
      return k + 1; // Singular matrix
    }
    
    pivot[k] = maxRow;
    
    // Swap rows in A if needed
    if (maxRow != k) {
      for (int j = 0; j < n; j++) {
        Real tmp = A[k * lda + j];
        A[k * lda + j] = A[maxRow * lda + j];
        A[maxRow * lda + j] = tmp;
      }
    }
    
    // Eliminate column
    Real diag = A[k * lda + k];
    for (int i = k + 1; i < n; i++) {
      Real factor = A[i * lda + k] / diag;
      A[i * lda + k] = factor; // Store multiplier
      for (int j = k + 1; j < n; j++) {
        A[i * lda + j] -= factor * A[k * lda + j];
      }
    }
  }
  
  // Apply pivoting to B
  for (int k = 0; k < n; k++) {
    if (pivot[k] != k) {
      for (int rhs = 0; rhs < nrhs; rhs++) {
        Real tmp = B[k * ldb + rhs];
        B[k * ldb + rhs] = B[pivot[k] * ldb + rhs];
        B[pivot[k] * ldb + rhs] = tmp;
      }
    }
  }
  
  // Forward substitution: solve L*Y = B for each column
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      Real factor = A[i * lda + j];
      for (int rhs = 0; rhs < nrhs; rhs++) {
        B[i * ldb + rhs] -= factor * B[j * ldb + rhs];
      }
    }
  }
  
  // Backward substitution: solve U*X = Y for each column
  for (int i = n - 1; i >= 0; i--) {
    Real diag = A[i * lda + i];
    for (int j = i + 1; j < n; j++) {
      Real factor = A[i * lda + j];
      for (int rhs = 0; rhs < nrhs; rhs++) {
        B[i * ldb + rhs] -= factor * B[j * ldb + rhs];
      }
    }
    for (int rhs = 0; rhs < nrhs; rhs++) {
      B[i * ldb + rhs] /= diag;
    }
  }
  
  return 0; // Success
}

/**
 * @brief Device-compatible function to rotate DOFs for reduced quintic element
 *
 * @param dofs_p Input DOFs in original orientation (6 DOFs per vertex)
 * @param sin_theta_p Sine of rotation angle
 * @param cos_theta_p Cosine of rotation angle
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
void rotateDof(Real dofs_p[6], Real sin_theta_p, Real cos_theta_p)
{
  const Real s  = sin_theta_p;
  const Real c  = cos_theta_p;
  const Real ss = s * s;
  const Real cc = c * c;
  const Real sc = s * c;

  const Real dofs_r[6] = {
    dofs_p[0],
    c  * dofs_p[1] + s  * dofs_p[2],
    c  * dofs_p[2] - s  * dofs_p[1],
    cc * dofs_p[3] + 2*sc * dofs_p[4] + ss * dofs_p[5],
    -sc * dofs_p[3] + (cc - ss)        * dofs_p[4] + sc * dofs_p[5],
    ss * dofs_p[3] - 2*sc * dofs_p[4] + cc * dofs_p[5]
  };

  for (int i = 0; i < 6; i++)
    dofs_p[i] = dofs_r[i];
}

/**
 * @brief Device-compatible function to transform DOFs from physical to local coordinates
 * 
 * Transforms derivatives from physical (x,y) coordinates to local (ξ,η) coordinates
 * using a rotation. This is necessary because ReducedQuintic shape functions are
 * defined in a local coordinate system aligned with the triangle's geometry.
 * 
 * @param dof_idx DOF component index (0-5: value, ∂x, ∂y, ∂²x², ∂²xy, ∂²y²)
 * @param sin_theta Sine of rotation angle (from elemCoeffs)
 * @param cos_theta Cosine of rotation angle (from elemCoeffs)
 * @param allDofs Array of all 6 DOFs for this vertex in physical coordinates
 * @return The transformed DOF value in local coordinates
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
Real transformDofPhysicalToLocal(
    int dof_idx,
    Real sin_theta,
    Real cos_theta,
    const Real allDofs[6])
{
  const Real s  = sin_theta;
  const Real c  = cos_theta;
  const Real ss = s * s;
  const Real cc = c * c;
  const Real sc = s * c;

  // Transform based on DOF type
  switch (dof_idx) {
    case 0: // value - unchanged
      return allDofs[0];
    case 1: // ∂/∂x → ∂/∂ξ
      return c * allDofs[1] + s * allDofs[2];
    case 2: // ∂/∂y → ∂/∂η
      return c * allDofs[2] - s * allDofs[1];
    case 3: // ∂²/∂x²
      return cc * allDofs[3] + 2*sc * allDofs[4] + ss * allDofs[5];
    case 4: // ∂²/∂x∂y
      return -sc * allDofs[3] + (cc - ss) * allDofs[4] + sc * allDofs[5];
    case 5: // ∂²/∂y²
      return ss * allDofs[3] - 2*sc * allDofs[4] + cc * allDofs[5];
    default:
      return allDofs[dof_idx];
  }
}

/**
 * @brief Device-compatible function to reorder triangle vertices to put longest edge along local xi-axis
 * 
 * @param coords Triangle vertex coordinates [3][2]
 * @param order Output vertex order [3] - order[i] gives original index of reordered vertex i
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
void reorderTriangleVertices(
    const Real coords[3][2],
    int order[3])
{
  // Compute edge vectors
  Real v1v2[2] = {coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]};
  Real v1v3[2] = {coords[2][0] - coords[0][0], coords[2][1] - coords[0][1]};
  Real v2v3[2] = {coords[2][0] - coords[1][0], coords[2][1] - coords[1][1]};
  
  // Check if counter-clockwise
  int counterclockwise = 1;
  if (v1v2[0]*v1v3[1] - v1v2[1]*v1v3[0] < 0) {
    counterclockwise = 0;
  }
  
  // Compute edge length squares
  Real v1v2lensq = v1v2[0]*v1v2[0] + v1v2[1]*v1v2[1];
  Real v1v3lensq = v1v3[0]*v1v3[0] + v1v3[1]*v1v3[1];
  Real v2v3lensq = v2v3[0]*v2v3[0] + v2v3[1]*v2v3[1];
  
  // Find longest edge and set order
  if (v1v2lensq > v1v3lensq && v1v2lensq > v2v3lensq) {
    // Edge v0-v1 is longest
    if (counterclockwise) {
      order[0] = 0;
      order[1] = 1;
      order[2] = 2;
    } else {
      order[0] = 1;
      order[1] = 0;
      order[2] = 2;
    }
  } else {
    if (v1v3lensq > v2v3lensq) {
      // Edge v0-v2 is longest
      if (counterclockwise) {
        order[0] = 2;
        order[1] = 0;
        order[2] = 1;
      } else {
        order[0] = 0;
        order[1] = 2;
        order[2] = 1;
      }
    } else {
      // Edge v1-v2 is longest
      if (counterclockwise) {
        order[0] = 1;
        order[1] = 2;
        order[2] = 0;
      } else {
        order[0] = 2;
        order[1] = 1;
        order[2] = 0;
      }
    }
  }
}

/**
 * @brief Device-compatible function to compute geometric parameters for reduced quintic triangle element
 * 
 * @param coords Original triangle vertex coordinates [3][2]
 * @param origin Output: origin of local coordinate system
 * @param a Output: distance from origin to first vertex in local system
 * @param b Output: distance from origin to second vertex in local system
 * @param c Output: perpendicular distance to third vertex
 * @param sin_theta Output: sine of rotation angle
 * @param cos_theta Output: cosine of rotation angle
 * @param order Output: vertex reordering [3] - order[i] gives original index
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
void computeReducedQuinticGeometry(
    const Real coords[3][2],
    Real origin[2],
    Real& a, Real& b, Real& c,
    Real& sin_theta, Real& cos_theta,
    int order[3])
{
  // First reorder vertices to put longest edge along xi-axis
  reorderTriangleVertices(coords, order);
  
  // Apply reordering
  Real abcCoord[3][2];
  for (int i = 0; i < 3; i++) {
    int idx = order[i];
    abcCoord[i][0] = coords[idx][0];
    abcCoord[i][1] = coords[idx][1];
  }
  
  // Edge from v0 to v1 (now the longest edge)
  Real ab[2] = {abcCoord[1][0] - abcCoord[0][0], abcCoord[1][1] - abcCoord[0][1]};
  Real ablensq = ab[0]*ab[0] + ab[1]*ab[1];
  Real ablen = Kokkos::sqrt(ablensq);
  
  // Vector from v0 to v2
  Real ac[2] = {abcCoord[2][0] - abcCoord[0][0], abcCoord[2][1] - abcCoord[0][1]};
  
  // Project v2 onto v0-v1 edge to find origin
  Real buffer = (ab[0]*ac[0] + ab[1]*ac[1]) / ablensq;
  Real ac2ab[2] = {ab[0] * buffer, ab[1] * buffer};
  origin[0] = ac2ab[0] + abcCoord[0][0];
  origin[1] = ac2ab[1] + abcCoord[0][1];
  
  // Distances from origin to each vertex (in reordered system)
  Real vec_a[2] = {origin[0] - abcCoord[0][0], origin[1] - abcCoord[0][1]};
  Real vec_b[2] = {origin[0] - abcCoord[1][0], origin[1] - abcCoord[1][1]};
  Real vec_c[2] = {origin[0] - abcCoord[2][0], origin[1] - abcCoord[2][1]};
  
  b = Kokkos::sqrt(vec_a[0]*vec_a[0] + vec_a[1]*vec_a[1]);  // distance to v0
  a = Kokkos::sqrt(vec_b[0]*vec_b[0] + vec_b[1]*vec_b[1]);  // distance to v1
  c = Kokkos::sqrt(vec_c[0]*vec_c[0] + vec_c[1]*vec_c[1]);  // distance to v2
  
  // Angle of v0-v1 edge
  sin_theta = ab[1] / ablen;
  cos_theta = ab[0] / ablen;
}

/**
 * @brief Device-compatible function to compute coefficient matrix for reduced quintic element
 * 
 * @param a, b, c Geometric parameters from computeReducedQuinticGeometry
 * @param coeffs Output: coefficient matrix [18][20]
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
void computeReducedQuinticCoefficients(
    Real a, Real b, Real c,
    Real coeffs[18][20])
{
  // Compute powers
  const Real a2 = a*a, a3 = a2*a, a4 = a2*a2, a5 = a2*a3;
  const Real b2 = b*b, b3 = b2*b, b4 = b2*b2, b5 = b2*b3;
  const Real c2 = c*c, c3 = c2*c, c4 = c2*c2, c5 = c2*c3;
  
  // Build constraint matrix A (20x20)
  Real A[20][20] = {
    {1., -b, 0., b2, 0, 0, -b3, 0, 0, 0, b4, 0, 0, 0, 0, -b5, 0, 0, 0, 0},
    {0., 1., 0., -2.*b, 0., 0., 3*b2, 0, 0, 0, -4*b3, 0, 0, 0, 0, 5*b4, 0, 0, 0, 0},
    {0, 0, 1, 0, -b, 0, 0, b2, 0, 0, 0, -b3, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 2, 0, 0, -6*b, 0, 0, 0, 12*b2, 0, 0, 0, 0, -20*b3, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, -2*b, 0, 0, 0, 3*b2, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 2, 0, 0, -2*b, 0, 0, 0, 2*b2, 0, 0, 0, -2*b3, 0, 0, 0},
    {1, a, 0, a2, 0, 0, a3, 0, 0, 0, a4, 0, 0, 0, 0, a5, 0, 0, 0, 0},
    {0, 1, 0, 2*a, 0, 0, 3*a2, 0, 0, 0, 4*a3, 0, 0, 0, 0, 5*a4, 0, 0, 0, 0},
    {0, 0, 1, 0, a, 0, 0, a2, 0, 0, 0, a3, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 2, 0, 0, 6*a, 0, 0, 0, 12*a2, 0, 0, 0, 0, 20*a3, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 2*a, 0, 0, 0, 3*a2, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 2, 0, 0, 2*a, 0, 0, 0, 2*a2, 0, 0, 0, 2*a3, 0, 0, 0},
    {1, 0, c, 0, 0, c2, 0, 0, 0, c3, 0, 0, 0, 0, c4, 0, 0, 0, 0, c5},
    {0, 1, 0, 0, c, 0, 0, 0, c2, 0, 0, 0, 0, c3, 0, 0, 0, 0, c4, 0},
    {0, 0, 1, 0, 0, 2*c, 0, 0, 0, 3*c2, 0, 0, 0, 0, 4*c3, 0, 0, 0, 0, 5*c4},
    {0, 0, 0, 2, 0, 0, 0, 2*c, 0, 0, 0, 0, 2*c2, 0, 0, 0, 0, 2*c3, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 0, 2*c, 0, 0, 0, 0, 3*c2, 0, 0, 0, 0, 4*c3, 0},
    {0, 0, 0, 0, 0, 2, 0, 0, 0, 6*c, 0, 0, 0, 0, 12*c2, 0, 0, 0, 0, 20*c3},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5*a4*c, 3*a2*c3-2*a4*c, -2*a*c4+3*a3*c2, c5-4*a2*c3, 5*a*c4},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5*b4*c, 3*b2*c3-2*b4*c, 2*b*c4-3*b3*c2, c5-4*b2*c3, -5*b*c4}
  };
  
  // Initialize RHS as 20×18 identity matrix (first 18 rows are identity)
  Real B[20][18];
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 18; j++) {
      B[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
  
  // Solve A*X = B using our LU solver (row-major)
  int info = solveLU_internal(20, 18, &A[0][0], 20, &B[0][0], 18);
  
  // Copy solution to coeffs[18][20]
  for (int j = 0; j < 18; j++) {
    for (int i = 0; i < 20; i++) {
      coeffs[j][i] = B[i][j];
    }
  }
}

/**
 * @brief Precompute all reduced quintic coefficients for a mesh, entirely on device
 * 
 * @param numTriangles Number of triangles in the mesh
 * @param triCoords_d Triangle vertex coordinates on device, shape [numTriangles][3][2] (flattened: tri*6 + v*2 + d)
 * @return Kokkos::View with coefficients, shape [numTriangles][8 + 18*20]
 *         Each row: [order[0], order[1], order[2], a, b, c, sin_theta, cos_theta, coeff_0_0, ..., coeff_17_19]
 */
inline Kokkos::View<Real**> precomputeReducedQuinticCoefficients(
    int numTriangles,
    Kokkos::View<Real**> triCoords_d)
{
  // Allocate device view with extended storage for sin_theta and cos_theta
  // Layout: [order[0], order[1], order[2], a, b, c, sin_theta, cos_theta, coeff_0_0, ..., coeff_17_19]
  Kokkos::View<Real**> coeffs_d("coefficients_device", numTriangles, 8 + 18 * 20);
  
  // Compute for each triangle entirely on device
  Kokkos::parallel_for(
      "precomputeReducedQuinticCoefficients", numTriangles,
      KOKKOS_LAMBDA(int tri) {
    Real coords[3][2];
    for (int v = 0; v < 3; v++) {
      coords[v][0] = triCoords_d(tri, v * 2 + 0);
      coords[v][1] = triCoords_d(tri, v * 2 + 1);
    }
    
    // Compute geometric parameters
    Real origin[2], a, b, c, sin_theta, cos_theta;
    int order[3];
    computeReducedQuinticGeometry(coords, origin, a, b, c, sin_theta, cos_theta, order);
    
    // Store vertex order, geometric parameters, and rotation angles
    coeffs_d(tri, 0) = static_cast<Real>(order[0]);
    coeffs_d(tri, 1) = static_cast<Real>(order[1]);
    coeffs_d(tri, 2) = static_cast<Real>(order[2]);
    coeffs_d(tri, 3) = a;
    coeffs_d(tri, 4) = b;
    coeffs_d(tri, 5) = c;
    coeffs_d(tri, 6) = sin_theta;
    coeffs_d(tri, 7) = cos_theta;
    
    // Compute coefficients
    Real coeffs_tri[18][20];
    computeReducedQuinticCoefficients(a, b, c, coeffs_tri);
    
    // Store in flattened format after order, geometric parameters, and rotation angles
    for (int i = 0; i < 18; i++) {
      for (int j = 0; j < 20; j++) {
        coeffs_d(tri, 8 + i * 20 + j) = coeffs_tri[i][j];
      }
    }
  });
  
  return coeffs_d;
}
} // namespace MeshField

#endif // MESHFIELD_REDUCEDQUINTIC_HPP