#!/usr/bin/env python3
"""Generate the reference basis of the extended Bell element.

The Bell element is the subspace of quintic polynomials on a triangle whose
normal derivative is cubic on every edge, with 18 degrees of freedom: the
value, gradient, and Hessian at each vertex.  Following Kirby, "A general
approach to transforming finite elements", SMAI J. Comput. Math. 4 (2018),
section 4.2, it is mapped from a reference element for the full quintic space
(21 functions) whose dual basis adds one functional per edge:

    lambda_e(p) = int_0^1 P4(2s - 1) n_e . grad p(v_a + s (v_b - v_a)) ds

the moment of the normal derivative against the degree 4 Legendre polynomial.
A quintic has a cubic normal derivative on edge e iff lambda_e(p) = 0, so the
first 18 basis functions of this element are the Bell basis on the reference
triangle, and the last 3 are needed only to transform it to other triangles.

The dual basis matches FIAT's Bell element (FIAT/bell.py in
https://github.com/firedrakeproject/fiat):
  - reference triangle (0,0), (1,0), (0,1)
  - for each vertex: p, p_x, p_y, p_xx, p_xy, p_yy
  - edges e0 = (v1, v2), e1 = (v0, v2), e2 = (v0, v1), oriented from the first
    to the second vertex, with unit normal (t_y, -t_x) for t = v_b - v_a

The basis is computed exactly with SymPy (https://www.sympy.org) as the
inverse of the 21 x 21 matrix of functionals applied to the monomials, and
printed as a C++ table of monomial coefficients.

Usage: python3 generate_bell_reference_basis.py > table.txt
"""

import sympy as sp

x, y, s = sp.symbols("x y s")

# monomials x^a y^b ordered by total degree, then by decreasing power of x
MONOMIALS = [(d - j, j) for d in range(6) for j in range(d + 1)]
VERTICES = [sp.Matrix([0, 0]), sp.Matrix([1, 0]), sp.Matrix([0, 1])]
EDGES = [(1, 2), (0, 2), (0, 1)]


def vertex_functionals(v):
    """value, gradient, and Hessian at vertex v"""
    at = {x: v[0], y: v[1]}
    return [
        lambda p: p.subs(at),
        lambda p: sp.diff(p, x).subs(at),
        lambda p: sp.diff(p, y).subs(at),
        lambda p: sp.diff(p, x, 2).subs(at),
        lambda p: sp.diff(p, x, y).subs(at),
        lambda p: sp.diff(p, y, 2).subs(at),
    ]


def edge_functional(a, b):
    """moment of the normal derivative against P4 along edge (a, b)"""
    va, vb = VERTICES[a], VERTICES[b]
    t = vb - va
    n = sp.Matrix([t[1], -t[0]]) / t.norm()
    point = va + s * t
    legendre4 = sp.legendre(4, 2 * s - 1)

    def functional(p):
        dn = n[0] * sp.diff(p, x) + n[1] * sp.diff(p, y)
        along = dn.subs({x: point[0], y: point[1]}, simultaneous=True)
        return sp.integrate(sp.expand(legendre4 * along), (s, 0, 1))

    return functional


def functionals():
    nodes = []
    for v in VERTICES:
        nodes.extend(vertex_functionals(v))
    nodes.extend(edge_functional(a, b) for a, b in EDGES)
    return nodes


def basis_coefficients():
    """C[k][j] is the coefficient of MONOMIALS[j] in basis function k"""
    monomials = [x**a * y**b for a, b in MONOMIALS]
    nodes = functionals()
    A = sp.Matrix(len(nodes), len(monomials),
                  lambda i, j: nodes[i](monomials[j]))
    # nodes[i](phi_k) = delta_ik with phi_k = sum_j C[k][j] monomials[j]
    return (A.inv()).T


def main():
    C = basis_coefficients()
    rows = []
    for k in range(C.rows):
        entries = ", ".join(f"{float(C[k, j]):.17g}" for j in range(C.cols))
        rows.append(f"    {{{entries}}}")
    print("// clang-format off")
    print("constexpr Real coefficients[21][21] = {")
    print(",\n".join(rows))
    print("};")
    print("// clang-format on")


if __name__ == "__main__":
    main()
