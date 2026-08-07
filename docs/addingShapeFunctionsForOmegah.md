# Adding Shape Functions for Omega_h Element Topologies {#adding-shape-functions-omegah}

This guide walks through every step required to add a new shape function
for an Omega\_h element topology (triangle, tetrahedron, quad, etc.).
Four source locations must be modified in order:

1. Define the **shape function struct** in `src/MeshField_Shape.hpp`
2. Define the **Omega\_h node mapping struct** in `src/MeshField.hpp`
3. Register the shape + mapping in the **element factory** in `src/MeshField.hpp`
4. Add **integration support** in `src/MeshField_Integrate.hpp`

An optional fifth step covers writing an `Integrator`-derived class to
perform numerical integration with the new shape.

---

## Background

### Parametric Coordinates and Node Ordering

meshFields defines shape functions in a canonical parametric coordinate
system that is independent of Omega\_h.  Omega\_h uses its own node
ordering for element vertices and edges, which may differ from the
meshFields canonical ordering.  The **node mapping struct** (Step 2)
bridges the two conventions.

For simplex elements the parametric coordinates are reduced barycentric
coordinates.  The redundant coordinate \f$L_0 = 1 - \sum \xi_i\f$ is
omitted:

| Topology      | Parametric coords | Range           |
|---------------|-------------------|-----------------|
| Edge (1D)     | \f$\xi\f$         | \f$[-1, 1]\f$   |
| Triangle (2D) | \f$(\xi_0,\xi_1)\f$  | \f$[0,1]^2\f$, \f$\xi_0+\xi_1 \le 1\f$ |
| Tet (3D)      | \f$(\xi_0,\xi_1,\xi_2)\f$ | \f$[0,1]^3\f$, \f$\sum \xi_i \le 1\f$ |

### Terminology

- **Node** – a parametric location on an element that carries a DOF.
- **DOF holder** – the mesh entity (Vertex, Edge, …) that owns a DOF.
  For a linear triangle the three nodes coincide with the three vertices.
  For a quadratic triangle there are also three mid-edge nodes.
- **Node mapping** – a callable that, given a node index local to an
  element, returns the global mesh entity index and topology type that
  holds the corresponding DOF.

---

## Step 1 – Define the Shape Function Struct

Add a new struct to `src/MeshField_Shape.hpp` inside `namespace MeshField`.

### Required Members

| Member | Type | Description |
|--------|------|-------------|
| `numNodes` | `static const size_t` | Total nodes per element |
| `meshEntDim` | `static const size_t` | Parametric space dimension |
| `Order` | `constexpr static size_t` | Polynomial order |
| `DofHolders` | `constexpr static Mesh_Topology[]` | Entity types that hold DOFs |
| `getNodeParametricCoords()` | `KOKKOS_INLINE_FUNCTION` | Flat array of node coordinates (length `numNodes * meshEntDim`) |
| `getValues(xi)` | `KOKKOS_INLINE_FUNCTION` | Shape function values at `xi` (length `numNodes`) |
| `getLocalGradients(xi)` | `KOKKOS_INLINE_FUNCTION` | Gradients at `xi` (length `meshEntDim * numNodes`, row-major: \f$[\partial N_0/\partial\xi_0, \partial N_0/\partial\xi_1, \ldots]\f$) |

Optional (needed by quadratic and higher-order shapes):

| Member | Type | Description |
|--------|------|-------------|
| `NumDofHolders` | `constexpr static size_t[]` | Count of entities of each type in `DofHolders` |
| `DofsPerHolder` | `constexpr static size_t[]` | DOFs per entity for each type in `DofHolders` |

### Validating Parametric Coordinates

Use the helper functions declared at the top of `MeshField_Shape.hpp` to
assert that incoming parametric coordinates are in range.  Pass
`MeshField::ParametricCoordTol` as the tolerance.

```cpp
assert(eachGreaterThanOrEqual(xi, 0.0, ParametricCoordTol));
assert(eachLessThanOrEqual(xi, 1.0, ParametricCoordTol));
// for barycentric remainder:
const Real L0 = 1 - xi[0] - xi[1];
assert(greaterThanOrEqual(L0, 0.0, ParametricCoordTol));
```

### Example: `QuadraticTriangleShape` (from `src/MeshField_Shape.hpp`)

\snippet src/MeshField_Shape.hpp QuadraticTriangleShape

---

## Step 2 – Define the Omega\_h Node Mapping Struct

The mapping struct lives in `src/MeshField.hpp` inside
`namespace MeshField::Omegah`.  It translates element-local node indices
(as numbered by the shape function) into the global mesh entity indices
(as numbered by Omega\_h) that hold the corresponding DOFs.

### Interface Requirements

| Member | Description |
|--------|-------------|
| Constructor `(Omega_h::Mesh &)` | Cache connectivity arrays from Omega\_h; validate mesh family/dimension |
| `static constexpr getTopology()` | Return a `Kokkos::Array<Mesh_Topology, N>` listing element topologies this mapping applies to |
| `operator()(LO nodeIdx, LO compIdx, LO elem, Mesh_Topology topo)` | Return `ElementToDofHolderMap{node, comp, ent, topo}` |

`ElementToDofHolderMap` packs four values: `{nodeLocalIdx, componentIdx, globalEntityIdx, entityTopology}`.
For typical shape functions `nodeLocalIdx` is always `0`.

### Omega\_h Connectivity APIs

| Query | API call |
|-------|----------|
| Element→vertex connectivity | `mesh.ask_elem_verts()` → flat `LOs` array, stride = `simplex_degree(elemDim, 0)` |
| Element→edge connectivity   | `mesh.ask_down(elemDim, 1).ab2b` → flat `LOs` array, stride = `simplex_degree(elemDim, 1)` |
| Element→face connectivity   | `mesh.ask_down(elemDim, 2).ab2b` |

### Vertex-Ordering Correction

Omega\_h numbers vertices and edges within a simplex differently from the
meshFields canonical ordering used by the shape functions.
The existing linear-triangle and linear-tetrahedron mappings correct for
this with a cyclic rotation:

```cpp
// For triangles (triDim=2, vtxDim=0):
const auto localVtxIdx =
    (Omega_h::simplex_down_template(triDim, vtxDim, nodeIdx, /*ignored=*/-1) + 2) % 3;

// For tetrahedra (tetDim=3, vtxDim=0):
const auto localVtxIdx =
    (Omega_h::simplex_down_template(tetDim, vtxDim, nodeIdx, /*ignored=*/-1) + 3) % 4;
```

You must determine the correct rotation offset for your topology by
comparing the Omega\_h canonical vertex/edge ordering (see
`Omega_h_simplex.hpp` and the `simplex_down_template` function) with the
node ordering established by `getNodeParametricCoords()` in your shape
struct.  Validate with a unit test that evaluates the shape functions at
the parametric coordinates of each node and checks that the value for
that node is 1.0 and all others are 0.0.

### Example Skeleton (Linear Quad, vertices only)

```cpp
struct LinearQuadToVertexField {
  Omega_h::LOs quadVerts;

  LinearQuadToVertexField(Omega_h::Mesh &mesh)
      : quadVerts(mesh.ask_elem_verts()) {
    if (mesh.dim() != 2 || mesh.family() != OMEGA_H_HYPERCUBE)
      MeshField::fail("Mesh must be 2D hypercube (quads)\n");
  }

  static constexpr KOKKOS_FUNCTION Kokkos::Array<MeshField::Mesh_Topology, 1>
  getTopology() {
    return {MeshField::Quad};
  }

  KOKKOS_FUNCTION MeshField::ElementToDofHolderMap
  operator()(MeshField::LO nodeIdx, MeshField::LO compIdx,
             MeshField::LO elem,   MeshField::Mesh_Topology topo) const {
    assert(topo == MeshField::Quad);
    const auto quadDim = 2, vtxDim = 0;
    // Determine the rotation offset by comparing Omega_h vertex ordering
    // with your shape function's getNodeParametricCoords() ordering.
    const auto localVtxIdx =
        (Omega_h::simplex_down_template(quadDim, vtxDim, nodeIdx, -1) + OFFSET) % 4;
    const auto stride = Omega_h::simplex_degree(quadDim, vtxDim);
    const MeshField::LO vtx = quadVerts[elem * stride + localVtxIdx];
    return {0, compIdx, vtx, MeshField::Vertex};
  }
};
```

Replace `OFFSET` with the value (0–3) that maps Omega\_h's vertex order
to the meshFields canonical order for your element.  See the existing
`LinearTriangleToVertexField` (offset 2) and
`LinearTetrahedronToVertexField` (offset 3) in `src/MeshField.hpp` for
reference.

---

## Step 3 – Register in the Element Factory

Add a new factory function (or extend an existing one) in
`src/MeshField.hpp` inside `namespace MeshField::Omegah`.
The existing helpers follow the pattern below:

```cpp
template <int ShapeOrder> auto getQuadElement(Omega_h::Mesh &mesh) {
  static_assert(ShapeOrder == 1); // extend as higher orders are added
  if constexpr (ShapeOrder == 1) {
    struct result {
      MeshField::LinearQuadShape       shp;
      LinearQuadToVertexField          map;
    };
    return result{MeshField::LinearQuadShape(),
                  LinearQuadToVertexField(mesh)};
  }
}
```

Callers obtain the shape and mapping via structured bindings:

```cpp
const auto [shp, map] = MeshField::Omegah::getQuadElement<1>(mesh);
MeshField::FieldElement fes(mesh.nelems(), field, shp, map);
```

---

## Step 4 – Add Integration Support

`src/MeshField_Integrate.hpp` provides quadrature rules through the
`getIntegration<Mesh_Topology>()` template function.  To integrate over
your new topology you must:

### 4a – Define an `EntityIntegration` Class

Derive from `EntityIntegration<D>` where `D = meshEntDim` of your shape.
Implement at least one `Integration<D>` inner class that returns a vector
of `IntegrationPoint<D>` objects.  Each point carries:

| Field | Meaning |
|-------|---------|
| `param` | Parametric coordinates (reduced barycentric, length `D`) |
| `weight` | Quadrature weight |
| `dim` | Topological dimension of the entity the point is classified on |
| `idx` | Local entity index within the element |

For the triangle, quadrature weights include the reference area factor
(\f$1/2\f$); for the tetrahedron they include \f$1/6\f$.  Match the
convention used by `TriangleIntegration` and `TetrahedronIntegration`
already in the file.

```cpp
class QuadIntegration : public EntityIntegration<2> {
public:
  class N1 : public Integration<2> {
  public:
    int countPoints() const override { return 1; }
    std::vector<IntegrationPoint<2>> getPoints() const override {
      // 1-point Gauss rule for the reference quad [-1,1]^2, weight=4*(1/4)=1
      return {IntegrationPoint(Vector2{0.0, 0.0}, 1.0, 2, 0)};
    }
    int getAccuracy() const override { return 1; }
  };
  int countIntegrations() const override { return 1; }
  Integration<2> const *getIntegration(int i) const override {
    static N1 i1;
    static Integration<2> *integrations[1] = {&i1};
    return integrations[i];
  }
};
```

### 4b – Extend `getIntegration<topo>()`

Add a branch to the existing `getIntegration` function:

```cpp
template <Mesh_Topology topo> auto const getIntegration() {
  if constexpr (topo == Triangle) {
    return std::make_shared<TriangleIntegration>();
  } else if constexpr (topo == Tetrahedron) {
    return std::make_shared<TetrahedronIntegration>();
  } else if constexpr (topo == Quad) {          // <-- add this
    return std::make_shared<QuadIntegration>();
  }
  fail("getIntegration does not support given topology\n");
}
```

---

## Step 5 – Implement an Integrator (Optional)

Derive from `MeshField::Integrator` and override `atPoints` to perform
the actual integration.  `Integrator::process(FieldElement)` handles
quadrature point setup and calls `atPoints` with:

| Argument | Type | Description |
|----------|------|-------------|
| `p` | `Kokkos::View<Real**>` | Local parametric coordinates, shape `(numElems * numPts, dim)` |
| `w` | `Kokkos::View<Real*>` | Quadrature weights, length `numElems * numPts` |
| `dV` | `Kokkos::View<Real*>` | Jacobian determinants, length `numElems * numPts` |

```cpp
template <typename FieldElement>
class MyIntegrator : public MeshField::Integrator {
  FieldElement &fes;
  MeshField::Real result = 0.0;
public:
  MyIntegrator(FieldElement &fe) : Integrator(/*order=*/1), fes(fe) {}

  void atPoints(Kokkos::View<MeshField::Real **> p,
                Kokkos::View<MeshField::Real  *> w,
                Kokkos::View<MeshField::Real  *> dV) override {
    MeshField::Real local = 0.0;
    Kokkos::parallel_reduce(
        p.extent(0),
        KOKKOS_LAMBDA(int i, MeshField::Real &sum) { sum += w(i) * dV(i); },
        local);
    result += local;
  }

  MeshField::Real getResult() const { return result; }
};
```

Invoke it:

```cpp
const auto [shp, map] = MeshField::Omegah::getQuadElement<1>(mesh);
MeshField::FieldElement fes(mesh.nelems(), coordField, shp, map);
MyIntegrator<decltype(fes)> integrator(fes);
integrator.process(fes);
```

See `test/testCountIntegrator.cpp` for a working end-to-end example using
the existing triangle and tetrahedron shapes.

---

## Checklist

- [ ] Shape struct added to `src/MeshField_Shape.hpp` with all required members
- [ ] `getValues` and `getLocalGradients` validated at each node's parametric coords (partition-of-unity and Kronecker-delta checks)
- [ ] Omega\_h mapping struct added to `src/MeshField.hpp`, `namespace MeshField::Omegah`
- [ ] Vertex/edge ordering offset determined and verified against Omega\_h simplex templates
- [ ] Factory function `get<Topology>Element<Order>` added or extended in `src/MeshField.hpp`
- [ ] `EntityIntegration` class added to `src/MeshField_Integrate.hpp`
- [ ] `getIntegration<topo>()` extended for the new topology
- [ ] Test added that constructs a mesh, builds a `FieldElement`, and calls `Integrator::process`
