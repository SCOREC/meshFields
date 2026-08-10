# Adding Shape Functions for Omega_h Element Topologies {#adding-shape-functions-omegah}

This guide walks through every step required to add a new shape function
for an Omega\_h element topology (triangle, tetrahedron, quad, etc.).
The steps are:

1. Define the **shape function struct** in `src/MeshField_Shape.hpp`
2. Define the **Omega\_h node mapping struct** in `src/MeshField.hpp`
3. Register the shape + mapping in the **element factory** in `src/MeshField.hpp`
4. Add an **Accessor** (if needed) in `src/MeshField_ShapeField.hpp`
5. Extend **`CreateLagrangeField`** in `src/MeshField_ShapeField.hpp`
6. Add **integration support** in `src/MeshField_Integrate.hpp`
7. (Optional) Write an **`Integrator`-derived class** to perform numerical integration with the new shape.

[TOC]

---

## Background

### Terminology

Refer to the @ref nomenclature "Nomenclature" section on the main page.

### Parametric Coordinates and Node Ordering

MeshFields defines shape functions in a canonical parametric coordinate
system that is independent of Omega\_h.  The **node mapping struct** (Step 2)
bridges the two conventions.

The MeshFields canonical parametric coordinates and node ordering for linear and
quadratic triangles and tetrahedrons follows,
"The Finite Element Method: Its Basis and Fundamentals", 2013,
Zienkiewicz, Taylor, and Zhu.

MeshFields uses \f$d-1\f$ parametric coordinates to specify a location within an
element of dimension \f$d\f$.  The redundant coordinate \f$L_0 = 1 - \sum \xi_i\f$ is
omitted:

| Topology      | Parametric coords | Range           |
|---------------|-------------------|-----------------|
| Edge (1D)     | \f$\xi\f$         | \f$[-1, 1]\f$   |
| Triangle (2D) | \f$(\xi_0,\xi_1)\f$  | \f$[0,1]^2\f$, \f$\xi_0+\xi_1 \le 1\f$ |
| Tet (3D)      | \f$(\xi_0,\xi_1,\xi_2)\f$ | \f$[0,1]^3\f$, \f$\sum \xi_i \le 1\f$ |

---

## Step 1 - Define the Shape Function Struct {#step1}

@ref adding-shape-functions-omegah "Back to top"

Add a new struct to `src/MeshField_Shape.hpp` inside `namespace MeshField`.

### Required Members

| Member | Description |
|--------|-------------|
| `numNodes` | Total nodes per element |
| `meshEntDim` | Parametric space dimension |
| `Order` | Polynomial order |
| `DofHolders` | Entity types that hold DOFs |
| `getNodeParametricCoords()` | Returns a flat array of node coordinates (length `numNodes * meshEntDim`) |
| `getValues(xi)` | Returns an array of shape function values at `xi` (length `numNodes`) |
| `getLocalGradients(xi)` | Returns a flat array of gradients at `xi` (length `meshEntDim * numNodes`, row-major: \f$[\partial N_0/\partial\xi_0, \partial N_0/\partial\xi_1, \ldots, \partial N_d/\partial\xi_0, \partial N_d/\partial\xi_1]\f$) |

Optional (needed by quadratic and higher-order shapes):

| Member | Description |
|--------|-------------|
| `NumDofHolders` | Count of entities of each type in `DofHolders` |
| `DofsPerHolder` | DOFs per entity for each type in `DofHolders` |

### Checking Parametric Coordinates

Use the helper functions declared at the top of `MeshField_Shape.hpp` to
assert that incoming parametric coordinates are in range.
This is only a check for outliers as `MeshField::ParametricCoordTol` is
a loose tolerance.

```cpp
assert(eachGreaterThanOrEqual(xi, 0.0, ParametricCoordTol));
assert(eachLessThanOrEqual(xi, 1.0, ParametricCoordTol));
const Real L0 = 1 - xi[0] - xi[1];
assert(greaterThanOrEqual(L0, 0.0, ParametricCoordTol));
```

### Example: `QuadraticTriangleShape` (from `src/MeshField_Shape.hpp`)

\snippet src/MeshField_Shape.hpp QuadraticTriangleShape

---

## Step 2 - Define the Omega\_h Node Mapping Struct {#step2}
@ref adding-shape-functions-omegah "Back to top"

The mapping struct lives in `src/MeshField.hpp` inside
`namespace MeshField::Omegah`.  It translates element-local node indices
(as numbered by the shape function) into the on-process mesh entity indices
(as numbered by Omega\_h) that hold the corresponding DOFs.

### Vertex-Ordering Correction

As previously stated, Omega\_h numbers vertices and edges within an element 
differently from the MeshFields canonical ordering used by the shape functions.
The Omega\_h canonical orderings for simplices are depicted below:

<a href="omegah_tri.png"><img src="omegah_tri.png" width="20%" alt="Omega_h triangle ordering"/></a>

Omega_h triangle ordering.

<a href="omegah_tetOrdering.png"><img src="omegah_tetOrdering.png" width="20%" alt="Omega_h tetrahedron ordering"/></a>

Omega_h tetrahedron ordering.

Meshes composed of hypercubes (quadrilaterals and hexahedrons), pyramids, and prisms are supported by Omega_h, but they cannot be adapted.
Version 1.1.0 of MeshFields only supports Omega\_h simplices.

The existing linear-triangle and linear-tetrahedron mappings correct for
the difference in Omega\_h vs MeshFields ordering with a cyclic rotation.

```cpp
// For triangles (triDim=2, vtxDim=0):
const auto localVtxIdx =
    (Omega_h::simplex_down_template(triDim, vtxDim, nodeIdx, /*ignored=*/-1) + 2) % 3;

// For tetrahedra (tetDim=3, vtxDim=0):
const auto localVtxIdx =
    (Omega_h::simplex_down_template(tetDim, vtxDim, nodeIdx, /*ignored=*/-1) + 3) % 4;
```

You must determine the correct rotation offset for your topology and DOF holders by
comparing the Omega\_h canonical vertex/edge ordering (see
figures above based on `Omega_h_simplex.hpp` and the `simplex_down_template` function)
with the node ordering established by `getNodeParametricCoords()` in your shape
struct.  It is a good idea to add a unit test that evaluates the shape functions at
the parametric coordinates of each node and checks that the value for
that node is 1.0 and all others are 0.0.

### Interface Requirements

| Member | Description |
|--------|-------------|
| Constructor `(Omega_h::Mesh &)` | Cache connectivity arrays from Omega\_h; validate mesh family/dimension |
| `getTopology()` | Return an array of mesh topologies (vertex, edge, triangle, etc...) this mapping applies to |
| `operator()(LO nodeIndex_in, LO componentIndex_in, LO elementIndex_in, Mesh_Topology elementTopo_in)` | Return mapping tuple - see @ref MeshField::ElementToDofHolderMap |

### Omega\_h Connectivity APIs

| Query | API call |
|-------|----------|
| Element->vertex connectivity | `mesh.ask_elem_verts()` -> flat `LOs` array, stride = `simplex_degree(elemDim, 0)` |
| Element->edge connectivity   | `mesh.ask_down(elemDim, 1).ab2b` -> flat `LOs` array, stride = `simplex_degree(elemDim, 1)` |
| Element->face connectivity   | `mesh.ask_down(elemDim, 2).ab2b` |

### Example: `QuadraticTriangleToField` (from `src/MeshField.hpp`)

\snippet src/MeshField.hpp QuadraticTriangleToField

---

## Step 3 - Register in the Element Factory {#step3}
@ref adding-shape-functions-omegah "Back to top"

Add a new factory function (or extend an existing one) in
`src/MeshField.hpp` inside `namespace MeshField::Omegah`.
The existing `getTriangleElement` factory (from `src/MeshField.hpp`) shows the full pattern:

\snippet src/MeshField.hpp getTriangleElement

Callers obtain the shape and mapping via structured bindings:

```cpp
const auto [shp, map] = MeshField::Omegah::getTriangleElement<2>(mesh);
MeshField::FieldElement fes(mesh.nelems(), field, shp, map);
```

---

## Step 4 - Add an Accessor {#step4}
@ref adding-shape-functions-omegah "Back to top"

An Accessor is a templated struct that provides `operator()(entity, node, component, topology)` for reading and writing DOF values from the underlying field storage.
It is passed as a variadic template argument (`Mixins...`) to @ref MeshField::ShapeField "ShapeField".

Two built-in Accessors cover the common cases:

@ref MeshField::LinearAccessor "LinearAccessor" -- for shapes whose DOFs live
only at vertices.  All calls to `operator()` are forwarded to a single vertex
field regardless of topology:

\snippet src/MeshField_ShapeField.hpp LinearAccessor

@ref MeshField::QuadraticAccessor "QuadraticAccessor" -- for shapes whose DOFs
live at both vertices and edges.  The `operator()` dispatches to the vertex
field or edge field based on the `topology` argument:

\snippet src/MeshField_ShapeField.hpp QuadraticAccessor

If your shape introduces DOFs at a new entity type (e.g., triangle faces for a
cubic shape), define a new Accessor following the same pattern:

1. Add a field member (as done in `QuadraticAccessor` for `EdgeAccessor edgeField;,`) for each new entity type.
2. Extend `operator()` with a branch for the added topology.
3. List all supported topologies in the `topo` array.

---

## Step 5 - Extend `CreateLagrangeField` {#step5}
@ref adding-shape-functions-omegah "Back to top"

@ref MeshField::CreateLagrangeField "CreateLagrangeField<ExecutionSpace, Controller, DataType, order, dim, numComp>(meshInfo)" (in `src/MeshField_ShapeField.hpp`) is the user-facing factory that allocates field storage and assembles a @ref MeshField::ShapeField "ShapeField".
Add a new `if constexpr` branch for your shape's `order` and `dim`.
The steps within that branch are:

1. **Validate** `meshInfo` counts for every entity type that holds DOFs (e.g., `meshInfo.numVtx`, `meshInfo.numEdge`).  Call `fail()` if a required count is zero.
2. **Allocate storage** -- construct a `Controller` sized for each DOF-holding entity type.  Each field in the controller needs entries for `numEntities * dofsPerHolder * numComp`.
3. **Create fields** -- call `MeshField::makeField<Ctrlr, N>(ctrl)` for each field index `N` to obtain typed slice objects.
4. **Construct the Accessor** -- pass the slice objects to your Accessor's constructor (e.g., `QuadraticAccessor{vtxField, edgeField}`).
5. **Return** `FieldWithController<Ctrlr, ShapeField<numComp, YourShape, YourAccessor>>{ctrl, ShapeField(meshInfo, accessor)}`.

See the existing `order == 2 && dim == 2` branch (quadratic triangle) as a
concrete reference:

\snippet src/MeshField_ShapeField.hpp CreateFieldControllerQuadratic

---

## Step 6 - Add Integration Support {#step6}
@ref adding-shape-functions-omegah "Back to top"

`src/MeshField_Integrate.hpp` provides quadrature rules through the
`getIntegration<Mesh_Topology>()` template function.  To integrate over
your new topology you must:

### 6a - Define an `EntityIntegration` Class

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

The existing `TriangleIntegration` (from `src/MeshField_Integrate.hpp`) shows the full pattern including multiple quadrature orders:

\snippet src/MeshField_Integrate.hpp TriangleIntegration

### 6b - Extend `getIntegration<topo>()`

Add a branch to the existing `getIntegration` function (current state in `src/MeshField_Integrate.hpp`):

\snippet src/MeshField_Integrate.hpp getIntegration

---

## Step 7 - Implement an Integrator (Optional) {#step7}
@ref adding-shape-functions-omegah "Back to top"

Derive from `MeshField::Integrator` and override `atPoints` to perform
the actual integration.  `Integrator::process(FieldElement)` handles
quadrature point setup and calls `atPoints` with:

| Argument | Type | Description |
|----------|------|-------------|
| `p` | `Kokkos::View<Real**>` | Local parametric coordinates, shape `(numElems * numPts, dim)` |
| `w` | `Kokkos::View<Real*>` | Quadrature weights, length `numElems * numPts` |
| `dV` | `Kokkos::View<Real*>` | Jacobian determinants, length `numElems * numPts` |

`test/testCountIntegrator.cpp` provides a working example. The integrator class:

\snippet test/testCountIntegrator.cpp CountIntegrator

And the function that wires together the element factory, `FieldElement`, integrator, and `process` call:

\snippet test/testCountIntegrator.cpp doRun

