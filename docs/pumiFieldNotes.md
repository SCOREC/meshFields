## References

- pumi users guide: https://www.scorec.rpi.edu/pumi/PUMI.pdf
- pumi TOMS paper: https://www.scorec.rpi.edu/REPORTS/2015-4.pdf
- pumi apf source code
- pumi doxygen: https://www.scorec.rpi.edu/pumi/doxygen/
- pumi apf library documentation: https://github.com/SCOREC/core/blob/bcfbd128b65a629241b629c90e3665b539e2e9ae/apf/apf.tex
- Mark Beall's thesis, Chapter 8: https://scorec.rpi.edu/REPORTS/1999-6.pdf 
  - An object-oriented field API used within the framework ('Trellis') of
    other objects described in the thesis (mesh, model, solution, etc.).
  - Designed to support p-adaptivity (i.e., non-uniform field order)

## Design Ideas

- don't want dependency on omegah
- have omegah implement the interface outside of meshfields
- pumi APF requires attaching the data to the mesh object
  - omegah supports attached data
  - can the per-entity functions in apf be made callable on the GPU? e.g., call `getValues()` within a kernel
     - no, `getValues` allocates memory for the result via can::NewArray which
       calls the runtime can::Array(n) constructor

## Questions

- Do we want/need to support the following features?
  - non-uniform P
  - Nedelec shape functions
  - mixed meshes
  - polygonal meshes - i.e., wachpress shape functions for seaice
- How will these fields interact with mesh adaptation?
  - convert back to mesh library native format?
    - consider defining a field backend that uses flat arrays for easy/clean
      conversion. Christian's serialize/deserialize functions do this.  It may
      be 'better' for these operations to be explicit/exposed.
  - can omegah adapt use these operations for cavity based field transfer
    without invasive changes? Do we define 'cavity' specific field operations?
- Are their fields that require storage of a matrix of values at each dof?


## Terminology

- dof - exists at a dof holder, can be scalar, vector, matrix, etc.
- dof holder
  - can contain a dof
  - possible holder: mesh entities, quadrature points, element centroids, etc.
- node - a location on a mesh entitiy that is a dof holder. Multiple nodes can
         exist per mesh entity.  For example, a mesh edge could have multiple
         nodes for a high order shape function.

## Functionality

- data types - any POD, don't store everything as a double
- primative data containers that can be stored per dof in a given field
  - scalar
  - vector - three components 
     - is it general enough to support vectorNd where N={1,2,3}
- field
  - requires shape, ownership, and mesh association
- shape - defines the node distribution where the coordinates and field values (DOF’s) are stored
  - pumi supports the following - each row contains <name, order [, notes]>
    - linear, 1
    - lagrange, {1,2}
    - serendipity, 2
    - constant, 1, constant shape fn over every entity up to dimension 3
    - integration point, {1,2,3}, not clear
    - voroni, ?, not clear - related to integration point field shape
    - integration point fixed, ?, not clear
    - hierarchic, {1,2}
  - do we want to support different orders across a given mesh
    (adaptive/non-uniform p)? - may need something like element blocks to do
    this effectively on gpus
- dof numbering
  - pumi supports this
  - landice needs this
  - the rules/protocol for numbering may not be consistent between applications
- node 
  - get/set nodal values
  - get/set nodal coordinates - why?
  - number of nodes for each mesh entity type
  - etc.
- interogation/math ops
  - add - add a constant (scalar/vector/matrix) to each dof
  - scale - multiply each dof by scalar value
  - evaluate - compute value of field at parametric location within mesh entity
             - see Beall Section 8.2, pg 79 (of 139)
             - **how does this work with the shape function definitions? axpy is part of it IIUC**
- inter-process parallelism - distributed memory across multiple GPUs/CPUs via MPI
  - ownership - which process owns each node, can be defined via 
                function (a protocol) or array of ints (one for each node)
  - synchronize - owner to non-owners
  - accumulate - sum non-owners and owner then synchronize
  - isSynchronized - checks if field is synchronized 
- i/o
  - embed meta data about creation, version, mesh association etc.
  - render - output to vtu - will require mesh topology and shape info
  - write to binary file - supports basic checkpointing
  - write to ascii file
- mesh adaptation
  - support cavity based operators
  - conservative field transfer will require knowing about cavities 
  - unsure what operations will be needed
  - in short term we can convert the field to an omegah tag and let omega manage
    the transfer
  - in longer term we will have to modify omegah (e.g., hooks to call meshField
    APIs from kernels)

## PUMI Fields Review

Questions - how is data stored?  which class has the data?

### Classes:

FieldBase 
- pure virtual
- parent of Field <- FieldOf <- MatrixField, MixedVectorField, ScalarField, VectorField
- parent of NumberingOf
- includes pointer to Mesh, FieldShape, and FieldData
- not templated
- provide meta data functions - counts, types, {set|get}{Data|Shape}, rename, etc.
- no math

Field
- pure virtual
- parent of FieldOf <- MatrixField, MixedVectorField, ScalarField, VectorField
- parent of PackedField
- not templated
- adds public methods
  - getElement - see below
  - getValueType - scalar, vector, mixedVector (nedelec)
  - project and axpy - PackedField does not support these

FieldOf
- pure virtual
  - does not implement Field::get{Value|Shape}Type
- parent of MatrixField, MixedVectorField, ScalarField, VectorField
- templated on field data type
  - explicitly instantiated in apfFieldOf.cc for Matrix3x3, double, Vector3
- adds {set|get}NodeValue
- implements Field::project and Field::axpy by calling wrapper version defined
  in apfFieldOf.cc

{Scalar|Vector|Matrix3x3}Field
- can be instaniated 
- derived from FieldOf<{double|Vector3|Matrix3x3}>
- implements getElement, countComponents, Field::get{Value|Shape}Type

PackedField
- can be instaniated
- contains multiple field values instead of a single double, vector3, etc. in a
  user defined layout that is not known to the class
  - we had a CFD applicaton whose native storage was seven or so doubles per
    mesh vertex that defined velocity (x3), pressure (x1), temperature (x1),
    etc.
  - doc string: 
    > Create a field of N components without a tensor type.
    > Packed fields are used to interface with applications whose fields are 
    > not easily categorized as tensors of order 0,1,2. They contain enough 
    > information to interpolate values in an element and such, but some
    > higher-level functionality is left out.
- implements Field::project and Field::axpy - error if called
- adds private member 'components'
- storage is `double`

FieldData
- pure virtual
- parent of FieldDataOf <- CoordData, UserData, TagDataOf


EntityShape
FieldShape

### Field class functions
getElement - creates an 'Element' instance
getValueType - returns integer that maps to
 - Scalar (1 dof/node)
 - Vector (3 dofs/node)
 - Mixed  (??? dofs/node) - the comment in apfField.h is not clear
getShapeType - returns integer that maps to 'Regular' or something else - need
to look into this
getScalarType - always returns 'Mesh::DOUBLE' - can we modernize this with
                (C++17) type traits (https://www.internalpointers.com/post/quick-primer-type-traits-modern-cpp)
getData - returns instance of 'FieldDataOf<double>'
project(Field* from) - pure virtual, returns void, ?
axpy(double a, Field* from)
 - used in field projection routine
 - constant times a vector plus a vector (see BLAS 'daxpy')
 - apfFieldOf.cc defines class Axpy (derived from FieldOp)
   - inputs:
     - two FieldOf pointers
     - a 'MeshEntity' (this is likely something specific to Fields)

### EntityShape class functions - apfShape.h

Shape functions over an entity (called 'element' in the code).

Note, nedelec specific functions ommitted.

getValues - evaluate shape function at parametric location within a mesh entity
getLocalGradients - evaluate shape function gradients at parametric location within a mesh entity
countNodes - nodes on the specified entity
alignSharedNodes - convert from shared node order to local/canonical entity order - longer description in code

### FieldShape class functions - apfShape.h

Describes field distribution and shape functions.  Typically singletons, one for each shape function scheme (lagrange, serendipity, etc.)

Note, nedelec specific functions ommitted. This is a 'vector' based shape function.

getEntityShape - get an EntityShape object for a given entity type (vtx, edge, face, region)
hasNodesIn - check if there are nodes associated with the given entity type
countNodesOn - return the number of nodes associated with the given entity type
getOrder - get polynomial order of shape function scheme
getNodeXi - get the parametric coordinates of a given node for a given entity type
getNodeTangent - get the tangent vector of a node for a given entity type
