## References

- pumi users guide: https://www.scorec.rpi.edu/pumi/PUMI.pdf
- pumi TOMS paper: https://www.scorec.rpi.edu/REPORTS/2015-4.pdf
- pumi apf source code
- pumi doxygen: https://www.scorec.rpi.edu/pumi/doxygen/

## Design Ideas

- don't want dependency on omegah
- have omegah implement the interface outside of meshfields

## Questions

- Do we want/need to support the following features?
  - non-uniform P
  - Nedelec shape functions
  - mixed meshes
  - polygonal meshes - i.e., wachpress shape functions for seaice
- How will these fields interact with mesh adaptation?
  - convert back to mesh library native format?
  - can omegah adapt use these operations for cavity based field transfer
    without invasive changes? Do we define 'cavity' specific field operations?

## Functions

- terminology
  - dof - exists at a dof holder, can be scalar, vector, matrix, etc.
  - dof holder
    - can contain a dof
    - possible holder: mesh entities, quadrature points, element centroids, etc.
  - node - a location on a mesh entitiy that is a dof holder, multiple nodes can
           exist per mesh entity
- storage - scalar, vector, matrix per dof
- field
  - requires shape, ownership, and mesh association
- shape - defines the node distribution where the coordinates and field values (DOF’s) are stored
  - pumi supports the following 
    - name, order, notes
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
             - **how does this work with the shape function definitions? axpy is part of it IIUC**
- parallel - distributed memory (i.e., multiple GPUs)
  - ownership - which process owns each node, can be defined via function (a protocol) or array of ints (one for each node)
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

### Classes:

Field (derived from Field) - ?
FieldBase - ?
Element - this support integration etc. and is not just mesh topology
FieldData - 
FieldDataOf (derived from FieldData) - templated on type, which seems to be hardcoded to double
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