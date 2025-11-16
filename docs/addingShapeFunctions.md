In order to add shape function, you must create a struct in
https://github.com/SCOREC/meshFields/blob/main/src/MeshField_Shape.hpp.

You must define numNodes, meshEntDim, and Order as const variables as well as the functions
getValues and getLocalGradients. You then need to define a mapping from how you store the
nodes to the ordering of the shape function defined earlier. The Omegah mappings are in
https://github.com/SCOREC/meshFields/blob/main/src/MeshField.hpp#L63. In the Jacobian
tests we also an identity mapping defined as another example:
https://github.com/SCOREC/meshFields/blob/main/test/testElementJacobian2d.cpp#L12.

In order to use integration with the added shape function it must be added here:
https://github.com/SCOREC/meshFields/blob/main/src/MeshField_Integrate.hpp#L44.

Finally, when trying to compute an integral using the shape function, a class derived from the
class Integrator (defined here:
https://github.com/SCOREC/meshFields/blob/main/src/MeshField_Integrate.hpp#L204) must be
declared and should implement the atPoints function. An example of this can be found here:
https://github.com/SCOREC/meshFields/blob/main/test/testCountIntegrator.cpp#L41.
