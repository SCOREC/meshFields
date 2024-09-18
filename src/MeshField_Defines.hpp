#ifndef MESHFIELD_DEFINES_H
#define MESHFIELD_DEFINES_H
namespace MeshField {
using Real = double;
using LO = int;
enum Mesh_Topology {
 Vertex,
 Edge,
 Triangle,
 Quad,
 Tetrahedron,
 Hexahedron,
 Prism,
 Pyramid
};
}
#endif
