#ifndef MESHFIELD_DEFINES_H
#define MESHFIELD_DEFINES_H
namespace MeshField {
using Real8 = double;
using Real4 = double;
using Real = Real8; //a sane default
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
