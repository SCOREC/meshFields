#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double intFunc(const Vector &x) {
  return x[0];
}

int main(int argc, char *argv[]) {
  Mesh mesh = Mesh::MakeCartesian3D(100, 100, 100, Element::TETRAHEDRON, 100, 100, 100, false);
  //6 million element mesh
  H1_FECollection fec(1, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  
  FunctionCoefficient custom(intFunc);

  LinearForm b(&fespace);

  b.AddDomainIntegrator(new DomainLFIntegrator(custom)); 

  b.Assemble(); 
  cout << b.Sum() << endl;
  return 0;
}
