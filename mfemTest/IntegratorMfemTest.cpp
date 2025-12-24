#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double intFunc(const Vector &x) {
  return x[0];
}

int main(int argc, char *argv[]) {
  
  OptionsParser args(argc, argv);
  int order = 1;
  int size = 1;
  int numberOfElements = 1;
  int type = 1;
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&size, "-s", "--size", "Size of mesh");
  args.AddOption(&numberOfElements, "-n", "--numElem", "number of elements in each direction");
  args.AddOption(&type, "-t", "--type", "type of element(1 for tet, 0 for triangle");
  args.ParseCheck();
  
  Mesh mesh = Mesh::MakeCartesian3D(numberOfElements, numberOfElements, type == 1 ? numberOfElements : 1, type == 1 ? Element::TETRAHEDRON : Element::TRIANGLE, size, size, type == 1 ? size : 1, false);
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  
  FunctionCoefficient custom(intFunc);

  LinearForm b(&fespace);

  b.AddDomainIntegrator(new DomainLFIntegrator(custom)); 

  b.Assemble(); 
  cout << b.Sum() << endl;
  return 0;
}
