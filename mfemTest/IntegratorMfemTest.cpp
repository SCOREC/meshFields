#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

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
  int runs = 1;
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&size, "-s", "--size", "Size of mesh");
  args.AddOption(&numberOfElements, "-n", "--numElem", "number of elements in each direction");
  args.AddOption(&type, "-t", "--type", "type of element(1 for tet, 0 for triangle");
  args.AddOption(&runs, "-r", "--runs", "number of times to run test");
  args.Parse();
  
  if (type == 1) {
    Mesh mesh = Mesh::MakeCartesian3D(numberOfElements, numberOfElements, numberOfElements, Element::TETRAHEDRON, size, size, size);
    H1_FECollection fec(order, mesh.Dimension());
    FiniteElementSpace fespace(&mesh, &fec);
  
    FunctionCoefficient custom(intFunc);

    for (int i = 0; i < runs; ++i) {
    	LinearForm b(&fespace);

    	b.AddDomainIntegrator(new DomainLFIntegrator(custom)); 
	auto start = std::chrono::high_resolution_clock::now();
    	b.Assemble(); 
    	auto end = std::chrono::high_resolution_clock::now();
    	cout << "RESULT: " << mfem::GetGitStr() << "," << mesh.GetNE() << "," << order << "," << b.Sum() << "," << std::chrono::duration<double>(end - start).count() << endl;
    }
  }
  else { 
    Mesh mesh = Mesh::MakeCartesian2D(numberOfElements, numberOfElements, Element::TRIANGLE, size, size);
    H1_FECollection fec(order, mesh.Dimension());
    FiniteElementSpace fespace(&mesh, &fec);
  
    FunctionCoefficient custom(intFunc);

    for (int i = 0; i < runs; ++i) {
    	LinearForm b(&fespace);

    	b.AddDomainIntegrator(new DomainLFIntegrator(custom)); 
	auto start = std::chrono::high_resolution_clock::now();
    	b.Assemble(); 
    	auto end = std::chrono::high_resolution_clock::now();
    	cout << "RESULT: " << mfem::GetGitStr() << "," << mesh.GetNE() << "," << order << "," << b.Sum() << "," << std::chrono::duration<double>(end - start).count() << endl;
    }
  }
  return 0;
}
