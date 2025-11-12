#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double intFunc(const Vector &x) {
  return x[0]; 
}

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();
   Mesh mesh(mesh_file);

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
  
   FunctionCoefficient custom(intFunc);

   LinearForm b(&fespace);

   b.AddDomainIntegrator(new DomainLFIntegrator(custom)); 

   b.Assemble(); 

   cout << b.Sum() << endl;
   return 0;
}
