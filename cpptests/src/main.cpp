#include <iostream>
#include "opoly1d.h"
#include "families.h"
#include "main.h"

using namespace UncertainSCI;

int main(int argc, const char *argv[])
{
  std::cout << HEADER << std::endl;

  // ensure the correct number of parameters are used.
  if (argc < 3)
  {
    std::cout << USAGE << std::endl;
    return 1;
  }

  auto alpha = 0.0;
  auto beta = 0.0;
  auto N = 100;
  auto k = 15;

  auto ab = Families::jacobi_recurrence_values(N, alpha, beta);

  auto [x,w] = OPoly1D::gauss_quadrature_driver(ab, N);
  auto V = OPoly1D::eval_driver(x, np::arange(k), 0, ab);

  //plt.plot(x, V[:,:k])
  //plt.show()

  return 0;
}
