#include <opoly1d.h>
#include "main.h"

int main(int argc, const char *argv[])
{
  std::cout << HEADER << std::endl;

  // ensure the correct number of parameters are used.
  if (argc < 3)
  {
    std::cout << USAGE << std::endl;
    return 1;
  }

  JacobiPolynomials J;
  auto N = 100;
  auto k = 15;

  auto [x,w] = J.gauss_quadrature(N);
  auto V = J.eval(x, range(k));

  //plt.plot(x, V[:,:k])
  //plt.show()

  return 0;
}
