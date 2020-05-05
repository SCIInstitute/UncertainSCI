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

  auto alpha = 0.0;
  auto beta = 0.0;
  auto N = 100;
  auto k = 15;

  auto ab = jacobi_recurrence_values(N, alpha, beta);

  auto [x,w] = gauss_quadrature_driver(ab, N);
  auto V = eval_driver(x, np.arange(k), 0, ab);

  //plt.plot(x, V[:,:k])
  //plt.show()

  return 0;
}
