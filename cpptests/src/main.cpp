#include "opoly1d.h"
#include "families.h"
#include "main.h"
#include <fstream>

using namespace UncertainSCI;

int main(int argc, const char *argv[])
{
  std::cout << HEADER << std::endl;

  // ensure the correct number of parameters are used.
  // if (argc < 3)
  // {
  //   std::cout << USAGE << std::endl;
  //   return 1;
  // }

  auto alpha = 0.0;
  auto beta = 0.0;
  auto N = 100;
  auto k = 15;

  try
  {
    auto ab = Families::jacobi_recurrence_values(N, alpha, beta);

    auto [x,w] = OPoly1D::gauss_quadrature_driver(ab, N);
    auto V = OPoly1D::eval_driver(x, np::arange(k), 0, ab);

    if (V.size() < 1)
    {
      std::cerr << "Error computing V" << std::endl;
      return 1;
    }

    std::cout << "Completed successfully: returned x(" << x.rows() << "x" << x.cols() << ") and V_0("
      << V[0].rows() << "x" << V[0].cols() << ")." << std::endl;

    std::ofstream o("x.txt");
    o << std::setprecision(16) << x;
    std::ofstream v("v.txt");
    v << std::setprecision(16) << V[0];



    //plt.plot(x, V[:,:k])
    //plt.show()
  }
  catch (const char* e)
  {
    std::cerr << "Caught error: " << e << std::endl;
    return 1;
  }
  catch (const std::string& e)
  {
    std::cerr << "Caught error: " << e << std::endl;
    return 1;
  }

  return 0;
}
