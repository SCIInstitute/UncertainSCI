#ifndef SCI_UNCERTAINSCI_TYPES_H
#define SCI_UNCERTAINSCI_TYPES_H

#include <variant>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace UncertainSCI
{
  using Vector1D = Eigen::Matrix<double, Eigen::Dynamic, 1>;
  using IntList = std::vector<int>;
  using Matrix2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Matrix3D = std::vector<Matrix2D>;
  using XType = std::variant<int, double, Vector1D>;

  class np
  {
  public:
    static IntList arange(int k);
    static Matrix2D ones(int r, int c);
  };
}

namespace UncertainSCI
{
  using Function1D = std::function<double(double)>;
}

#define ERROR_NOT_IMPLEMENTED throw __FUNCTION__ + std::string(" in ") + __FILE__ + ":" + std::to_string(__LINE__);

#endif
