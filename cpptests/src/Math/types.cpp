#include <numeric>
#include "types.h"

using namespace UncertainSCI;

IntList np::arange(int k)
{
  IntList r(k);
  std::iota(EIGEN_BEGIN_END(r), 0.0);
  return r;
}

Matrix2D np::ones(int r, int c)
{
  Matrix2D m(r, c);
  m.fill(1);
  return m;
}

IntList np::arange(int start, int end)
{
  IntList r(end - start);
  std::iota(EIGEN_BEGIN_END(r), static_cast<double>(start));
  return r;
}

std::vector<int> np::range(int start, int end)
{
  std::vector<int> r(end - start);
  std::iota(r.begin(), r.end(), start);
  return r;
}
