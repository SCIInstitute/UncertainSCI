#include <numeric>
#include "types.h"

using namespace UncertainSCI;

IntList np::arange(int k)
{
  IntList r(k);
  std::iota(r.begin(), r.end(), 0);
  return r;
}

Matrix2D np::ones(int r, int c)
{
  Matrix2D m(r, c);
  m.fill(1);
  return m;
}
