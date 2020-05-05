#include <numeric>
#include "types.h"

using namespace TODO_REPLACE_WITH_EIGEN;

IntList np::arange(int k)
{
  IntList r(k);
  std::iota(r.begin(), r.end(), 0);
  return r;
}
