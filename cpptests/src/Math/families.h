#ifndef SCI_UNCERTAINSCI_FAMILIES_H
#define SCI_UNCERTAINSCI_FAMILIES_H

#include "types.h"

namespace UncertainSCI
{
  class Families
  {
  public:
    static Matrix2D jacobi_recurrence_values(int N, double alpha, double beta);
  };

}

#endif
