#include <division.h>
#include "main.h"

int main(int argc, const char *argv[])
{
  std::cout << HEADER;

  // ensure the correct number of parameters are used.
  if (argc < 3) {
    std::cout << USAGE << std::endl;
    return 1;
  }
  //
  // f.numerator = atoll(argv[1]);
  // f.denominator = atoll(argv[2]);
  //
  // Division d = Division(f);
  // try {
  //   DivisionResult r = d.divide();
  //
  //   std::cout << "Division : " << f.numerator << " / " << f.denominator << " = " << r.division << "\n";
  //   std::cout << "Remainder: " << f.numerator << " % " << f.denominator << " = " << r.remainder << "\n";
  // } catch (DivisionByZero) {
  //   std::cout << "Can not divide by zero, Homer. Sober up!\n";
  // }
  return 0;
}
