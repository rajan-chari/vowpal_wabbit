#include "../base.h"
#include "../autofb_schema.h"

#include "test.h"

#include "autofb_test_shared.h"

using namespace typesys;
using namespace autofb::test;

RUN_ONCE({
  std::cout << "Who needs main() anyways?" << std::endl;



  //std::cout << "test bfbs" << std::endl;

  const type_registry& tr = test_types.get();

  // print the number of known types in tr
  std::cout << "Known Types: " << tr.types.size() << std::endl;

  //autofb::test::print_known_types();
});
//CALL_ONCE(test_autofb_binary_schemagen);
