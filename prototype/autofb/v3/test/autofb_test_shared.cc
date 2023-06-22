#include "autofb_test_shared.h"

namespace autofb { namespace test {
  ::test::let<types_registrator, type_registry> test_types;

  void print_known_types()
  {
    const type_registry& tr = test_types.get();

    std::cout << "Printing Known Types" << std::endl;

    for (const auto& td : tr.types)
    {
      std::cout << "Type: " << td.name << std::endl;
      for (const auto& prop : td.properties)
      {
        std::cout << "  Property: " << prop.name << std::endl;
      }
    } 
  }

  RUN_ONCE({
    print_known_types();
  });
}}
