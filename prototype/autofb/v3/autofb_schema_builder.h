#include "autofb_schema.h"

#include "type_registry.h"

namespace autofb
{
  class schema_builder
  {
  public:
    schema_builder(std::string ns, const typesys::type_registry& tr = typesys::type_registry::instance())
      : descriptor{ ns }, tr{ tr }
    {
      // should we do the build-out on construction?
    }

    fbs_data build_idl();

    inline schema build()
    {
      return schema{ descriptor, build_idl().to_binary() };
    }

  private:
    const typesys::type_descriptor* find_type(const typesys::erased_type& type);

    schema_descriptor descriptor;
    const typesys::type_registry& tr;
  };
}