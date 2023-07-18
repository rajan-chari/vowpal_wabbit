#pragma once

#include "vw/serialization/base.h"
#include "vw/serialization/type_registry.h"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/reflection.h"
#include "flatbuffers/reflection_generated.h"
#include "flatbuffers/util.h"

namespace autofb 
{

  struct bfbs_data
  {
    std::string binary_data;
  };

  struct fbs_data
  {
    std::string text_data;

    bfbs_data to_binary();
  };

  struct schema_descriptor
  {
    std::string schema_namespace; // the default namespace

    // TODO: version and inheritance?
    // dependent schemas?

    inline std::string make_qualified_name(std::string type_name) const 
    {
      return schema_namespace + "." + type_name;
    }
  };

  class schema
  {
  public:
    schema(schema_descriptor descriptor, bfbs_data schema_data) 
      : descriptor{ descriptor }, schema_data{ schema_data }
    {
    }

    schema(std::string default_namespace, std::string path) 
      : descriptor{ default_namespace }
    {
      if (flatbuffers::LoadFile(path.c_str(), true, &schema_data.binary_data))
      {
      }
      else
      {
        // TODO: Error out?
      }
    }

    const schema_descriptor descriptor;

    const reflection::Schema* get() const
    {
      return reflection::GetSchema(schema_data.binary_data.c_str());
    }

  private:
    bfbs_data schema_data;
  };

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