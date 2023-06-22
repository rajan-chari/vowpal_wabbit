#include "vw/serialization/autofb_schema.h"
#include "vw/serialization/autofb_serializer.h"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/bfbs_generator.h"
#include "flatbuffers/util.h"

#include <sstream>

#pragma region autofb_schema
namespace autofb
{
  using type_map = std::unordered_map<std::type_index, std::string>;
  static const type_map flatbuffers_typemap = {
    { typeid(std::int8_t), "int8" },
    { typeid(std::int16_t), "int16" },
    { typeid(std::int32_t), "int32" },
    { typeid(std::uint8_t), "uint8" },
    { typeid(std::uint16_t), "uint16" },
    { typeid(std::uint32_t), "uint32" },
    { typeid(std::uint64_t), "uint64" },
    { typeid(float), "float" },
    { typeid(double), "double" },
    { typeid(std::string), "string" },
    { typeid(bool), "bool" },
  };

  bfbs_data fbs_data::to_binary()
  {
    flatbuffers::IDLOptions opts;

    flatbuffers::Parser parser;
    if (!parser.Parse(text_data.c_str()))
    {
      // TODO: ERROR!
      std::string last_error = "Failed to parse schema: " + parser.error_;
      std::cout << last_error << std::endl << std::endl;
      std::cout << text_data << std::endl;

      throw new std::exception(last_error.c_str());
    }

    parser.Serialize();

    auto buf = parser.builder_.GetBufferPointer();
    auto size = parser.builder_.GetSize();

    // The assertion here is simply to flag a situation in which the fb API changes somehow so that its
    // buffer representation is no longer conveniently put into a std::string. (We should probably be
    // using vectors instead, honestly, but this is based on the reflection examples.)
    static_assert(
        sizeof(std::remove_pointer_t<decltype(buf)>) == sizeof(char), "fb buffer type is not the same as char*");

    std::string result;
    result.resize(size);

    memcpy_s(&result[0], result.size(), buf, size);

    return {result};
  }

  const typesys::type_descriptor* schema_builder::find_type(const typesys::erased_type& type)
  {
    typesys::type_registry::type_iter it = tr.find_type(type.tindex);
    if (it == tr.types_end())
    {
      return nullptr;
    }
    
    return it._Ptr;
  }

  fbs_data schema_builder::build_idl()
  {
    using std::endl;
    
    std::stringstream idl;

    idl << "namespace " << this->descriptor.schema_namespace << ";" << endl << endl;

    // TODO: Support dependent schemas, but for now, single-file

    std::for_each(this->tr.types.begin(), this->tr.types.end(),
    [this, &idl](const typesys::type_descriptor& ti)
    {
      if (ti.properties.size() == 0) { return; }

      idl << "table " << ti.name << " {" << endl;

      std::for_each(ti.properties.begin(), ti.properties.end(),
      [this, &idl](const typesys::property_descriptor& pi)
      {
        idl << "  ";
        idl << pi.name << ":";

        // TODO: Map?!
        auto* maybe_pti = this->find_type(pi.eftype.evalue);
        if (!maybe_pti)
        {
          // TODO: commented out line for debugging?
          return;
        }

        auto& pti = *maybe_pti;
        if (pi.eftype.is_vector()) { idl << "["; }

        if (this->tr.is_builtin(pti.etype))
        {
        // std::string fb_name = flatbuffers_typemap.at(pti.etype.tindex);
          idl << flatbuffers_typemap.at(pti.etype.tindex);
        }
        else
        {
          // TODO: type_import?
          idl << pti.name;
        }
        if (pi.eftype.is_vector()) { idl << "]"; }

        idl << ";" << endl;
      });

      idl << "}" << endl << endl;
    });

    return fbs_data{idl.str()};
  }
}
#pragma endregion

#pragma region autfb_serializer
namespace autofb
{
  // TODO: Port the prototype code to here
}
#pragma endregion