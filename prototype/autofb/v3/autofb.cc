#include "autofb_schema.h"
#include "autofb_schema_builder.h"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/bfbs_generator.h"
#include "flatbuffers/util.h"

#include <sstream>

namespace autofb
{
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
}

namespace autofb
{
  const typesys::type_descriptor* schema_builder::find_type(const typesys::erased_type& type)
  {
    typesys::type_registry::type_iter it = tr.find_type(type.tindex);
    if (it != tr.types_end())
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

        // TODO: Map?!
        auto* maybe_pti = this->find_type(pi.eftype.evalue);
        if (!maybe_pti)
        {
          return;
        }

        auto& pti = *maybe_pti;

        if (pi.eftype.is_vector()) { idl << "["; }

        idl << pi.name << ":" << pti.name << ";";

        if (pi.eftype.is_vector()) { idl << "]"; }

        idl << endl;
      });

      idl << "}" << endl << endl;
    });

    return fbs_data{idl.str()};
  }
}