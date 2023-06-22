#include "../base.h"
#include "../autofb_schema.h"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/reflection.h"
#include "flatbuffers/reflection_generated.h"

#include "test.h"

#include "autofb_test_shared.h"

#include <unordered_map>

using namespace typesys;
using namespace autofb::test;

namespace expected
{

template <typename T>
struct builder_i;

template <typename T, template <typename> typename B>
struct builder_owner : B<T>
{
private:
  T _prototype;

public:
  builder_owner() : B<T>(_prototype)
  {}

  builder_owner(T&& prototype) : _prototype(prototype), B<T>(_prototype)
  {}

  builder_owner(builder_owner& other) : _prototype(other._prototype), B<T>(_prototype)
  {}

  builder_owner(builder_owner&& other) : _prototype(other._prototype), B<T>(_prototype)
  {}

  T&& build()
  {
    return std::move(_prototype);
  }
};

template <typename T>
using builder = builder_owner<T, builder_i>;

struct property
{
  std::string name;
  std::string flatbuffer_type;

  bool is_vector;
  bool is_optional;

  property(std::string name, std::string flatbuffer_type, bool is_vector = false, bool is_optional = false)
    : name(name)
    , flatbuffer_type(flatbuffer_type)
    , is_vector(is_vector)
    , is_optional(is_optional)
  {
  }
};

template <>
struct builder_i<expected::property>
{
private:
  expected::property& prototype;

public:
  builder_i(expected::property& prototype) : prototype(prototype)
  {
  }

  builder_i& vector(bool is_vector = true)
  {
    prototype.is_vector = is_vector;
  }

  builder_i& optional(bool is_optional = true)
  {
    prototype.is_optional = is_optional;
  }
};

struct table
{
  std::string name;
  std::vector<expected::property> properties;

  table(std::string name, std::vector<expected::property> properties)
    : name(name)
    , properties(properties)
  {
  }

  table(std::string name)
    : name(name)
  {
  }
};

template <>
struct builder_i<expected::table>
{
private:
  expected::table& prototype;

public:
  builder_i(expected::table& prototype) : prototype(prototype)
  {
  }

  inline builder_i<expected::property> add_property(std::string name, std::string flatbuffer_type)
  {
    prototype.properties.emplace_back(name, flatbuffer_type);
    return builder_i<expected::property>(prototype.properties.back());
  }
};

using namespace_t = std::vector<expected::table>;

struct schema
{
  std::unordered_map<std::string, expected::namespace_t> namespaces;
};

template <>
struct builder_i<expected::schema>
{
private:
  expected::schema& prototype;

public:
  builder_i(expected::schema& prototype) : prototype(prototype)
  {
  }

  inline builder_i<expected::table> add_table(std::string namespace_name, std::string name)
  {
    prototype.namespaces[namespace_name].emplace_back(name);
    return builder_i<expected::table>(prototype.namespaces["autofb_proto"].back());
  }
};



} // namespace expected

template <typename T>
struct validator;

template <>
struct validator<reflection::Field>
{
  static bool are_equal(const reflection::Field& field, expected::property& expected, const reflection::Schema& context)
  {
    // look up the type from context by the type index
    //context.objects()->Get(field.type()->index())->name()->str();

    //return p.name == f.name()->str() &&
    //       p.flatbuffer_type == f.type()->name()->str() &&
    //  p.is_vector == f.is_vector() &&
    //  p.is_optional == f.is_optional();
    return true;
  }
};

std::string GetFbTypeName(const reflection::Schema* context, const reflection::Type* type)
{
  // check the base type
  switch (type->base_type())
  {

    /*
    UType
  
    */
    default:
    case reflection::BaseType::None:
    case reflection::BaseType::UType:
      return "<ERROR-TYPE>";
    
    case reflection::BaseType::Bool:
      return "bool";

    case reflection::BaseType::Byte:
      return "byte";

    case reflection::BaseType::UByte:
      return "ubyte";

    case reflection::BaseType::Short:
      return "short";

    case reflection::BaseType::UShort:
      return "ushort";

    case reflection::BaseType::Int:
      return "int";

    case reflection::BaseType::UInt:
      return "uint";

    case reflection::BaseType::Long:
      return "long";

    case reflection::BaseType::ULong :
      return "ulong";

    case reflection::BaseType::Float :
      return "float";

    case reflection::BaseType::Double :
      return "double";


    case reflection::BaseType::String :
      return "string";

    case reflection::BaseType::Vector :
      return "vector";

    case reflection::BaseType::Obj :
      return "obj";

    case reflection::BaseType::Union :
    case reflection::BaseType::Array :
      return "<ERROR-TYPE>";
  }
}

void SmokeTestAutoFbSchema()
{
//std::cout << "Who needs main() anyways?" << std::endl;

  // test the mechanism to generate a binary schema, then load it using "reflection"
  // and print it out
  const char* fb_schema = R"(namespace autofb_proto;

table baseV1 {
b : uint8;
}
)";



  auto fbs = autofb::fbs_data{ fb_schema };
  auto bfbs = fbs.to_binary();

  // const reflection::Schema* get() const
  //   {
  //     return reflection::GetSchema(schema_data.binary_data.c_str());
  //   }

  const reflection::Schema& schema = *reflection::GetSchema(bfbs.binary_data.c_str());

  // iterate through the types in the BFBS type schema and print them
  auto types = schema.objects();
  for ( size_t i = 0; i < types->Length(); i++ ) {
    const reflection::Object* type = types->Get( i );
    const char* type_name = type->name()->c_str();
    std::cout << "type: " << type_name << std::endl;

    auto type_table = types->LookupByKey(type_name);

    // iterate through the properties and print them
    auto fields = type_table->fields();

    for (size_t j = 0; j < fields->Length(); j++)
    {
      auto field = fields->Get(j);
      std::cout << "  -> " << field->name()->str() << ": " << GetFbTypeName(&schema, field->type()) << std::endl;
    }

    std::cout << std::endl;
  }
}

RUN_ONCE({
  //TestAutoFbSchemaGeneration();
});
//CALL_ONCE(test_autofb_binary_schemagen);
