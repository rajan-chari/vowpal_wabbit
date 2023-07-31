#include "vw/serialization/autofb_schema.h"
#include "vw/serialization/autofb_serializer.h"
#include "vw/serialization/type_registry.h"
#include "vw/serialization/type_erase.h"
#include "vw/serialization/type_activation.h"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
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

      throw new std::runtime_error(last_error.c_str());
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

    //memcpy_s(&result[0], result.size(), buf, size);
    memcpy(&result[0], buf, size);

    return {result};
  }

  const typesys::type_descriptor* schema_builder::find_type(const typesys::erased_type& type)
  {
    typesys::type_registry::type_iter it = tr.find_type(type.tindex);
    if (it == tr.types_end())
    {
      return nullptr;
    }
    const typesys::type_descriptor* ret_val = it->has_base_type() ? it->get_base_type() : &*it;
    return ret_val;
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

struct fbb_AddElement_dispatcher
{
  #define __FBB_ADD_ELEMENT_DISPATCHER_PACK flatbuffers::FlatBufferBuilder&, const reflection::Field&, typesys::erased_lvalue_ref&
  using dispatch_f = typesys::dispatch_f<__FBB_ADD_ELEMENT_DISPATCHER_PACK>;
  using dispatch_table = typesys::erased_dispatch_table<__FBB_ADD_ELEMENT_DISPATCHER_PACK>;

  fbb_AddElement_dispatcher()
  {
    dt.add<std::int8_t, &fbb_AddElement_dispatcher::dispatcher<std::int8_t>>()
      .add<std::int16_t, &fbb_AddElement_dispatcher::dispatcher<std::int16_t>>()
      .add<std::int32_t, &fbb_AddElement_dispatcher::dispatcher<std::int32_t>>()
      .add<std::int64_t, &fbb_AddElement_dispatcher::dispatcher<std::int64_t>>()
      .add<std::uint8_t, &fbb_AddElement_dispatcher::dispatcher<std::uint8_t>>()
      .add<std::uint16_t, &fbb_AddElement_dispatcher::dispatcher<std::uint16_t>>()
      .add<std::uint32_t, &fbb_AddElement_dispatcher::dispatcher<std::uint32_t>>()
      .add<std::uint64_t, &fbb_AddElement_dispatcher::dispatcher<std::uint64_t>>()
      .add<float, &fbb_AddElement_dispatcher::dispatcher<float>>()
      .add<double, &fbb_AddElement_dispatcher::dispatcher<double>>()
      .add<bool, &fbb_AddElement_dispatcher::dispatcher<bool>>();
  }

  void operator()(flatbuffers::FlatBufferBuilder& fbb, const reflection::Field& field, typesys::erased_lvalue_ref& value)
  {
    dt.dispatch(value._type, fbb, field, value);
  }

private:
  dispatch_table dt;

  template <typename T>
  static void dispatcher(flatbuffers::FlatBufferBuilder& fbb, const reflection::Field& field, typesys::erased_lvalue_ref& erased_lvalue)
  {
    fbb.AddElement<T>(field.offset(), erased_lvalue.get<T>());
  }
};

struct fbb_ReadElement_dispatcher
{
  #define __FBB_READ_ELEMENT_DISPATCHER_PACK const flatbuffers::Table&, const reflection::Field&, typesys::erased_lvalue_ref&
  using dispatch_f = typesys::dispatch_f<__FBB_READ_ELEMENT_DISPATCHER_PACK>;
  using dispatch_table = typesys::erased_dispatch_table<__FBB_READ_ELEMENT_DISPATCHER_PACK>;

  fbb_ReadElement_dispatcher()
  {
    dt.add<std::int8_t, &fbb_ReadElement_dispatcher::dispatcher<std::int8_t>>()
      .add<std::int16_t, &fbb_ReadElement_dispatcher::dispatcher<std::int16_t>>()
      .add<std::int32_t, &fbb_ReadElement_dispatcher::dispatcher<std::int32_t>>()
      .add<std::int64_t, &fbb_ReadElement_dispatcher::dispatcher<std::int64_t>>()
      .add<std::uint8_t, &fbb_ReadElement_dispatcher::dispatcher<std::uint8_t>>()
      .add<std::uint16_t, &fbb_ReadElement_dispatcher::dispatcher<std::uint16_t>>()
      .add<std::uint32_t, &fbb_ReadElement_dispatcher::dispatcher<std::uint32_t>>()
      .add<std::uint64_t, &fbb_ReadElement_dispatcher::dispatcher<std::uint64_t>>()
      .add<float, &fbb_ReadElement_dispatcher::dispatcher<float>>()
      .add<double, &fbb_ReadElement_dispatcher::dispatcher<double>>()
      .add<bool, &fbb_ReadElement_dispatcher::dispatcher<bool>>();
  }

  void operator()(const flatbuffers::Table& table, const reflection::Field& field, typesys::erased_lvalue_ref& value)
  {
    dt.dispatch(value._type, table, field, value);
  }

private:
  dispatch_table dt;

  template <typename T>
  static void dispatcher(const flatbuffers::Table& table, const reflection::Field& field, typesys::erased_lvalue_ref& erased_lvalue)
  {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

    if (std::is_floating_point<T>::value)
    {
      erased_lvalue.set<T>(table.GetField<T>(field.offset(), field.default_real()));
    }
    else if (std::is_integral<T>::value)
    {
      erased_lvalue.set<T>(table.GetField<T>(field.offset(), field.default_integer()));
    }
    else
    {
      // TODO: this should never happen
      assert(false);
    }
  }
};

void add_flatbuffer_field(
  flatbuffers::FlatBufferBuilder& fbb,
  const reflection::Field& field,
  typesys::erased_lvalue_ref& value)
{
  static fbb_AddElement_dispatcher add_element_d;

  add_element_d(fbb, field, value);
}

void read_flatbuffer_field(
  const flatbuffers::Table& table,
  const reflection::Field& field,
  typesys::erased_lvalue_ref& value)
{
  static fbb_ReadElement_dispatcher read_element_d;

  read_element_d(table, field, value);
}

void read_flatbuffer_vector_builtin(
  const flatbuffers::VectorOfAny& source,
  const reflection::Schema& schema,
  const reflection::BaseType element_type,
  const typesys::type_descriptor& ti,
  typesys::erased_vector& espan)
{
  const uint8_t* begin;
  const uint8_t* end;

  // either a vector of strings or a vector of built-in scalars
    if (espan._type.is<std::string>())
    {
      std::vector<std::string> strings;

      for (auto i = 0; i < source.size(); i++)
      {
        strings.push_back(flatbuffers::GetAnyVectorElemS(&source, element_type, i));
      }

      begin = reinterpret_cast<const uint8_t*>(strings.data());
      end = reinterpret_cast<const uint8_t*>(begin + (sizeof(std::string)));
    }
    else
    {
      begin = source.Data();
      end = begin + (source.size() * espan._type.size );
    }

  if (begin != nullptr && end - begin > 0)
  {
    espan.assign_from(begin, end);
  }
}
void read_flatbuffer_vector_field(
  const flatbuffers::Table& container,
  const reflection::Schema& schema,
  const reflection::Field& field,
  const typesys::property_descriptor& pi,
  const typesys::type_descriptor& pti,
  typesys::erased_vector& espan)
{
  using namespace typesys;
  assert (pi.eftype.is_vector());

  const flatbuffers::VectorOfAny* maybe_vector = flatbuffers::GetFieldAnyV(container, field);
  if (maybe_vector == nullptr)
  {
    return;
  }

  const flatbuffers::VectorOfAny& vector = *maybe_vector;

  if (type_registry::instance().is_builtin(pti))
  {
    read_flatbuffer_vector_builtin(vector, schema, field.type()->base_type(), pti, espan);
  }
  else
  {
    //read_flatbuffer_vector_table();
  }
}

flatbuffers::uoffset_t serialize_flatbuffer_vector(
  flatbuffers::FlatBufferBuilder& fbb,
  const schema_descriptor& descriptor,
  const reflection::Schema& schema,
  const typesys::type_descriptor& ti,
  typesys::erased_vector& espan);

flatbuffers::uoffset_t serialize_flatbuffer_table(
  flatbuffers::FlatBufferBuilder& fbb, 
  const schema_descriptor& descriptor,
  const reflection::Schema& schema,
  const typesys::type_descriptor& ti,
  typesys::erased_lvalue_ref& value)
{
  using namespace typesys;
  assert(!type_registry::instance().is_builtin(ti)); //, "built-in type misinterpreted as table"

  std::string qname = descriptor.make_qualified_name(ti.name);
  auto maybe_object = schema.objects()->LookupByKey(qname.c_str());
  assert(maybe_object); // "table not found in schema"

  const reflection::Object& object = *maybe_object;

  std::unordered_map<name_t, flatbuffers::uoffset_t> offsets;

  // pass1: depth-first build the offsets for complex object (non-builtins, or strings)
  std::for_each(ti.properties.begin(), ti.properties.end(), 
    [&offsets, &fbb, &descriptor, &schema, &object, &value](auto& it)
    {
      auto& eftype = it.eftype;
      const type_descriptor* maybe_ti = nullptr;

      auto typeit = type_registry::instance().find_type(eftype.evalue.tindex);
      if (typeit != type_registry::instance().types_end())
      {
        maybe_ti = &(*typeit);
      }
      
      assert(maybe_ti != nullptr); //, "property type was not registered property (missing builtin?)"

      const type_descriptor& pti = *maybe_ti;

      // Filter out non-(vector, string, table)s (in other words, scalar, non-string built-ins) 
      // after caching the type lookup
      if (!eftype.is_vector() && 
          type_registry::instance().is_builtin(pti) && 
          !eftype.evalue.template is<std::string>()) 
      { return; }

      flatbuffers::uoffset_t offset;
      if (eftype.is_vector())
      {
        assert(eftype.evalue.e_vector_builder != nullptr);
        erased_vector espan = eftype.evalue.e_vector_builder();
        offset = serialize_flatbuffer_vector(fbb, descriptor, schema, pti, espan);
      }
      else if (type_registry::instance().is_builtin(pti))
      {
        // this property is a string
        erased_lvalue_ref* pvalue = nullptr;
        if(!it.binder.try_bind(value, pvalue))
        {
          //TODO: error condition
          return;
        }
        std::string str = pvalue->get<std::string>();
        offset = fbb.CreateString(str).o;
      }
      else
      {
        // this property is a table
        erased_lvalue_ref* pvalue = nullptr;
        if(!it.binder.try_bind(value, pvalue))
        {
          //TODO: error condition
          return;
        }
        auto inner_qname = descriptor.make_qualified_name(pti.name);
        offset = serialize_flatbuffer_table(fbb, descriptor, schema, pti, *pvalue);
      }

      offsets[it.name.c_str()] = offset;
    });

  // pass2: add the offsets and scalars to the table
  flatbuffers::uoffset_t table_offset = fbb.StartTable();

  std::for_each(ti.properties.begin(), ti.properties.end(), 
    [&offsets, &fbb, &schema, &object, &value](auto& it)
    {
      auto& eftype = it.eftype;
      
      // our second pass, means we know that the type is cached
      const type_descriptor& pti = *type_registry::instance().find_type(eftype.evalue.tindex);

      const reflection::Field* maybe_field = object.fields()->LookupByKey(it.name.c_str());
      if (maybe_field == nullptr)
      {
        // this property is not in the schema
        // TODO: error?
        return;
      }

      const reflection::Field& field = *maybe_field;

      auto offsetit = offsets.find(it.name.c_str());
      if (offsetit == offsets.end())
      {
        assert(type_registry::instance().is_builtin(pti));
        assert(!eftype.is_vector()); //, "builtin property cannot be a vector"
        erased_lvalue_ref* pvalue = nullptr;
        if(!it.binder.try_bind(value, pvalue))
        {
          //TODO: error condition
          return;
        }
        add_flatbuffer_field(fbb, field, *pvalue);
      }
      else
      {
        fbb.AddOffset(field.offset(), flatbuffers::Offset<void>(offsetit->second));
      }
    });

  table_offset = fbb.EndTable(table_offset);
  return table_offset;
}

flatbuffers::uoffset_t serialize_flatbuffer_vector(
  flatbuffers::FlatBufferBuilder& fbb,
  const schema_descriptor& descriptor,
  const reflection::Schema& schema,
  const typesys::type_descriptor& ti,
  typesys::erased_vector& espan)
{
  using namespace typesys;
  const static fbb_AddElement_dispatcher add_element_d;

  if (type_registry::instance().is_builtin(ti))
  {
    if (espan._type.is<std::string>())
    {
      return fbb.CreateVectorOfStrings(reinterpret_cast<std::string*>(espan.data()), reinterpret_cast<std::string*>(espan.data()) + espan.size()).o;
    }
    else
    {
      // this is a vector of scalars which are not strings
      uint8_t* buf;
      flatbuffers::uoffset_t result = fbb.CreateUninitializedVector(espan.size(), espan._type.size, &buf);
      espan.copy_to(reinterpret_cast<void*>(buf), espan.size() * espan._type.size);

      return result;
    }
  }
  else
  {
    std::vector<flatbuffers::uoffset_t> offsets;
    offsets.reserve(espan.size());

    uint8_t* buf = reinterpret_cast<uint8_t*>(espan.data());
    static_assert(sizeof(uint8_t) == 1, "our pointer type better be a single byte");

    const size_t advance = espan._type.size;

    for (size_t i = 0; i < espan.size(); i++)
    {
      void* pi = buf + i * advance;

      // we have to work using the one-level-higher indirect type
      // (void*) vs T&, which means 
      void** magic = reinterpret_cast<void**>(pi);

      erased_lvalue_ref elv (espan._type, ref(*magic));

      offsets.push_back(serialize_flatbuffer_table(fbb, descriptor, schema, ti, elv));
    }

    //espan.reduce<void>(f);

    fbb.StartVector(espan.size(), sizeof(flatbuffers::uoffset_t), flatbuffers::AlignOf<flatbuffers::uoffset_t>());
    for (int i = espan.size(); i > 0;)
    {
      fbb.PushElement(offsets[--i]);
    }
    return fbb.EndVector(espan.size());
  }
}

void read_flatbuffer_table(
  const flatbuffers::Table& source,
  const reflection::Schema& schema,
  const reflection::Object& table,
  const typesys::type_descriptor& ti,
  typesys::erased_lvalue_ref& target)
{
  using namespace typesys;

  std::for_each(ti.properties.begin(), ti.properties.end(), 
    [&schema, &table, &source, &target](auto& it)
    {
      auto& eftype = it.eftype;

      // cache the type lookup
      const type_descriptor* maybe_ti = nullptr;

      auto typeit = type_registry::instance().find_type(eftype.evalue.tindex);
      if (typeit != type_registry::instance().types_end())
      {
        maybe_ti = &(*typeit);
      }

      assert(maybe_ti != nullptr); //, "property type was not registered property (missing builtin?)"

      const type_descriptor& pti = *maybe_ti;

      //

      const reflection::Field* maybe_field = table.fields()->LookupByKey(it.name.c_str());
      if (maybe_field == nullptr)
      {
        // this property is not in the schema
        // TODO: Error
        return;
      }

      const reflection::Field& field = *maybe_field;

      if (eftype.is_vector())
      {
        flatbuffers::VectorOfAny* maybe_vector = flatbuffers::GetFieldAnyV(source, field);
        if (maybe_vector == nullptr)
        {
          // TODO: error?
          return;
        }

        erased_vector espan = eftype.evalue.e_vector_builder();

        read_flatbuffer_vector_field(source, schema, field, it, pti, espan);
      }
      else if (type_registry::instance().is_builtin(pti))
      {
        erased_lvalue_ref* pvalue = nullptr;
        if(!it.binder.try_bind(target, pvalue))
        {
          //TODO: error condition
          return;
        }

        if (eftype.evalue.template is<std::string>())
        {
          auto str = flatbuffers::GetFieldS(source, field);
          pvalue->set<std::string>(std::string(flatbuffers::GetString(str)));
        }
        else
        {
          read_flatbuffer_field(source, field, *pvalue);
        }
      }
      else
      {
        // table type
        assert(field.type()->base_type() == reflection::BaseType::Obj);

        erased_lvalue_ref* pvalue = nullptr;
        if(!it.binder.try_bind(target, pvalue))
        {
          //TODO: error condition
          return;
        }
        
        // get the underlying Object* representing the table info
        flatbuffers::Table* inner_source = flatbuffers::GetFieldT(source, field);
        const reflection::Object* inner_table = schema.objects()->Get(field.type()->index());
        read_flatbuffer_table(*inner_source, schema, *inner_table, pti, *pvalue);
      }
    });
  }

offset_of_any serializer::write_flatbuffer(
  flatbuffers::FlatBufferBuilder& fbb, 
  typesys::erased_lvalue_ref& target)
{
  // todo: it would be nice to enforce these known invariants at compile time
  // but for now, we'll just know they hold and assert
  using namespace typesys;
  auto it = type_registry::instance().find_type(target._type.tindex);
  assert(it != type_registry::instance().types_end()); //, "type not registered");

  const type_descriptor& ti = *it;
  assert(!type_registry::instance().is_builtin(ti)); //, "we cannot serialize built-ins directly; need to be in a table"

  auto maybe_schema = _schema.get();
  assert(maybe_schema); //, "schema not found for type - TODO: this is a real possible error, handle it");

  const reflection::Schema& schema = *maybe_schema;

  //print all of the names of the types in the schemas
  auto types = schema.objects();
  for ( size_t i = 0; i < types->Length(); i++ ) {
      const reflection::Object* type = types->Get( i );
      std::cout << "type: " << type->name()->str() << std::endl;
  }
  return offset_of_any{serialize_flatbuffer_table(fbb, _schema.descriptor, schema, ti, target)};
}

typesys::activation serializer::read_flatbuffer(
  const uint8_t* buf,
  typesys::erased_type target_type)
  {
    using namespace typesys;
    activation result = target_type.activator();
    void* res = result.get();
    erased_lvalue_ref target_ref {target_type, ref(res)};

    auto it = type_registry::instance().find_type(target_type.tindex);
    assert(it != type_registry::instance().types_end()); //, "type not registered");

    const type_descriptor& ti = *it;

    auto maybe_schema = _schema.get();
    assert(maybe_schema); //, "schema not found for type - TODO: this is a real possible error, handle it");

    const reflection::Schema& schema = *maybe_schema;

    std::string qname = _schema.descriptor.make_qualified_name(ti.name);
    auto maybe_table = schema.objects()->LookupByKey(qname.c_str());
    assert(maybe_table); //, "type not found in schema");

    const reflection::Object& table = *maybe_table;

    const flatbuffers::Table* maybe_source = flatbuffers::GetAnyRoot(buf);
    assert(maybe_source); //, "buffer is not a table");

    const flatbuffers::Table& source = *maybe_source;

    read_flatbuffer_table(source, schema, table, ti, target_ref);

    return result;
  }
}
#pragma endregion
