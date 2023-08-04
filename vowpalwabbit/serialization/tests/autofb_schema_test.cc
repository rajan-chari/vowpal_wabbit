#include "vw/serialization/autofb_schema.h"
#include "vw/serialization/autofb_serializer.h"
#include "vw/serialization/type_registry.h"
#include "vw/serialization/type_builder.h"
#include "vw/serialization/type_constructors.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace typesys;
using namespace autofb;

// function that takes a string and returns a string and
// replaces newlines and tabs with spaces
std::string normalize(const std::string& str)
{
  // replace newlines and tabs with spaces
  std::string ret = str;
  std::replace(ret.begin(), ret.end(), '\n', ' ');
  std::replace(ret.begin(), ret.end(), '\t', ' ');

  // remove multiple spaces
  ret.erase(std::unique(
    ret.begin(),
    ret.end(),
    [](char a, char b) { return a == ' ' && b == ' '; }),
    ret.end());
  return ret;
}

class NoDefaultConstructor
{
  public:
    NoDefaultConstructor(int a) : a(a) {}
    int a;
};

class DefaultConstructor
{
  public:
    DefaultConstructor() = default;
    DefaultConstructor(int a) : a(a) {}
    int a;
};

// TODO: test all basic types Prop<>, Vec<>
// TODO: struct, struct of struct
// Note: need a default constructor for now

TEST(Serialization, PropClasses)
{
  Prop<int> a;
  a=1;
  EXPECT_EQ(a.val, 1);
  EXPECT_EQ(a, 1);

  Prop<int> b(1);
  EXPECT_EQ(b.val, 1);
  EXPECT_EQ(b, 1);

  Prop<int> c = 1;
  EXPECT_EQ(c.val, 1);
  EXPECT_EQ(c, 1);

  Prop<DefaultConstructor> d;

  // The following is a compile error, as expected
  //Prop<NoDefaultConstructor> d;

  Prop<NoDefaultConstructor> e(1);
  EXPECT_EQ(e.val.a, 1);
}

template <typename T>
struct test_single
{
  Prop<T> a;
};

template<typename T>
auto test_single_type(const std::string& type_str) -> void
{
  std::string schema_str = "namespace test; table test_single { a:" + type_str + "; } ";
        
  type_registry registry;
  type_descriptor td = type_builder_ex<test_single<T>>::register_type(
                        registry, "test_single"
                      )
                      .with_property<Prop<T>, &test_single<T>::a>("a")
                      .descriptor();
  schema_builder builder("test", registry);
  fbs_data fbs = builder.build_idl();
  EXPECT_EQ(normalize(fbs.text_data), normalize(schema_str));
}

TEST(Serialization, IndividualType_SchemaGeneration)
{
  test_single_type<int8_t>("int8");
  test_single_type<int16_t>("int16");
  test_single_type<int32_t>("int32");
  test_single_type<int64_t>("int64");
  test_single_type<uint8_t>("uint8");
  test_single_type<uint16_t>("uint16");
  test_single_type<uint32_t>("uint32");
  test_single_type<uint64_t>("uint64");
  test_single_type<bool>("bool");
  test_single_type<float>("float");
  test_single_type<double>("double");
  test_single_type<std::string>("string");
}

TEST(SerializationSupport, ErasedLValueAndErasedField)
{
  // This is the struct we want to manipulate
  test_single<int> t;
  t.a = 1000;

  // erased ref to t.a
  erased_lvalue_ref elv_t_a {
    type<Prop<int>>::erase(),
    ref(t.a)
  };

  // erased ref to t
  erased_lvalue_ref elv_t {
    type<test_single<int>>::erase(),
    ref(t)
  };

  // set t.a using erased_lvalue_ref 
  auto& val = elv_t_a.get<Prop<int>>();
  assert(val.val == 1000);
  val.val = 2000;
  assert(t.a == 2000);

  // field_ptr holds offset a given member variable for a given class
  // In this case, we want to hold offset to 'a' for test_single<int>
  constexpr field_ptr<test_single<int>, Prop<int>> field_ptr_a {&test_single<int>::a};

  // Declare the pointer that will eventually point to t.a
  erased_lvalue_ref* elv_field_ptr = nullptr;

  // binder that can be used to bind t.a to an erased_lvalue_ptr 
  const erased_field_binder erased_field_binder_a = field_ptr_a.erase_binder();
  assert(erased_field_binder_a.try_bind(elv_t, elv_field_ptr));

  auto& val2 = elv_field_ptr->get<Prop<int>>();
  
  val2 = 3000;
  assert(t.a == 3000);
  delete elv_field_ptr;
}

TEST(SerializationSupport, FieldPtr)
{
  test_single<int> t;

  field_ptr_id<test_single<int>,Prop<int>,&test_single<int>::a> ptr_id;
  field_ptr<test_single<int>, Prop<int>> ptr = ptr_id();
  Prop<int>& x = ptr.bind(t);
  x = 1000;
  assert(t.a == 1000);

  field_ptr<test_single<int>, Prop<int>> ptr2 {&test_single<int>::a};
  erased_lvalue_ref elv_t { type<test_single<int>>::erase(), ref(t) };
  Prop<int>* y;
  assert(ptr2.try_bind(elv_t,y));
  y->val = 2000;
  assert(t.a == 2000);

  ///// Study this
  erased_lvalue_ref* elv_ptr;  // This will be a pointer to a Prop<int>
  erased_field_binder binder = ptr2.erase_binder();  // This is a binder for Prop<int>
  assert(binder.try_bind(elv_t, elv_ptr));  // This binds elv_ptr to t.a
  elv_ptr->set(3000);
  assert(t.a == 3000);

  delete elv_ptr;
}

TEST(Serialization, IndividualType_Write_Read)
{
  // register with the global instance
  auto& registry = type_registry::instance();
  //Register type
  type_descriptor td = type_builder_ex<test_single<int>>::register_type(
                        registry, "test_single"
                      )
                      .with_property<Prop<int>, &test_single<int>::a>("a")
                      .descriptor();
  schema_builder builder("test", registry);
  // Serialize and deserialize test

  // Begin serialize
  schema schema_var = builder.build();
  serializer serializer(schema_var);
  flatbuffers::FlatBufferBuilder fbb;

  constexpr int input_val = 1791;
  test_single<int> t{input_val};
  auto erased = type<test_single<int>>::erase();
  ref a_ref(t); 
  erased_lvalue_ref elv { erased, a_ref };
  offset_of_any offset = serializer.write_flatbuffer(fbb, elv);
  fbb.Finish(flatbuffers::Offset<void>(offset));
  // End serialize

  // Begin deserialize
  uint8_t* buf = fbb.GetBufferPointer();
  size_t size = fbb.GetSize();

  activation act = serializer.read_flatbuffer(buf, erased);
  test_single<int> out_struct = act.get<test_single<int>>();
  EXPECT_EQ(input_val, out_struct.a);
  // End deserialize
}

struct test_type
{
  Prop<int> a;
  Vec<float> b;
  Prop<std::string> c;
};

TEST(SerializationSchema, SmokeTest)
{
  std::string schema_str = R"(namespace test;

table test_type {
  a:int32;
  b:[float];
  c:string;
}

)";

  type_registry registry;
  

  type_descriptor td = type_builder_ex<test_type>::register_type(registry, "test_type") // relying on type_info::name() is a bad idea
                                                                                        // because the C++ standard just cannot get 
                                                                                        // over its need to be nonstandard, and this is
                                                                                        // thus, implementation-defined.
    .with_property<Prop<int>, &test_type::a>("a")
    .with_property<Vec<float>, &test_type::b>("b")
    .with_property<Prop<std::string>, &test_type::c>("c")
    .descriptor();


  schema_builder builder("test", registry);
  fbs_data fbs = builder.build_idl();

  EXPECT_EQ(fbs.text_data, schema_str);

  }

