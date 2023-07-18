#include "vw/serialization/autofb_schema.h"
#include "vw/serialization/type_registry.h"
#include "vw/serialization/type_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace typesys;
using namespace autofb;

struct test_type
{
  Prop<int> a;
  Vec<float> b;
  Prop<std::string> c;
};

TEST(SerializationSchema, SmokeTest)
{
  std::string schema = R"(namespace test;

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

  EXPECT_EQ(fbs.text_data, schema);

}