#pragma once

#include "../base.h"
#include "../type_registry.h"
#include "../type_constructors.h"
#include "../reflection.h"

#include "test.h"

namespace autofb { namespace test {
  using namespace typesys;

  struct simple_types
  {
    #define _(type) Prop<type> _ ## type
    _(bool);
    _(int8_t);
    _(uint8_t);
    _(int16_t);
    _(uint16_t);
    _(int32_t);
    _(uint32_t);
    _(int64_t);
    _(uint64_t);
    _(float);
    _(double);
    #undef _
  };

  void print_known_types();

  struct types_registrator
  {
    type_registry operator()()
    {
      //std::cout << "Registering Types" << std::endl;

      type_registry result;

      type_descriptor& td = result.register_type<simple_types>();
      
      #define REGISTER_PROP(prop_type)  \
      td.register_property(property_descriptor{ \
        #prop_type, \
        erased_field_type::scalar(type<prop_type>::erase()), \
        field_ptr_id<simple_types, Prop<prop_type>, &simple_types::_ ## prop_type>()().erase_binder() \
      });

      REGISTER_PROP(bool);
      REGISTER_PROP(int8_t);
      REGISTER_PROP(uint8_t);
      REGISTER_PROP(int16_t);
      REGISTER_PROP(uint16_t);
      REGISTER_PROP(int32_t);
      REGISTER_PROP(uint32_t);
      REGISTER_PROP(int64_t);
      REGISTER_PROP(uint64_t);
      REGISTER_PROP(float);
      REGISTER_PROP(double);
      
      return result;
    }
  };

  extern ::test::let<types_registrator, type_registry> test_types;
}}
