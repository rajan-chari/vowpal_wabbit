#pragma once

#include "../base.h"
#include "../vwtypes/vwtypes.h"

#include "../type_reflection.h"

#include "reduction_descriptor.h"

namespace VW
{
  namespace t3_REDUCTION
  {
    template <typename T>
    class option_builder
    {
    public:
      option_builder(std::vector<option_descriptor>& options, std::string name = ESSENTIAL) : options(options), name(name)
      {
      }

      option_builder& add_alias(std::string name)
      {
        aliases.push_back(name);

        return *this;
      }

      option_builder& with_default(T default_value)
      {
        default_init_f = make_default_initializer(default_value);

        return *this;
      }

      option_builder& with_help(std::string help)
      {
        help = help;

        return *this;
      }

      option_builder& bind_property(const typesys::property_descriptor& pd)
      {
        options.push_back(option_descriptor{
          typesys::property_descriptor(pd), name, help, aliases, default_init_f
        });

        return *this;
      }

    private:
      std::string name;
      std::string help;
      std::vector<std::string> aliases;
      std::function<bool(typesys::erased_lvalue_ref&)> default_init_f = no_op;

      std::vector<option_descriptor>& options;
    };

    template <typename C, typename P>
    class option_binding_builder : typesys::binding_builder<C, P>
    {
    public:
      option_binding_builder(BaseT& base, option_descriptor& od) : typesys::binding_builder<C, P, option_binding_builder<C, P>>(base), od(od)
      {
      }

      //option_binding_builder

    private:
      option_descriptor& od;
    };

    class reduction_options
    {
    public:
      template <typename T>
      option_builder<T> add(std::string name)
      {
        return option_builder<T>(options, name);
      }

      template <typename T>
      option_builder<T> add(const char* name)
      {
        return add<T>(std::string(name));
      }

      template <typename T>
      option_builder<T> add_enabling()
      {
        return add<T>(ESSENTIAL);
      }

      std::vector<option_descriptor> get()
      {
        return options;
      }

      template <typename C, typename P>
      option_binding_builder<C, P> bind(typesys::property_builder_ex<C, P>& pb, const typesys::property_descriptor& pd)
      {
        option_descriptor od { property_descriptor(pd) };

        options.push_back(od)

        return option_binding_builder<C, P>(*options.back());
      }

    private:
      std::vector<option_descriptor> options;
    };
  };
}