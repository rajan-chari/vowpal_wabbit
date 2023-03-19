#include "simple.h"

#include "../../type_constructors.h"
#include "../../type_registry.h"
#include "../../vwtypes/vwtypes.h"

#include "../../type_builder.h"

#include "../reduction_options.h"

#include <type_traits>

template <typename V>
class option_builder
{
public:
private:
  typesys::property_descriptor _pd;
  V _default_value;
};

namespace VW
{
namespace simple
{
namespace details_v1
{
using namespace VW::t3_REDUCTION;
using namespace typesys;

// template <typename T, const char* name>
// struct type_descriptor_witness
// {
// public:
//   type_descriptor_witness() : td(name, type<T>::erase())
//   {
//   }

//   type_descriptor td;
// };

// ////

struct empty_t
{
};

extern const char simple_config_name[] = "simple_config";
struct simple_config
{
  Prop<int> a;
  Prop<float> b;
};

pseudo_vw::VW::LEARNER::base_learner* simple_init(
    pseudo_vw::VW::setup_base_i* stack_builder, const simple_config* config)
{
  std::cout << "simple_init with " << config->a << std::endl;

  /*
  // exactly as currently done in setup()
  return make_learner().with_learn<simple_learn>().with_predict<simple_predict>();

  */

  return nullptr;
}

// template <typename T>
// struct type_builder
// {
//   template <typename P, P T::*ptr>
//   struct property_builder
//   {
//     struct option_builder
//     {
//     public:
//       option_builder& with_help(std::string help)
//       {
//         return *this;
//       }

//       option_builder& necessary()
//       {
//         return *this;
//       }

//       option_builder& with_alias(std::string alias)
//       {
//         return *this;
//       }

//       option_builder& with_short_alias(char alias)
//       {
//         return *this;
//       }

//       option_builder& allow_override()
//       {
//         return *this;
//       }

//       // TODO: Others

//       operator property_builder&()
//       {
//         return parent;
//       }

//     private:
//       option_descriptor& descriptor;
//       property_builder& parent;
//     };

//     option_builder& bind_option(std::vector<option_descriptor>& options)
//     {
//       options.push_back({});
      
//       return { options.back() , *this };
//     }

//     operator type_builder&()
//     {
//       return _tb;
//     }
//   private:
//     const property_descriptor& _pd;
//     type_builder& _tb;
//   };

//   static type_builder<T> register_type()
//   {
//     type_descriptor td = type_registry::instance().register_type(typeid(std::decay_t<T>).name(), type<T>::erase());
//     auto& it = type_registry::instance().find_type(typeid(T));

//     return type_builder<T>(td);
//   }

//   inline type_descriptor& descriptor()
//   {
//     return td;
//   }

//   inline const type_descriptor& descriptor() const
//   {
//     return td;
//   }

//   template <typename P, P T::*ptr>
//   property_builder<P, ptr>& bind_property(std::string name)
//   {
//     field_ptr_id<T, P, ptr> id;

//     if (typesys::is_instance<P, typesys::Prop>::value)
//     {
//       // this is a scalar
//       td.register_property({
//           name, 
//           erased_field_type::scalar(type<typename P::value_type>::erase()), 
//           id().erase_binder()});
//     }
//     else if (typesys::is_instance<P, typesys::Vec>::value)
//     {
//       // this is a vector
//       td.register_property({
//           name, 
//           erased_field_type::vector(type<typename P::value_type>::erase()), 
//           id().erase_binder()});
//     }
//     // else if (typesys::is_instance<P, typesys::UMap>::value)
//     // {
//     //   // this is a map not quite sure how to extract the internal types on P (=UMap<K, V>)
//     // }
    
//     return { td.properties.back(), *this };
//   }

// private:
//   type_builder(type_descriptor& td) : td(td)
//   {
//   }

//   type_descriptor& td;
// };

//template <typename C, typename P>

// TC_DATA(simple_config_tc)
// {
//   prop_(float, a).bind_option().with_help();
//   prop_(float, b).bind_option().with_help();
// };

reduction_descriptor describe_v1()
{
  type_descriptor& etd = type_builder_ex<empty_t>::register_type().descriptor();

  reduction_options options;

  reduction_data_descriptor data =
  {
    type_builder_ex<simple_config>::register_type()
      .with_property<Prop<int>, &simple_config::a>("a")
        .bind(options.add<int>("a").with_help("help for a in simple config"))
        // I think we can get it to something like:
        // bind<option_builder>().with_help("help for a in simple config")
      .with_property<Prop<float>, &simple_config::b>("b")
        .bind(options.add<float>("b").with_help("help for b in simple config"))
      .descriptor(),
    etd,
    etd
  };
  
  // TODO: I really would like to get this inline with simple_config
  // option_group<simple_config> simple_options("simple options");
  // simple_options.bind<&simple_config::a>().essential().with_help("help for a in simple options");

  reduction_descriptor desc
  {
    "simple",
    1,
    options.get(),
    data,
    erase_init_f<simple_config, simple_init>()
  };

  return desc;
}

}  // namespace details_v1
}  // namespace simple
}  // namespace VW