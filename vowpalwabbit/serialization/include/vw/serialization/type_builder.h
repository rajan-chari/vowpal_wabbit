#pragma once

#include "vw/serialization/base.h"
#include "vw/serialization/type_reflection.h"
#include "vw/serialization/type_registry.h"
#include "vw/serialization/type_constructors.h"

namespace typesys
{
  template <typename C, typename P> //, P C::*ptr>
  class property_builder_ex;

  template <typename C>
  class type_builder_ex
  {
  public:
    inline static type_builder_ex register_type()
    {
      return register_type(type_registry::instance());
    }

    inline static type_builder_ex register_type(type_registry& tr)
    {
      const char* name = typeid(std::decay_t<C>).name();
      return register_type(tr, name);
    }

    inline static type_builder_ex register_type(type_registry& tr, const char* name)
    {
      return type_builder_ex(tr, name);
    }

    template <typename P, P C::*ptr>
    property_builder_ex<C, P> with_property(std::string name)
    {
      field_ptr_id<C, P, ptr> id;

      if (typesys::is_instance<P, typesys::Prop>::value)
      {
        // need to pull out the inner type of the property
        std::cout << "type: " << typeid(typename P::value_type).name() << std::endl;

        // this is a scalar
        td.register_property({
            name, 
            erased_field_type::scalar(type<typename P::value_type>::erase()), 
            id().erase_binder()});
      }
      else if (typesys::is_instance<P, typesys::Vec>::value)
      {
        // this is a vector
        td.register_property({
            name, 
            erased_field_type::vector(type<typename P::value_type>::erase()), 
            id().erase_binder()});
      }
      // else if (typesys::is_instance<P, typesys::UMap>::value)
      // {
      //   // this is a map not quite sure how to extract the internal types on P (=UMap<K, V>)
      // }

      // TODO: Check the exact nature of the memory operations here
      return property_builder_ex<C, P>(*this, td.properties.back() );
    }

    type_descriptor& descriptor()
    {
      return td;
    }

  protected:
    type_builder_ex(type_builder_ex& tb) = default;
    type_builder_ex(type_builder_ex&& tb) = default;

  private:
    type_descriptor& td;

    type_builder_ex(type_registry& tr, const char* name) : td(tr.register_type(name, type<C>::erase()))
    {}

    //template <typename P> friend property_builder_ex<C, P>;
  };

  template <typename C, typename P>
  class binding_builder;

  template <typename C, typename P> //, P C::*ptr>
  class property_builder_ex : public type_builder_ex<C>
  {
  private:
    using Base = type_builder_ex<C>;

  public:
    property_builder_ex(type_builder_ex<C>& tb, property_descriptor& pd) : Base(tb), pd(pd)
    {}

    template <typename Binder>
    property_builder_ex bind(Binder binder)
    {
      binder.bind_property(pd);

      return *this;
    }

    template <typename BinderT, std::enable_if_t<is_instance<typename BinderT::BuilderT, binding_builder>::value>>
    typename BinderT::BuilderT& bind(BinderT binder)
    {
      return binder.template bind<C, P>(*this, pd);
    }

  protected:
    property_builder_ex(property_builder_ex<C, P>& pb) = default;
    property_builder_ex(property_builder_ex<C, P>&& pb) = default;

  private:
    // TODO: check that this is safe
    property_descriptor& pd;

    friend class type_builder_ex<C>;
  };

  template <typename C, typename P>
  class binding_builder : public property_builder_ex<C, P>
  {
  protected:
    using BaseT = property_builder_ex<C, P>;
  public:
    binding_builder(BaseT& pb) : BaseT(pb)
    {}
  };
}
