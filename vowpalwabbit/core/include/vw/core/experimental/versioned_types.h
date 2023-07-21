#pragma once

#include "vw/serialization/type_constructors.h"
#include "vw/core/experimental/type_traits_ex.h"
#include "vw/serialization/type_registry.h"

#include <string>
#include <algorithm>
//#include <string_view> //C++17

namespace model_data
{
  using version_t = uint32_t;

  template <typename T>
  class container
  {
    void* p;
    typesys::ref v;

  public:
    container(void* p) : p(p), v(p)
    {
      //v = typesys::ref(); // TODO: pull the actual ref from p
    }

    inline operator T&() { return this->v.template get<T>(); }
    inline T& operator()() { return this->v.template get<T>(); } // TODO: force inline?
    inline T& operator()(const T& v) { this->v = v; return this->v; }
  };

  enum class persist_mode
  {
    always,
    save_resume
  };

  template <version_t max_version>
  class version_index
  {
  public:
    static constexpr version_t EXPERIMENTAL = static_cast<version_t>(-1);
    static constexpr version_t LATEST = max_version;

    template <typename...Versions>
    struct version_list
    {
      // check that the number of versions is the same as the number of types, or
      // it is larger than it by 1 and the last Version derives from type_version<EXPERIMENTAL>
      static_assert(
          sizeof...(Versions) == sizeof...(Versions) || 
          (
            sizeof...(Versions) == sizeof...(Versions) + 1 //&& 
             //std::is_base_of<type_version<EXPERIMENTAL>, typename std::tuple_element<sizeof...(Versions) - 1, std::tuple<Versions...>>::type>::value
          )
        , "The number of versions must be the same as the number of types, or one larger if the last version is EXPERIMENTAL");
      // check that the versions are in ascending order
      // TODO: this is not working for some reason
      //       is it even necessary?
      // static_assert(
      //     std::is_sorted(std::tuple<Versions...>::value_type::min_version...)
      //   , "The versions must be in ascending order");

      // mechanism to extract a type given a version_t
      //template <version_t v>
      //using version = type_version<version_index<max_version>::template version_to_index<v>()>;
    };

    template <version_t version>
    struct type_version
    {
      // check that the version is valid (either EXPERIMENTAL or in the range [1, max_version])
      static_assert(version == EXPERIMENTAL || (version >= 1 && version <= max_version), 
        "The version must be EXPERIMENTAL or in the range [1, max_version]");

    public:
      static constexpr version_t min_version = version;

      template <version_t v>
      static constexpr bool is_available()
      {
        return v >= min_version;
      }

    protected:
      template <typename T>
      container<T> transient(std::string name)
      {
        return container<T> { _state }; // represents producing a container around the state provider pointer
                                        // TODO: pipe through the state allocation
/*
        if (!transient)
        {
          register_with_persistence_story(name);
        }

        state.create_backing(name);
*/
      }

      template <typename T, persist_mode m = persist_mode::always>
      container<T> persisted(std::string name)
      {
        return container<T> { _state };
      }

    private:
      void* _state;
    };
  };

  struct no_exp {};

  template <typename T>
  struct is_vector : std::false_type {};

  template <typename V>
  struct is_vector<std::vector<V>>
  {
    static constexpr bool value = true;
  };

  // a template to extract the V type from std::vector<V>
  template <typename T>
  struct vector_type
  {
    using type = void;
  };

  template <typename V>
  struct vector_type<std::vector<V>>
  {
    using type = V;
  };

  // direct using for vector_type_t
  template <typename T>
  using vector_type_t = typename vector_type<T>::type;

  class i_serializer
  {
  public:
    virtual typesys::erased_type get_container_type() = 0;
    virtual version_t get_version() = 0;
    
    virtual typesys::erased_field_binder ensure_field(typesys::erased_field_type eftype, std::string name) = 0;

    template <typename V>
    inline typesys::erased_field_binder ensure_field(std::string name)
    {
      // TODO: detect if V is a vector type, and if so, use the vector type
      if (is_vector<V>::value)
      {
        return this->ensure_field(typesys::erased_field_type::vector(typesys::type<vector_type_t<V>>::erase()), name);
      }
      else
      {
        return this->ensure_field(typesys::erased_field_type::scalar(typesys::type<V>::erase()), name);
      }
      // TODO: map?
    }
  };

  template <typename T>
  class container_prototype
  {

  };

  template <typename T, typename... Rest>
  bool foreach_type(void* state, bool (*f)(void*, T&))
  {
    if (!f(state, *static_cast<T*>(state)))
    {
      return false;
    }

    return foreach_type<Rest...>(state, f);
  }

  // template <template <typename...Ts> typename typelist>
  // struct aggregate
  // {
  //   std::tuple<...Ts> data;

  // private:
  //   // a type function to strip the final type from a tuple
  //   template <typename T>
  //   struct strip_last_type;

  //   template <typename... Ts>
    
  // };

  template <typename Versioned, const char* type_name>
  class versioned_registrator
  {
  public:
    versioned_registrator()
    {
      auto registry = typesys::type_registry::instance();

      // check if the type is already registered
      if (registry.find_type(typeid(Versioned)) != registry.types_end() ||
          registry.find_type(type_name) != registry.types_end())
      {
        // TODO: how do we handle the error here?
        //   how does one deal with static-time errors?
        //   does constexpr/consteval solve the issue?
        return;
      }
      
      registry.register_type<Versioned>(type_name);
    }
  };

  template <version_t max_version>
  class versioned_data
  {
  public:
    template <typename T>
    bool try_get_version(T** out)
    {
      //TODO
      return false;
    }

    version_t get_version()
    {
      return this->instance_version;
    }
  
  protected:
    using i = version_index<max_version>;
    
    template <version_t v>
    using version = typename i::template type_version<v>;

    template <typename... Versions>
    using version_list = typename i::template version_list<Versions...>;

    template <typename... Versions>
    versioned_data(version_list<Versions...>, version_t version) : instance_version(version)
    {
    }

  private:
    version_t instance_version;

    //static void* _state_source;
  };

   #define call(macro, target) macro target
   #define print(a) a
   #define skip(a) 
   #define unwrap(a) call(print, a)

  #define DERIVE_VERSIONED(type) 
  #define VERSION_(version_number, wrapped_fields) \
  class v##version : version<version_number> \
  { \
  public: \
    call(print, wrapped_fields) \
    using version<version_number>::version; \
  };

  #define CONTAINERIZE_(type) container<type>
  #define ENSURE_FIELD_(field_type, type_tagged_identifier) \
    
  #define PROPERTY_(type, name) container<type> name
  #define TRANSIENT_(type, name) PROPERTY_(type, name) = ensure_transient<type>(#name)
  #define PERSISTED_(type, name) PROPERTY_(type, name) = ensure_persisted<type>(#name)
  #define RESUME_(type, name) PROPERTY_(type, name) = ensure_persisted<type, persist_mode::save_resume>(#name)


  class my_data : versioned_data<2>
  {
  public:

    class v1 : version<1>
    {
    public:
      //
      // C++17:
      // allows us to avoid having to construct the type to register its properties with the
      // serialization system (this simplifies the type system code; similar niceties can be
      // done with the type registration itself, which would allow us to avoid having to generate
      // a whole lot of spurious namespaces)
      // inline static auto property_a = ensure_transient<int>("a");
      // container<auto> a { property_a };
      
      container<int> a = transient<int>("a");
      container<int> b = persisted<int>("b");
      container<int> c = persisted<int, persist_mode::save_resume>("c");
      
      using version<1>::version; // inherit the ctor
    };
    class v2 : version<2>
    {
    public:
      container<int> d = transient<int>("d");
      container<int> e = persisted<int>("e");

      using version<2>::version; // inherit the ctor
    };
    class vEXP : version<i::EXPERIMENTAL>
    {
    public:
      container<int> f = transient<int>("f");
      container<int> g = persisted<int>("g");

      using version<i::EXPERIMENTAL>::version; // inherit the ctor
    };

    // DERIVE_VERSIONED_(my_data, (v1, v2, vEXP))
    using Versions = version_list<v1, v2, vEXP>;
    my_data(version_t version) : versioned_data(Versions{}, version)
    {}
    inline my_data() : my_data(i::LATEST) {}
    using versioned_data::try_get_version;
    using versioned_data::get_version;
  };

  void example()
  {
    my_data data;

    // imagine we somehow actually instantiated this

    my_data::v1* v1;
    if (data.try_get_version(&v1))
    {
      v1->a() = 1; // TODO: can we make this work without the ()?
      v1->b() = 2;
      v1->c() = 3;
    }

    my_data::v2* v2;
    if (data.try_get_version(&v2))
    {
      v2->d() = 4;
      v2->e() = 5;
    }
  }
}