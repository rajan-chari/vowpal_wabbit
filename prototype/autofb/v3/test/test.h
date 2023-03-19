#pragma once

#include <type_traits>
#include <typeinfo>

#include "../type_erase.h"

namespace test
{
  template <typename Key, typename T, typename ValueFactory>
  struct singleton_detail
  {
  public:
    using value_t = T;

    inline static value_t& get() { return singleton_detail::instance()._value; };
  
  private:
    value_t _value = ValueFactory{}();

  public:
    static singleton_detail& instance()
    {
      static singleton_detail<Key, T, ValueFactory> instance;
      return instance;
    }
  };

  template <typename KeyCat>
  using singleton = singleton_detail<KeyCat, typename KeyCat::value_t, typename KeyCat::value_factory>;

  template <typename T, typename ValueFactory>
  struct key
  {
    using value_t = T;
    using value_factory = ValueFactory;

  private:
    using vf_return_type = decltype(std::declval<ValueFactory>()());
    static_assert(std::is_same<value_t, vf_return_type>::value, "ValueFactory must return T");
  };

  template <typename Functor, typename T>
  struct let
  {
  public:
    using key_t = key<T, Functor>;

  public:
    inline const T& get()
    {
      return singleton<key_t>::get();
    }
  };

  template <typename Functor>
  struct let<Functor, void>
  {
  public:
    let()
    {
      Functor{}();
    }
  };

  template <typename Functor>
  using run_static = let<Functor, void>;
}

#define CALL_ONCE(Functor) namespace { ::test::run_static<Functor> _run_static_##Functor; }
#define RUN_ONCE(Body) namespace { struct Functor { void operator()() { Body }}; ::test::run_static<Functor> _run_static_##Functor; }
