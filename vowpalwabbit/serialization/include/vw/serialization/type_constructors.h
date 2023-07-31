#pragma once

#include "vw/serialization/base.h"
#include "vw/serialization/type_reflection.h"
#include <type_traits>

namespace typesys
{
template <typename T, const field_kind kind>
struct PropBase
{
  using field_kind_t = std::integral_constant<field_kind, kind>;
  using storage_type = T;

  // (C++ 14 has enable_if_t, but this works with C++ 11)
  // If T is default-constructible, provide a default constructor for PropBase<T>
  template <typename U = T, typename std::enable_if<std::is_default_constructible<U>::value>::type* = nullptr>
  PropBase() : val(T()) {}

  PropBase(const T& val) : val(val) {} // copy constructor that takes a T
  PropBase(T&& val) : val(std::move(val)) {} // move constructor that takes a T

protected:
  T val;
};

template <typename T>
struct Prop : public PropBase<T, field_kind::scalar>
{
  using value_type = T;

  static erased_field_type& eftype()
  {
    static auto _eftype = erased_field_type::scalar(type<T>::erase());
    return _eftype;
  }

  // (C++ 14 has enable_if_t, but this works with C++ 11)
  // If T is default-constructible, provide a default constructor for PropBase<T>
  template <typename U = T, typename std::enable_if<std::is_default_constructible<U>::value>::type* = nullptr>
  Prop() : PropBase<T, field_kind::scalar>{} {}
  
  Prop(const T& val) : PropBase<T, field_kind::scalar>{val} {}
  Prop(T&& val) : PropBase<T, field_kind::scalar>{std::move(val)} {}

  // copy assignment operator
  Prop& operator=(const T& other)  
  {
    this->val = other;
    return *this;
  }

  // move assignment operator
  Prop& operator=(T&& other)  
  {
    this->val = std::move(other);
    return *this;
  }

  inline erased_lvalue_ref reflect() { return erased_lvalue_ref{eftype().evalue, ref{val}}; }

  using PropBase<T, typesys::field_kind::scalar>::val;

  // implicit cast to T&
  inline operator T&() { return val; }
  inline operator const T&() const { return val; }
};

template <typename T>
struct Vec : public PropBase<std::vector<T>, field_kind::vector>
{
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  Vec() = default;
  Vec(const std::vector<T>& val) : PropBase<std::vector<T>, field_kind::vector>{val} {}
  Vec(std::vector<T>&& val) : PropBase<std::vector<T>, field_kind::vector>{std::move(val)} {}

  iterator begin() { return this->val.begin(); }
  iterator end() { return this->val.end(); }
  const_iterator begin() const { return this->val.begin(); }
  const_iterator end() const { return this->val.end(); }

  inline erased_vector reflect() { return vtype<T>::erase(this->val); }
};

template <typename K, typename V>
struct UMap : public PropBase<std::unordered_map<K, V>, field_kind::map>
{
  using key_type = K;
  using value_type = V;

  UMap() = default;
  UMap(const std::unordered_map<K, V>& val) : PropBase<std::unordered_map<K, V>, field_kind::map>{val} {}
  UMap(std::unordered_map<K, V>&& val) : PropBase<std::unordered_map<K, V>, field_kind::map>{std::move(val)} {}
};

template <typename C, typename T, template<typename _> typename Wrapper = Prop>
struct PropertyBuilder
{
  struct property_desc
  {
    std::string _name;
    typesys::erased_type _type;
    typesys::erased_field_binder _binder;
  };

  using field_ptr_t = Wrapper<T>(C::*);

  property_desc _desc;

  PropertyBuilder()
  {
    _desc._type = type<T>::erase();
  }

  PropertyBuilder& with_name(const char* name)
  {
    _desc._name = name;
    return *this;
  }

  PropertyBuilder& with_field_ptr(field_ptr_t ptr)
  {
    _desc._binder = typesys::field_ptr<C, T>{ptr}.get_binder();
    return *this;
  }

  // implicit cast operator to Prop<T>
  operator Prop<T>() const
  {
    // how do we ensure we can get the property_desc from somewhere?

    return Prop<T>{};
  }
};

template <typename K>
struct MapHelper
{
  template <typename V>
  using Wrapper = UMap<K, V>;
};

template <typename C, typename K, typename V>
using MapPropertyBuilder = PropertyBuilder<C, UMap<K, V>, MapHelper<K>::template Wrapper>;

template <typename, template <typename, typename...> typename>
struct is_instance : public std::false_type {};

template <typename...Ts, template <typename, typename...> typename U>
struct is_instance<U<Ts...>, U> : public std::true_type {};

}  // namespace typesys
