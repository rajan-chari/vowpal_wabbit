#pragma once

#include "base.h"

#include "type_reflection.h"

namespace typesys
{
  using name_t = const char*; // todo: std::string?
  using type_number = std::uint32_t;

  class type_registry
  {
  public:
    using types_t = std::vector<type_descriptor>;
    using type_iter = types_t::const_iterator;
    using type_map_t = std::unordered_map<name_t, type_number>;
    using std_type_map_t = std::unordered_map<std::type_index, type_number>;

    static type_registry& instance();

    type_registry();
    type_registry(const type_registry& tr) : _types(tr._types), types{&_types}, type_map(tr.type_map), std_type_map(tr.std_type_map) {}
    type_registry(type_registry&& tr) : _types(std::move(tr._types)), types{&_types}, type_map(std::move(tr.type_map)), std_type_map(std::move(tr.std_type_map)) {}

    type_descriptor& lookup_type(const type_number typeindex) { return _types[typeindex]; }
    const type_descriptor& lookup_type(const type_number typeindex) const { return _types[typeindex]; }
    
    type_iter find_type(const name_t name) const
    {
      auto iter = type_map.find(name);
      if (iter == type_map.end())
      {
        return _types.end();
      }

      return _types.begin() + (iter->second);
    }

    type_iter find_type(const std::type_index std_index) const
    {
      auto iter = std_type_map.find(std_index);
      if (iter == std_type_map.end())
      {
        return _types.end();
      }

      return _types.begin() + (iter->second);
    }

    //type_descriptor& lookup_type(const name_t name) { return types[type_map[name]]; }
    //type_descriptor& lookup_type(const std::type_index std_index) { return types[std_type_map[std_index]]; }

    template <typename T>
    type_descriptor& register_type()
    {
      return register_type(typeid(T).name(), type<T>::erase());
    }

    type_descriptor& register_type(const name_t name, erased_type etype)
    {
      return register_type_internal(name, etype, false);
    }

    type_iter types_begin() const { return _types.begin(); }
    type_iter types_end() const { return _types.end(); }

    const base::vector_view<type_descriptor> types;

    inline bool is_builtin(const type_descriptor& type) const 
    {
      return is_builtin(type.etype);
    }

    inline bool is_builtin(erased_type etype) const
    {
      auto it = std_type_map.find(etype.tindex);
      return it != std_type_map.end() && it->second <= max_builtin;
    }

  private:
    type_descriptor& register_type_internal(const name_t name, erased_type etype, bool is_builtin)
    {
      type_number typeindex = _types.size();
      _types.push_back(type_descriptor(name, etype));
      type_map[name] = typeindex;
      std_type_map[etype.tindex] = typeindex;
      return _types[typeindex];
    }

  private:
    types_t _types;
    type_number max_builtin;
    type_map_t type_map;
    std_type_map_t std_type_map;
  };
}