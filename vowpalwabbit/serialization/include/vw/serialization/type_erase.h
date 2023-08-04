#pragma once

#include "vw/serialization/base.h"
#include "vw/serialization/type_activation.h"

#include <typeinfo>
#include <typeindex>

#include <unordered_map>
#include <cstring>

namespace typesys
{
struct erased_vector;
using e_vector_builder_f = erased_vector(*)();

template <typename T>
struct type;
template <typename T>
struct vtype;

struct erased_type
{
private:
  // must be created through type::erase()
  erased_type() = default;

public:
  std::type_index tindex;
  size_t size;
  activator_f activator;
  // This builder is required to be able to get an erased_vector out of an erased_type
  e_vector_builder_f e_vector_builder;
  //e_map_builder_f e_map_builder;

public:
  template <typename T>
  bool is() const
  {
    return tindex == std::type_index{typeid(T)};
  }

  // equality operator based on type_index
  bool operator==(const erased_type& other) const
  {
    return tindex == other.tindex;
  }

  bool operator!=(const erased_type& other) const
  {
    return tindex != other.tindex;
  }

  // hash operator based on type_index
  struct hash
  {
    inline size_t operator()(const erased_type& et) const
    {
      return et.tindex.hash_code();
    }
  };

  template <typename T>
  friend class type;
};

template <typename T>
struct type
{
public:
  template< class Q = T,
            typename std::enable_if<
              base::is_specialization<Q, std::vector>::value, bool
            >::type = true>
  inline static erased_type erase()
  {
    return erased_type{typeid(T), sizeof(T), &activator<T>::activate, []() {return vtype<T>::erase();}};
  }

  template< class Q = T,
            typename std::enable_if<
              !base::is_specialization<Q, std::vector>::value, bool
            >::type = true>
  inline static erased_type erase()
  {
    return erased_type{typeid(T), sizeof(T), &activator<T>::activate, nullptr};
  }

  using typed_init_f = T* (*)();
  using typed_destroy_f = void (*)(T*);

  template <typed_init_f init>
  struct init_f_erased
  {
    inline void* operator()()
    {
      return static_cast<void*>(init());
    }
  };

  template <typed_destroy_f destroy>
  struct destroy_f_erased
  {
    inline void operator()(void* ptr)
    {
      destroy(static_cast<T*>(ptr));
    }
  };

  //type-eraser based on providing a default_init, default_destroy pair
  //this allows non-default constructible types to be erased
  template <typed_init_f init, typed_destroy_f destroy>
  inline static erased_type erase()
  {
    return erased_type{typeid(T), sizeof(T), &activator<T, init, destroy>::activate, nullptr};
  }
};

/////////////

using typeinfo_predicate_f = bool(*)(const std::type_info);

template <typename...Ts>
using dispatch_f = void(*)(Ts...args);

template <typename...Ts>
using erased_dispatch = std::pair<typeinfo_predicate_f, dispatch_f<Ts...>>;

template <typename T, typename...Ts>
struct type_dispatch
{
  template <dispatch_f<Ts...> D>
  static erased_dispatch<Ts...> erase()
  {
    return { &is_match, D };
  }

  static bool is_match(const std::type_info info)
  {
    return info == typeid(T);
  }
};

template <typename... Ts>
class erased_dispatch_table
{
  using erased_dispatch_t = erased_dispatch<Ts...>;

  public:
    // TODO: SFINAE
    template <typename T, dispatch_f<Ts...> D>
    erased_dispatch_table& add()
    {
      //erased_dispatch<Ts...> erased = {&type_dispatch<T, Ts...>::is_match, D};
      erased_dispatch<Ts...> erased = type_dispatch<T, Ts...>::template erase<D>();
      dispatch_table[typeid(T)] = erased;
      return *this;
    }

    template <typename T>
    static bool is_match(const std::type_info info)
    {
      return info == typeid(T);
    }

    template <typename T>
    erased_dispatch_table& add(dispatch_f<Ts...> D)
    {
      erased_dispatch<Ts...> erased = { &is_match<T>, D };
      dispatch_table[typeid(T)] = erased;
      return *this;
    }

    bool dispatch(const erased_type& type, Ts...args)
    {
      return dispatch(type.tindex, std::forward<Ts>(args)...); 
    }

    bool dispatch(const std::type_info type_info, Ts... args)
    {
      return dispatch(std::type_index{type_info}, std::forward<Ts>(args)...); 
    }

    bool dispatch(const std::type_index& type_index, Ts...args)
    {
      // find the dispatch based on the erased type's type_index
      auto iter = dispatch_table.find(type_index);
      if (iter == dispatch_table.end())
        return false;

      erased_dispatch<Ts...> dispatch = iter->second;
      dispatch.second(std::forward<Ts>(args)...);
      return true;
    }

  private:
    std::unordered_map<std::type_index, erased_dispatch_t> dispatch_table;
};

struct ref
{
  void* _r;

  ref(const ref& r)
  {
    _r = r._r;
  }

  ref(ref&& r)
  {
    _r = r._r;
  }

  // the enable_if does a very important job here - it prevents the compiler from
  // treating the template as a valid copy constructor for ref. Without it, we
  // double-wrap whenever we copy, leading to "fun" bugs.
  template <typename T, class = typename std::enable_if<!std::is_same<T, ref>::value>::type>
  ref(T& r) : _r(&r) 
  {
    // std::cout << "ref(T& r) : r(&r) @" << this->_r << std::endl << '\t';
    // std::cout << *(float*)(&r) << " | " << get<float>() << std::endl << '\t';

    // if (std::is_same<T, float>::value)
    // {
    //   std::cout << "is float" << std::endl;
    // }
    // else if (std::is_same<T, ref>::value)
    // {
    //   std::cout << "is ref" << std::endl;
    // }
    // else
    // {
    //   std::cout << "is not float or ref" << std::endl;
    // }
  }

  // void* ctor
  ref(void* r) : _r(r) 
  {
  }

  //getter
  template <typename T>
  T& get() const 
  { 
    // std::cout << "getting @" << _r << std::endl; 
    return *static_cast<T*>(_r); 
  }

  //getter as void*
  void*& get() const { return get<void*>(); }

  //setter
  template <typename T>
  T& operator=(T& t) { return *static_cast<T*>(_r) = t; }

  //setter as void*
  //void* operator=(void* t) { return *static_cast<void**>(r); }
};

// TODO: The name here is a little wrong, since ref is a non-owning reference.
// Rename this.
class erased_lvalue_ref
{
public:
  erased_lvalue_ref(erased_type etype, ref slot) : _type{etype}, slot{slot}
  {
  }

  erased_type _type;

  // TODO: this really should be more similar to how property<T> works
  void*& get() const
  { 
    return (slot.get()); 
  }

  template <typename T>
  T& get() const
  {
    return slot.get<T>();
  }

  template <typename T>
  void set(T t)
  {
    slot.get<T>() = t;
  }

  void set(void* data) 
  { 
    slot = data;
  }

private:
  ref slot;
};

// This represents the notion of a datum that is in the process of being moved.
// This of it as a type-erased equivalent of std::move's output. The implication is
// that it is the receiver's responsibility to put the data back into a typed context
// that understands how to delete it.
struct erased_moved
{
public:
  erased_moved(erased_type etype, void* data) : _type{etype}, _data{data}
  {
  }

  erased_moved(erased_type etype, activation&& source) : _type{etype}, _data{source.move_out()}
  {
  }

  erased_type _type;

  template <typename T>
  bool is() const
  {
    return _type.is<T>();
  }

  template <typename T>
  T&& move_out()
  {
    // TODO: what do we do if we are not is<T>()?

    return std::move(*static_cast<T*>(_data));
  }

private:
  void* _data;
};

struct i_vector
{
  using data_f = std::function<void*(void)>;
  data_f _data_f;

  using size_f = std::function<size_t(void)>;
  size_f _size_f;

  using assign_f = std::function<void(const void*, const void*)>;
  assign_f _assign_f;

  using push_back_f = std::function<void(erased_moved)>;
  push_back_f _push_back_f;
};

struct erased_vector
{
  erased_type _type;
  i_vector _vector;

  void* data() const { return _vector._data_f(); }
  size_t size() const { return _vector._size_f(); }

  template <typename T>
  void reduce(dispatch_f<const T*, size_t> dispatch) const
  {
    T* tdata = static_cast<T*>(data());
    dispatch(tdata, size());
  }

  template <typename T>
  void reduce(std::function<void(const T*, size_t)> dispatch) const
  {
    T* tdata = static_cast<T*>(data());
    dispatch(tdata, size());
  }

  void copy_to(void* dest, size_t dest_size) const
  {
    const size_t count = dest_size / _type.size;

    if (count < size())
    {
      // TODO: report size mismatch (how do we do this without throw?)
    }

    std::memcpy(dest, data(), count * _type.size);
  }

  void push_back(void* element)
  {
    _vector._push_back_f(erased_moved{_type, element});
  }

  void assign_from(const void* src_begin, const void* src_end)
  {
    _vector._assign_f(src_begin, src_end);
  }
};

template <typename T>
struct vtype
{
  static erased_vector erase(std::vector<T>& vec)
  {
    i_vector v = {
      [&vec]() -> void* { return vec.data(); }, 
      [&vec]() -> size_t { return vec.size(); },
      [&vec](const void* begin, const void* end) { vec.assign(static_cast<const T*>(begin), static_cast<const T*>(end)); },
      [&vec](erased_moved element) { vec.push_back(element.move_out<T>()); }
    };

    return erased_vector{ type<T>::erase(), v };
  }

  // TODO: Do we need to implement erase(v_array<>)?
};

}
