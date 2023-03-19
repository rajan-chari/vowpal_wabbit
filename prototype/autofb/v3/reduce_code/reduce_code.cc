#include "..\base.h"
#include "..\test\test.h"

RUN_ONCE({
  std::cout << "Who needs main() anyways? REDUX (now with more reduce_code)" << std::endl;

  
})

/*
 * Our fundamental issue is as follows: We need to have different layers (based on version) of types
 * used by the reductions, which are configured for auto-persist, or binding to options. Unfortunately,
 * because we lack a nameof() operator, and in particular, we are missing something like: */

#if 0
template <typename C, typename V, const V&(C::&member_ref)>
struct member
{
  static const char* name(); // { => which would return the declared name of the member }

  static V& bind(C& c) { return c.*member_ref; }
};

#define REFLECT_MEMBER(C, member) \
namespace reflection { namespace details { \
  template <typename C, > \
  member_reflector<C, decltype(C::member), &C::member>

template <typename C, const C& ref>
struct variable
{
  static const char* name(); // { => which would return the declared name of the member }

  static C& bind() { return ref; }
};

template <typename RefType>
#endif