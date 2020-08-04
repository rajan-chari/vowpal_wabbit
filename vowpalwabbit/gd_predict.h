// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

#include "interactions_predict.h"
#include "v_array.h"

namespace GD
{
// forward declare
uint32_t get_stack_depth();

// iterate through one namespace (or its part), callback function T(some_data_R, feature_value_x, feature_index)
template <class R, void (*T)(R&, float, uint64_t), class W>
void foreach_feature(W& /*weights*/, features& fs, R& dat, uint64_t offset = 0, float mult = 1.)
{
  for (features::iterator& f : fs) T(dat, mult * f.value(), f.index() + offset);
}

template <typename W, typename D>
inline void debug_weight_update_pre(W&, D&, float, features::iterator&, uint64_t)
{ /* Do nothing for most template parameters.  Specialized below for the type we care about*/ }

template <>
inline void debug_weight_update_pre<float, float>(float& w, float& dat, float mult, features::iterator& f, uint64_t offset)
{
  VW_DBG(get_stack_depth()) << "gd: update_feature: {pre_w=" << w << ", update=" << dat << ", mult=" << mult
              << ", f.v=" << f.value() << ", w[spare]=" << ((float*)&w)[3] << " (f.idx=" << f.index() << ", offset=" << offset
              << ")} ";
}

template <typename W, typename D>
inline void debug_weight_update_post(W&, D&)
{ /* Do nothing for most template parameters.  Specialized below for the type we care about*/ }

template <>
inline void debug_weight_update_post<float, float>(
    float& w, float&)
{
  VW_DBG_0 << "{w=" << w << "}"
                << "w[0] += update * mult * f.v * w[spare] " << std::endl;
}

// iterate through one namespace (or its part), callback function T(some_data_R, feature_value_x, feature_weight)
template <class R, void (*T)(R&, const float, float&), class W>
inline void foreach_feature(W& weights, features& fs, R& dat, uint64_t offset = 0, float mult = 1.)
{
  for (features::iterator& f : fs)
  {
    weight& w = weights[(f.index() + offset)];
    debug_weight_update_pre(w, dat, mult, f, offset);
    T(dat, mult * f.value(), w);
    debug_weight_update_post(w, dat);
  }
}

// iterate through one namespace (or its part), callback function T(some_data_R, feature_value_x, feature_weight)
template <class R, void (*T)(R&, const float, const float&), class W>
inline void foreach_feature(const W& weights, features& fs, R& dat, uint64_t offset = 0, float mult = 1.)
{
  for (features::iterator& f : fs)
  {
    const weight& w = weights[(f.index() + offset)];
    VW_DBG(get_stack_depth()) << "gd: vec_add: {pre_acc=" << dat << ", mult=" << mult << ", v=" << f.value()
                << ", w=" << w << " (f.idx=" << f.index() << ", offset=" << offset << ")} acc += mult * v * w"
                << std::endl;
    T(dat, mult * f.value(), w);
  }
}

template <class R>
inline void dummy_func(R&, const audit_strings*)
{
}  // should never be called due to call_audit overload

template <class R, class S, void (*T)(R&, float, S), class W>  // nullptr func can't be used as template param in old
                                                               // compilers
inline void generate_interactions(std::vector<std::vector<namespace_index>>& interactions, bool permutations, example_predict& ec,
    R& dat,
    W& weights)  // default value removed to eliminate
                 // ambiguity in old complers
{
  INTERACTIONS::generate_interactions<R, S, T, false, dummy_func<R>, W>(interactions, permutations, ec, dat, weights);
}

// iterate through all namespaces and quadratic&cubic features, callback function T(some_data_R, feature_value_x, S)
// where S is EITHER float& feature_weight OR uint64_t feature_index
template <class R, class S, void (*T)(R&, float, S), class W>
inline void foreach_feature(W& weights, bool ignore_some_linear, std::array<bool, NUM_NAMESPACES>& ignore_linear,
    std::vector<std::vector<namespace_index>>& interactions, bool permutations, example_predict& ec, R& dat)
{
  uint64_t offset = ec.ft_offset;
  if (ignore_some_linear)
    for (example_predict::iterator i = ec.begin(); i != ec.end(); ++i)
    {
      if (!ignore_linear[i.index()])
      {
        features& f = *i;
        foreach_feature<R, T, W>(weights, f, dat, offset);
      }
    }
  else
    for (features& f : ec) foreach_feature<R, T, W>(weights, f, dat, offset);

  generate_interactions<R, S, T, W>(interactions, permutations, ec, dat, weights);
}

inline void vec_add(float& p, const float fx, const float& fw) { p += fw * fx; }

template <class W>
inline float inline_predict(W& weights, bool ignore_some_linear, std::array<bool, NUM_NAMESPACES>& ignore_linear,
    std::vector<std::vector<namespace_index>>& interactions, bool permutations, example_predict& ec, float initial = 0.f)
{
  foreach_feature<float, const float&, vec_add, W>(
      weights, ignore_some_linear, ignore_linear, interactions, permutations, ec, initial);
  return initial;
}
}  // namespace GD
