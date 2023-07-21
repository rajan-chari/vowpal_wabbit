#pragma once

#include "vw/core/io_buf.h"

#include "vw/serialization/type_constructors.h"
#include "vw/config/options.h"
#include "vw/config/option_builder.h"

#include "vw/core/experimental/versioned_types.h"

#include <string>

namespace model_data
{

#define PROP_(type, name) typesys::Prop<type> name
#define VEC_(type, name) typesys::Vec<type> name

class header_config
{
public:
  typesys::Prop<std::string> version;
  typesys::Prop<std::string> id;
  typesys::Prop<float> min_label;
  typesys::Prop<float> max_label;

  typesys::Prop<uint32_t> num_bits;
  typesys::Prop<uint32_t> rank;
  typesys::Prop<uint32_t> lda;

  typesys::Vec<std::string> ngrams; // TODO: this should live in the feature_config
  typesys::Vec<std::string> skips;
};

class feature_config
{
public:
  uint32_t _version;
  size_t _size;
};

template <uint32_t version>
class feature_config_ : public feature_config
{
protected:
  feature_config_(size_t size)
  {
    _version = version;
    _size = size;
  }
};

class feature_config_v1 : feature_config_<1>
{
public:
  typesys::Prop<std::string> hash;
  typesys::Prop<uint32_t> hash_seed;

  typesys::Vec<std::string> ignored_namespaces;
  typesys::Vec<std::string> ignored_namespaces_linear;

  typesys::Vec<std::string> kept_namespaces;
  typesys::Vec<std::string> redefined_namespaces; // TODO: store as map<> directly?

  typesys::Prop<bool> no_constant;

  typesys::Prop<std::string> affix_features;
  typesys::Prop<std::string> spelling_features;
  typesys::Prop<std::string> dictionary_features;
  typesys::Prop<std::string> dictionary_path;

  typesys::Prop<std::string> interactions;
  // typesys::Prop<bool> permutations // DOES THIS NEED TO BE .keep()?
  // typesys::Prop<bool> leave_duplicate_interactions // DOES THIS NEED TO BE .keep()?

  // typesys::Prop<std::string> feature_limits; // DOES THIS NEED TO BE .keep()?

  // TODO: pull to feature_config_experimental

public:
  feature_config_v1() : feature_config_<1>(sizeof(feature_config_v1)) {}
};


class feature_config_vEXP : feature_config_<static_cast<uint32_t>(-1)>
{
public:
  typesys::Vec<std::string> ignore_features_EXP; 
  typesys::Prop<bool> full_name_interactions_EXP;
};

class durable_model
{
public:
  typesys::Prop<header_config> header;
  typesys::Prop<feature_config> feature_tweaks;

  // std::vector<reduction_config> reductions;
};

#undef PROP_
#undef VEC_

}

inline model_data::durable_model load_durable_model(VW::io_buf* buffer)
{
  model_data::durable_model model;
  return model;
}