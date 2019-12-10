#include "reductions.h"
#include "cb_sample.h"
#include "explore.h"

using namespace LEARNER;
using namespace VW;
using namespace VW::config;

namespace VW
{
// cb_sample is used to automatically sample and swap from a cb explore pdf.
struct cb_sample_data
{
  explicit cb_sample_data(std::shared_ptr<rand_state> &random_state) : _random_state(random_state) {}
  explicit cb_sample_data(std::shared_ptr<rand_state> &&random_state) : _random_state(random_state) {}

  inline void learn(multi_learner &base, multi_ex &examples)
  {
    multiline_learn_or_predict<true>(base, examples, examples[0]->ft_offset);
  }

  uint64_t get_random_seed(example *ec)
  {
    // Seed comes from the example
    bool tag_provided_seed = false;
    uint64_t seed = _random_state->get_current_state();
    if (!ec->tag.empty())
    {
      const std::string SEED_IDENTIFIER = "seed=";
      if (strncmp(ec->tag.begin(), SEED_IDENTIFIER.c_str(), SEED_IDENTIFIER.size()) == 0 &&
          ec->tag.size() > SEED_IDENTIFIER.size())
      {
        substring tag_seed{ec->tag.begin() + 5, ec->tag.begin() + ec->tag.size()};
        seed = uniform_hash(tag_seed.begin, substring_len(tag_seed), 0);
        tag_provided_seed = true;
      }
    }

    // Update the seed state in place if it was used.
    if (!tag_provided_seed)
      _random_state->get_and_update_random();

    return seed;
  }

  void sample(example* first_ex) {
    auto action_scores = first_ex->pred.a_s;
    uint32_t chosen_action = -1;

    // Get random seed used for sampling
    const uint64_t seed = get_random_seed(first_ex);

    // Sampling is done after the base learner has generated a pdf.
    auto result = exploration::sample_after_normalizing(
        seed, ACTION_SCORE::begin_scores(action_scores), ACTION_SCORE::end_scores(action_scores), chosen_action);
    assert(result == S_EXPLORATION_OK);
    _UNUSED(result);

    result = exploration::swap_chosen(action_scores.begin(), action_scores.end(), chosen_action);
    assert(result == S_EXPLORATION_OK);

    _UNUSED(result);
  }

  inline void predict(multi_learner &base, multi_ex &examples)
  {
    // Get action scores using base learner
    multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset);
    example *first_ex = examples[0];

    sample(first_ex);
  }

 private:
  std::shared_ptr<rand_state> _random_state;
};
}  // namespace VW

void learn(cb_sample_data &data, multi_learner &base, multi_ex &examples) { data.learn(base, examples); }

void predict(cb_sample_data &data, multi_learner &base, multi_ex &examples) { data.predict(base, examples); }

base_learner *cb_sample_setup(options_i &options, vw &all)
{
  bool cb_sample_option = false;

  option_group_definition new_options("CB Sample");
  new_options.add(make_option("cb_sample", cb_sample_option).keep().help("Sample from CB pdf and swap top action."));
  options.add_and_parse(new_options);

  if (!cb_sample_option)
    return nullptr;

  if (options.was_supplied("no_predict"))
  {
    THROW("cb_sample cannot be used with no_predict, as there would be no predictions to sample.");
  }

  auto data = scoped_calloc_or_throw<cb_sample_data>(all.get_random_state());
  return make_base(init_learner(data, as_multiline(setup_base(options, all)), learn,
      predict, 1 /* weights */, prediction_type::action_probs, "cb_sample"));
}
