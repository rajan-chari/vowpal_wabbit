// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

/*
 * Implementation of online boosting algorithms from
 *    Beygelzimer, Kale, Luo: Optimal and adaptive algorithms for online boosting,
 *    ICML-2015.
 */

#include <cfloat>
#include <climits>
#include <cmath>
#include "correctedMath.h"
#include <cstdio>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include "reductions.h"
#include "vw.h"
#include "rand48.h"

using namespace VW::LEARNER;
using namespace VW::config;

using std::cerr;
using std::endl;

inline float sign(float w)
{
  if (w <= 0.)
    return -1.;
  else
    return 1.;
}

int64_t choose(int64_t n, int64_t k)
{
  if (k > n)
    return 0;
  if (k < 0)
    return 0;
  if (k == n)
    return 1;
  if (k == 0 && n != 0)
    return 1;
  int64_t r = 1;
  for (int64_t d = 1; d <= k; ++d)
  {
    r *= n--;
    r /= d;
  }
  return r;
}

struct boosting
{
  int N;
  float gamma;
  std::string alg;
  vw* all;
  std::shared_ptr<rand_state> _random_state;
  std::vector<std::vector<int64_t> > C;
  std::vector<float> alpha;
  std::vector<float> v;
  int t;
};

//---------------------------------------------------
// Online Boost-by-Majority (BBM)
// --------------------------------------------------
template <bool is_learn>
void predict_or_learn(boosting& o, VW::LEARNER::single_learner& base, example& ec)
{
  label_data& ld = ec.l.simple;

  float final_prediction = 0;

  float s = 0;
  float u = ec.weight;

  if (is_learn)
    o.t++;

  for (int i = 0; i < o.N; i++)
  {
    if (is_learn)
    {
      float k = floorf((float)(o.N - i - s) / 2);
      int64_t c;
      if (o.N - (i + 1) < 0)
        c = 0;
      else if (k > o.N - (i + 1))
        c = 0;
      else if (k < 0)
        c = 0;
      else if (o.C[o.N - (i + 1)][(int64_t)k] != -1)
        c = o.C[o.N - (i + 1)][(int64_t)k];
      else
      {
        c = choose(o.N - (i + 1), (int64_t)k);
        o.C[o.N - (i + 1)][(int64_t)k] = c;
      }

      float w = c * (float)pow((double)(0.5 + o.gamma), (double)k) *
          (float)pow((double)0.5 - o.gamma, (double)(o.N - (i + 1) - k));

      // update ec.weight, weight for learner i (starting from 0)
      ec.weight = u * w;

      base.predict(ec, i);

      // ec.pred.scalar is now the i-th learner prediction on this example
      s += ld.label * ec.pred.scalar;

      final_prediction += ec.pred.scalar;

      base.learn(ec, i);
    }
    else
    {
      base.predict(ec, i);
      final_prediction += ec.pred.scalar;
    }
  }

  ec.weight = u;
  ec.partial_prediction = final_prediction;
  ec.pred.scalar = sign(final_prediction);

  if (ld.label == ec.pred.scalar)
    ec.loss = 0.;
  else
    ec.loss = ec.weight;
}

//-----------------------------------------------------------------
// Logistic boost
//-----------------------------------------------------------------
template <bool is_learn>
void predict_or_learn_logistic(boosting& o, VW::LEARNER::single_learner& base, example& ec)
{
  label_data& ld = ec.l.simple;

  float final_prediction = 0;

  float s = 0;
  float u = ec.weight;

  if (is_learn)
    o.t++;
  float eta = 4.f / sqrtf((float)o.t);

  for (int i = 0; i < o.N; i++)
  {
    if (is_learn)
    {
      float w = 1 / (1 + correctedExp(s));

      ec.weight = u * w;

      base.predict(ec, i);
      float z;
      z = ld.label * ec.pred.scalar;

      s += z * o.alpha[i];

      // if ld.label * ec.pred.scalar < 0, learner i made a mistake

      final_prediction += ec.pred.scalar * o.alpha[i];

      // update alpha
      o.alpha[i] += eta * z / (1 + correctedExp(s));
      if (o.alpha[i] > 2.)
        o.alpha[i] = 2;
      if (o.alpha[i] < -2.)
        o.alpha[i] = -2;

      base.learn(ec, i);
    }
    else
    {
      base.predict(ec, i);
      final_prediction += ec.pred.scalar * o.alpha[i];
    }
  }

  ec.weight = u;
  ec.partial_prediction = final_prediction;
  ec.pred.scalar = sign(final_prediction);

  if (ld.label == ec.pred.scalar)
    ec.loss = 0.;
  else
    ec.loss = ec.weight;
}

template <bool is_learn>
void predict_or_learn_adaptive(boosting& o, VW::LEARNER::single_learner& base, example& ec)
{
  label_data& ld = ec.l.simple;

  float final_prediction = 0, partial_prediction = 0;

  float s = 0;
  float v_normalization = 0, v_partial_sum = 0;
  float u = ec.weight;

  if (is_learn)
    o.t++;
  float eta = 4.f / (float)sqrtf((float)o.t);

  float stopping_point = o._random_state->get_and_update_random();

  for (int i = 0; i < o.N; i++)
  {
    if (is_learn)
    {
      float w = 1 / (1 + correctedExp(s));

      ec.weight = u * w;

      base.predict(ec, i);
      float z;

      z = ld.label * ec.pred.scalar;

      s += z * o.alpha[i];

      if (v_partial_sum <= stopping_point)
      {
        final_prediction += ec.pred.scalar * o.alpha[i];
      }

      partial_prediction += ec.pred.scalar * o.alpha[i];

      v_partial_sum += o.v[i];

      // update v, exp(-1) = 0.36788
      if (ld.label * partial_prediction < 0)
      {
        o.v[i] *= 0.36788f;
      }
      v_normalization += o.v[i];

      // update alpha
      o.alpha[i] += eta * z / (1 + correctedExp(s));
      if (o.alpha[i] > 2.)
        o.alpha[i] = 2;
      if (o.alpha[i] < -2.)
        o.alpha[i] = -2;

      base.learn(ec, i);
    }
    else
    {
      base.predict(ec, i);
      if (v_partial_sum <= stopping_point)
      {
        final_prediction += ec.pred.scalar * o.alpha[i];
      }
      else
      {
        // stopping at learner i
        break;
      }
      v_partial_sum += o.v[i];
    }
  }

  // normalize v vector in training
  if (is_learn)
  {
    for (int i = 0; i < o.N; i++)
    {
      if (v_normalization)
        o.v[i] /= v_normalization;
    }
  }

  ec.weight = u;
  ec.partial_prediction = final_prediction;
  ec.pred.scalar = sign(final_prediction);

  if (ld.label == ec.pred.scalar)
    ec.loss = 0.;
  else
    ec.loss = ec.weight;
}

void save_load_sampling(boosting& o, io_buf& model_file, bool read, bool text)
{
  if (model_file.num_files() == 0)
    return;
  std::stringstream os;
  os << "boosts " << o.N << endl;
  bin_text_read_write_fixed(model_file, (char*)&(o.N), sizeof(o.N), "", read, os, text);

  if (read)
  {
    o.alpha.resize(o.N);
    o.v.resize(o.N);
  }

  for (int i = 0; i < o.N; i++)
    if (read)
    {
      float f;
      model_file.bin_read_fixed((char*)&f, sizeof(f), "");
      o.alpha[i] = f;
    }
    else
    {
      std::stringstream os2;
      os2 << "alpha " << o.alpha[i] << endl;
      bin_text_write_fixed(model_file, (char*)&(o.alpha[i]), sizeof(o.alpha[i]), os2, text);
    }

  for (int i = 0; i < o.N; i++)
    if (read)
    {
      float f;
      model_file.bin_read_fixed((char*)&f, sizeof(f), "");
      o.v[i] = f;
    }
    else
    {
      std::stringstream os2;
      os2 << "v " << o.v[i] << endl;
      bin_text_write_fixed(model_file, (char*)&(o.v[i]), sizeof(o.v[i]), os2, text);
    }

  if (read)
  {
    cerr << "Loading alpha and v: " << endl;
  }
  else
  {
    cerr << "Saving alpha and v, current weighted_examples = "
         << o.all->sd->weighted_labeled_examples + o.all->sd->weighted_unlabeled_examples << endl;
  }
  for (int i = 0; i < o.N; i++)
  {
    cerr << o.alpha[i] << " " << o.v[i] << endl;
  }
  cerr << endl;
}

void return_example(vw& all, boosting& /* a */, example& ec)
{
  output_and_account_example(all, ec);
  VW::finish_example(all, ec);
}

void save_load(boosting& o, io_buf& model_file, bool read, bool text)
{
  if (model_file.num_files() == 0)
    return;
  std::stringstream os;
  os << "boosts " << o.N << endl;
  bin_text_read_write_fixed(model_file, (char*)&(o.N), sizeof(o.N), "", read, os, text);

  if (read)
    o.alpha.resize(o.N);

  for (int i = 0; i < o.N; i++)
    if (read)
    {
      float f;
      model_file.bin_read_fixed((char*)&f, sizeof(f), "");
      o.alpha[i] = f;
    }
    else
    {
      std::stringstream os2;
      os2 << "alpha " << o.alpha[i] << endl;
      bin_text_write_fixed(model_file, (char*)&(o.alpha[i]), sizeof(o.alpha[i]), os2, text);
    }

  if (!o.all->logger.quiet)
  {
    if (read)
      cerr << "Loading alpha: " << endl;
    else
      cerr << "Saving alpha, current weighted_examples = " << o.all->sd->weighted_examples() << endl;
    for (int i = 0; i < o.N; i++) cerr << o.alpha[i] << " " << endl;

    cerr << endl;
  }
}

VW::LEARNER::base_learner* boosting_setup(options_i& options, vw& all)
{
  free_ptr<boosting> data = scoped_calloc_or_throw<boosting>();
  option_group_definition new_options("Boosting");
  new_options.add(make_option("boosting", data->N).keep().help("Online boosting with <N> weak learners"))
      .add(make_option("gamma", data->gamma)
               .default_value(0.1f)
               .help("weak learner's edge (=0.1), used only by online BBM"))
      .add(
          make_option("alg", data->alg)
              .keep()
              .default_value("BBM")
              .help("specify the boosting algorithm: BBM (default), logistic (AdaBoost.OL.W), adaptive (AdaBoost.OL)"));
  options.add_and_parse(new_options);

  if (!options.was_supplied("boosting"))
    return nullptr;

  // Description of options:
  // "BBM" implements online BBM (Algorithm 1 in BLK'15)
  // "logistic" implements AdaBoost.OL.W (importance weighted version
  // 	    of Algorithm 2 in BLK'15)
  // "adaptive" implements AdaBoost.OL (Algorithm 2 in BLK'15,
  // 	    using sampling rather than importance weighting)

  if (!all.logger.quiet)
    cerr << "Number of weak learners = " << data->N << endl;
  if (!all.logger.quiet)
    cerr << "Gamma = " << data->gamma << endl;

  data->C = std::vector<std::vector<int64_t> >(data->N, std::vector<int64_t>(data->N, -1));
  data->t = 0;
  data->all = &all;
  data->_random_state = all.get_random_state();
  data->alpha = std::vector<float>(data->N, 0);
  data->v = std::vector<float>(data->N, 1);

  learner<boosting, example>* l;
  if (data->alg == "BBM")
    l = &init_learner<boosting, example>(
        data, as_singleline(setup_base(options, all)), predict_or_learn<true>, predict_or_learn<false>, data->N, "boosting");
  else if (data->alg == "logistic")
  {
    l = &init_learner<boosting, example>(data, as_singleline(setup_base(options, all)), predict_or_learn_logistic<true>,
        predict_or_learn_logistic<false>, data->N, "boosting-logistic");
    l->set_save_load(save_load);
  }
  else if (data->alg == "adaptive")
  {
    l = &init_learner<boosting, example>(data, as_singleline(setup_base(options, all)), predict_or_learn_adaptive<true>,
        predict_or_learn_adaptive<false>, data->N, "boosting-adaptive");
    l->set_save_load(save_load_sampling);
  }
  else
    THROW("Unrecognized boosting algorithm: \'" << data->alg << "\' Bailing!");

  l->set_finish_example(return_example);

  return make_base(*l);
}
