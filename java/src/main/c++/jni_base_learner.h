#ifndef VW_BASE_LEARNER_H
#define VW_BASE_LEARNER_H

#include <jni.h>

void throw_java_exception(JNIEnv *env, const char* name, const char* msg);
void rethrow_cpp_exception_as_java_exception(JNIEnv *env);

example* read_example(JNIEnv *env, jstring example_string, vw* vwInstance);
example* read_example(const char* example_string, vw* vwInstance);

// It would appear that after reading posts like
// http://stackoverflow.com/questions/6458612/c0x-proper-way-to-receive-a-lambda-as-parameter-by-reference
// and
// http://stackoverflow.com/questions/3203305/write-a-function-that-accepts-a-lambda-expression-as-argument
// it is more efficient to use another type parameter instead of std::function<T(example*)>
// but more difficult to read.
template<typename T, typename F>
T base_predict(
  JNIEnv *env,
  example* ex,
  bool learn,
  vw* vwInstance,
  const F& predictor,
  const bool predict)
{ T result = 0;
  try
  { if (learn)
      as_singleline(vwInstance->l)->learn(*ex);
    else
      as_singleline(vwInstance->l)->predict(*ex);

    if (predict)
      result = predictor(ex, env);

    as_singleline(vwInstance->l)->finish_example(*vwInstance, *ex);
  }
  catch (...)
  { rethrow_cpp_exception_as_java_exception(env);
  }
  return result;
}

template<typename T, typename F>
T base_predict(
  JNIEnv *env,
  jstring example_string,
  jboolean learn,
  jlong vwPtr,
  const F& predictor)
{ vw* vwInstance = (vw*)vwPtr;
  example* ex = read_example(env, example_string, vwInstance);
  return base_predict<T>(env, ex, learn, vwInstance, predictor, true);
}

template<typename T, typename F>
T base_predict(
  JNIEnv *env,
  jobjectArray example_strings,
  jboolean learn,
  jlong vwPtr,
  const F& predictor)
{ vw* vwInstance = (vw*)vwPtr;
  int example_count = env->GetArrayLength(example_strings);

  // When doing multiline prediction the final result is stored in the FIRST example parsed.
  multi_ex ex_coll = v_init<example*>();

  example* first_example = NULL;
  for (int i=0; i<example_count; i++)
  { jstring example_string = (jstring) (env->GetObjectArrayElement(example_strings, i));
    example* ex = read_example(env, example_string, vwInstance);
    ex_coll.push_back(ex);
    if (i == 0)
      first_example = ex;
  }
  env->DeleteLocalRef(example_strings);

  try
  { if (learn)
      as_multiline(vwInstance->l)->learn(ex_coll);
    else
      as_multiline(vwInstance->l)->predict(ex_coll);
  }
  catch (...)
  { rethrow_cpp_exception_as_java_exception(env);
  }
  
  as_multiline(vwInstance->l)->finish_example(*vwInstance, ex_coll);

  return predictor(first_example, env);
}

#endif // VW_BASE_LEARNER_H
