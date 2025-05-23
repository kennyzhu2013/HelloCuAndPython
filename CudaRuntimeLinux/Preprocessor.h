/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <CPortability.h>

/**
 * Necessarily evil preprocessor-related amenities.
 */

// MSVC's preprocessor is a pain, so we have to
// forcefully expand the VA args in some places.
#define FB_VA_GLUE(a, b) a b

/**
 * FB_ONE_OR_NONE(hello, world) expands to hello and
 * FB_ONE_OR_NONE(hello) expands to nothing. This macro is used to
 * insert or eliminate text based on the presence of another argument.
 */
#define FB_ONE_OR_NONE(a, ...) FB_VA_GLUE(FB_THIRD, (a, ##__VA_ARGS__, a))
#define FB_THIRD(a, b, ...) __VA_ARGS__

/**
 * Helper macro that extracts the first argument out of a list of any
 * number of arguments.
 */
#define FB_ARG_1(a, ...) a

/**
 * Helper macro that extracts the second argument out of a list of any
 * number of arguments. If only one argument is given, it returns
 * that.
 */
#ifdef _MSC_VER
// GCC refuses to expand this correctly if this macro itself was
// called with FB_VA_GLUE :(
#define FB_ARG_2_OR_1(...) \
  FB_VA_GLUE(FB_ARG_2_OR_1_IMPL, (__VA_ARGS__, __VA_ARGS__))
#else
#define FB_ARG_2_OR_1(...) FB_ARG_2_OR_1_IMPL(__VA_ARGS__, __VA_ARGS__)
#endif
// Support macro for the above
#define FB_ARG_2_OR_1_IMPL(a, b, ...) b

/**
 * Helper macro that provides a way to pass argument with commas in it to
 * some other macro whose syntax doesn't allow using extra parentheses.
 * Example:
 *
 *   #define MACRO(type, name) type name
 *   MACRO(FB_SINGLE_ARG(std::pair<size_t, size_t>), x);
 *
 */
#define FB_SINGLE_ARG(...) __VA_ARGS__

#define FOLLY_PP_DETAIL_APPEND_VA_ARG(...) , ##__VA_ARGS__

/**
 * Helper macro that just ignores its parameters.
 */
#define FOLLY_IGNORE(...)

/**
 * Helper macro that just ignores its parameters and inserts a semicolon.
 */
#define FOLLY_SEMICOLON(...) ;

/**
 * FB_ANONYMOUS_VARIABLE(str) introduces an identifier starting with
 * str and ending with a number that varies with the line.
 */
#ifndef FB_ANONYMOUS_VARIABLE
#define FB_CONCATENATE_IMPL(s1, s2) s1##s2
#define FB_CONCATENATE(s1, s2) FB_CONCATENATE_IMPL(s1, s2)
#ifdef __COUNTER__
// Modular builds build each module with its own preprocessor state, meaning
// `__COUNTER__` no longer provides a unique number across a TU.  Instead of
// calling back to just `__LINE__`, use a mix of `__COUNTER__` and `__LINE__`
// to try provide as much uniqueness as possible.
#if FOLLY_HAS_FEATURE(modules)
#define FB_ANONYMOUS_VARIABLE(str) \
  FB_CONCATENATE(FB_CONCATENATE(FB_CONCATENATE(str, __COUNTER__), _), __LINE__)
#else
#define FB_ANONYMOUS_VARIABLE(str) FB_CONCATENATE(str, __COUNTER__)
#endif
#else
#define FB_ANONYMOUS_VARIABLE(str) FB_CONCATENATE(str, __LINE__)
#endif
// FB_ANONYMOUS_VARIABLE_ODR_SAFE doesn't rely on __COUNTER__ and is safe to use
// in headers that should not violate the one-definition rule (ODR). It is
// especially useful for C++ modules that check for ODR violations.
#define FB_ANONYMOUS_VARIABLE_ODR_SAFE(str) FB_CONCATENATE(str, __LINE__)
#endif

/**
 * Use FOLLY_PP_STRINGIZE(x) when you'd want to do what #x does inside
 * another macro expansion.
 */
#define FOLLY_PP_STRINGIZE(x) #x

/**
 * Use FOLLY_PP_STRINGIZE_MACRO(x) when you want the string representation
 * of a non-string c++ preprocessing macro value, ex
 * FOLLY_PP_STRINGIZE_MACRO(__LINE__).
 */
#define FOLLY_PP_STRINGIZE_MACRO(x) FOLLY_PP_STRINGIZE(x)

#define FOLLY_PP_DETAIL_NARGS_1( \
    dummy,                       \
    _15,                         \
    _14,                         \
    _13,                         \
    _12,                         \
    _11,                         \
    _10,                         \
    _9,                          \
    _8,                          \
    _7,                          \
    _6,                          \
    _5,                          \
    _4,                          \
    _3,                          \
    _2,                          \
    _1,                          \
    _0,                          \
    ...)                         \
  _0
#define FOLLY_PP_DETAIL_NARGS(...) \
  FOLLY_PP_DETAIL_NARGS_1(         \
      dummy,                       \
      ##__VA_ARGS__,               \
      15,                          \
      14,                          \
      13,                          \
      12,                          \
      11,                          \
      10,                          \
      9,                           \
      8,                           \
      7,                           \
      6,                           \
      5,                           \
      4,                           \
      3,                           \
      2,                           \
      1,                           \
      0)

#define FOLLY_PP_DETAIL_FOR_EACH_REC_0(fn, ...)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_1(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_0(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_2(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_1(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_3(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_2(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_4(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_3(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_5(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_4(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_6(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_5(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_7(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_6(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_8(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_7(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_9(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_8(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_10(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_9(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_11(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_10(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_12(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_11(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_13(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_12(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_14(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_13(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_REC_15(fn, a, ...) \
  fn(a) FOLLY_PP_DETAIL_FOR_EACH_REC_14(fn, __VA_ARGS__)

#define FOLLY_PP_DETAIL_FOR_EACH_2(fn, n, ...) \
  FOLLY_PP_DETAIL_FOR_EACH_REC_##n(fn, __VA_ARGS__)
#define FOLLY_PP_DETAIL_FOR_EACH_1(fn, n, ...) \
  FOLLY_PP_DETAIL_FOR_EACH_2(fn, n, __VA_ARGS__)

/**
 *  FOLLY_PP_FOR_EACH
 *
 *  Used to invoke a preprocessor macro, the name of which is passed as the
 *  first argument, once for each subsequent variadic argument.
 *
 *  At present, supports [0, 16) arguments.
 *
 *  This input:
 *
 *    #define DOIT(a) go_do_it(a);
 *    FOLLY_PP_FOR_EACH(DOIT, 3, 5, 7)
 *    #undef DOIT
 *
 *  Expands to this output (with whitespace adjusted for clarity):
 *
 *    go_do_it(3);
 *    go_do_it(5);
 *    go_do_it(7);
 */
#define FOLLY_PP_FOR_EACH(fn, ...) \
  FOLLY_PP_DETAIL_FOR_EACH_1(      \
      fn, FOLLY_PP_DETAIL_NARGS(__VA_ARGS__), __VA_ARGS__)

#if defined(U)
#error defined(U) // literal U is used below
#endif

//  FOLLY_PP_CONSTINIT_LINE_UNSIGNED
//
//  MSVC with /ZI has a special backing variable for __LINE__ which is not a
//  literal - but token-pasting __LINE__ suppresses this backing variable. This
//  is done in MSVC to support its edit-and-continue feature.
//
//  This macro evaluates to:
//    __LINE__ ## U
//
//  So this macro may be ill-suited to cases which need exactly __LINE__.
//
//  Documentation:
//    https://docs.microsoft.com/en-us/cpp/build/reference/z7-zi-zi-debug-information-format?view=msvc-170#zi-1
//  Workaround:
//    https://stackoverflow.com/questions/57137351/line-is-not-constexpr-in-msvc
#define FOLLY_PP_CONSTINIT_LINE_UNSIGNED FB_CONCATENATE(__LINE__, U)
