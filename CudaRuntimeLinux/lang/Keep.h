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
#include <CppAttributes.h>

//  FOLLY_KEEP
//
//  When applied to a function, prevents removal of the function. Useful for
//  marking functions which it is desirable to keep.
//
//  A common use-case is for keeping unused functions in test or benchmark
//  programs which serve as convenient targets for inspecting how the compiler
//  translates source code to machine code. Such functions may be disassembled
//  for inspection in the debugger of with the objdump tool, and are often, for
//  lack of imagination, called check functions in test and benchmark programs.
//
//  Example:
//
//      extern "C" FOLLY_KEEP void check_throw_int_0() {
//        throw 0;
//      }
//
//      $ objdump -dCM intel-mnemonic path/to/binary | less
//      # less permits searching for check_throw_int_0
//
//      $ gdb path/to/binary -q -batch -ex 'disassemble check_throw_int_0'
//
//  Functions may be removed when building with both function-sections and
//  gc-sections. It is tricky to keep them; this utility captures the steps
//  required.
//
//  In order for this mechanism to work, this header must be included. This
//  mechanism relies on the hidden global variable below.
//
//  The linker, when asked to with --gc-sections, may throw out unreferenced
//  sections. When the compiler emits each function into its own section, as it
//  does when asked to with -ffunction-sections, the linker will throw out
//  unreferenced functions. The key is to move all kept functions into a single
//  section, avoiding the behavior of -ffunction-sections, and then force the
//  compiler to emit some reference to at least one function in that section.
//  This way, the linker will see at least one reference to the kept section,
//  and so will not throw it out.
#if __GNUC__ && __linux__
#define FOLLY_KEEP \
  [[gnu::section(".text.folly.keep"), gnu::used, FOLLY_ATTR_GNU_RETAIN]]
#else
#define FOLLY_KEEP
#endif

#if __GNUC__ && __linux__
#if defined(__clang__) || FOLLY_X64 || (FOLLY_ARM && !FOLLY_AARCH64)
#define FOLLY_KEEP_DETAIL_ATTR_NAKED [[gnu::naked]]
#else
#define FOLLY_KEEP_DETAIL_ATTR_NAKED
#endif
#define FOLLY_KEEP_DETAIL_ATTR_NOINLINE [[gnu::noinline]]
#else
#define FOLLY_KEEP_DETAIL_ATTR_NAKED
#define FOLLY_KEEP_DETAIL_ATTR_NOINLINE
#endif

namespace folly {
namespace detail {

//  marking this as [[gnu::naked]] gets clang not to emit any text for this
FOLLY_KEEP FOLLY_KEEP_DETAIL_ATTR_NAKED static void keep_anchor() {}

class keep {
 public:
  //  must accept the anchor function as an argument
  FOLLY_KEEP_DETAIL_ATTR_NOINLINE explicit keep(void (*)()) noexcept {}
};

//  noinline ctor and trivial dtor minimize the text size of this
static keep keep_instance{keep_anchor};

//  weak and noinline to prevent the compiler from eliding calls
template <typename... T>
FOLLY_ATTR_WEAK FOLLY_NOINLINE void keep_sink(T&&...) {}
template <typename... T>
FOLLY_ATTR_WEAK FOLLY_NOINLINE void keep_sink_nx(T&&...) noexcept {}

} // namespace detail
} // namespace folly
