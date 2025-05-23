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

#include <portability/Config.h>
#include <portability/IOVec.h>
#include <portability/SysTypes.h>

#if FOLLY_HAVE_PREADV || FOLLY_HAVE_PWRITEV
#include <sys/uio.h>
#endif

namespace folly {
#if !FOLLY_HAVE_PREADV
ssize_t preadv(int fd, const iovec* iov, int count, off_t offset);
#else
using ::preadv;
#endif
#if !FOLLY_HAVE_PWRITEV
ssize_t pwritev(int fd, const iovec* iov, int count, off_t offset);
#else
using ::pwritev;
#endif
} // namespace folly

#ifdef _WIN32
extern "C" ssize_t readv(int fd, const iovec* iov, int count);
extern "C" ssize_t writev(int fd, const iovec* iov, int count);
#endif

namespace folly {
#ifdef IOV_MAX // not defined on Android
constexpr size_t kIovMax = IOV_MAX;
#else
constexpr size_t kIovMax = UIO_MAXIOV;
#endif
} // namespace folly
