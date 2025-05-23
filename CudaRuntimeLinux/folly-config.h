

#pragma once

#ifdef __APPLE__
#include <TargetConditionals.h> // @manual
#endif

#if !defined(FOLLY_MOBILE)
#if defined(__ANDROID__) || \
    (defined(__APPLE__) &&  \
     (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define FOLLY_MOBILE 1
#else
#define FOLLY_MOBILE 0
#endif
#endif // FOLLY_MOBILE

#define FOLLY_HAVE_PTHREAD 1
#define FOLLY_HAVE_PTHREAD_ATFORK 1

#define FOLLY_HAVE_LIBGFLAGS 1

#define FOLLY_HAVE_LIBGLOG 1

// 以下表示使用JEMALLOC管理内存
// #define JEMALLOC_MANGLE
#define FOLLY_USE_JEMALLOC 1

#if __has_include(<features.h>)
#include <features.h>
#endif

#define FOLLY_HAVE_ACCEPT4 1
#define FOLLY_HAVE_GETRANDOM
#define FOLLY_HAVE_PREADV 0  // linux系统才支持PREADV
#define FOLLY_HAVE_PWRITEV 0  // linux系统才支持PWRITE
#define FOLLY_HAVE_CLOCK_GETTIME 1
#define FOLLY_HAVE_PIPE2 1

#define FOLLY_HAVE_IFUNC 1
#define FOLLY_HAVE_UNALIGNED_ACCESS 1
#define FOLLY_HAVE_VLA 1
#define FOLLY_HAVE_WEAK_SYMBOLS 0 // __attribute__((__weak__))属性当一个函数或变量被声明为弱符号时，如果在链接的其他文件中找到了同名的强符号（strong symbol），那么强符号会覆盖弱符号。
#define FOLLY_HAVE_LINUX_VDSO 1
#define FOLLY_HAVE_MALLOC_USABLE_SIZE 1
#define FOLLY_HAVE_INT128_T 0  // 只有gcc或clang支持__int128
#define FOLLY_HAVE_WCHAR_SUPPORT 1
#define FOLLY_HAVE_EXTRANDOM_SFMT19937 1
#define HAVE_VSNPRINTF_ERRORS 1

#define FOLLY_HAVE_LIBUNWIND 1
#define FOLLY_HAVE_DWARF 1
#define FOLLY_HAVE_ELF 1
#define FOLLY_HAVE_SWAPCONTEXT 1
#define FOLLY_HAVE_BACKTRACE 1
#define FOLLY_USE_SYMBOLIZER 1
#define FOLLY_DEMANGLE_MAX_SYMBOL_SIZE 1024

#define FOLLY_HAVE_SHADOW_LOCAL_WARNINGS 1

#define FOLLY_HAVE_LIBLZ4 1
#define FOLLY_HAVE_LIBLZMA 1
#define FOLLY_HAVE_LIBSNAPPY 1
#define FOLLY_HAVE_LIBZ 1
#define FOLLY_HAVE_LIBZSTD 1
#define FOLLY_HAVE_LIBBZ2 1

#define FOLLY_LIBRARY_SANITIZE_ADDRESS 0 // ASAN 是一种用于检测内存错误（如缓冲区溢出和使用后释放等）的工具。它通过在编译时插入检查代码来工作。

#define FOLLY_SUPPORT_SHARED_LIBRARY 1

#define FOLLY_HAVE_LIBRT


