/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "FaissException.h"
#include <cstdio>
#include <execinfo.h>

namespace faiss {

FaissException::FaissException(const std::string& m)
    : msg(m) {
}

FaissException::FaissException(const std::string& m,
                               const char* funcName,
                               const char* file,
                               int line) {
  int size = snprintf(nullptr, 0, "Error in %s at %s:%d: %s",
                      funcName, file, line, m.c_str());
  msg.resize(size + 1);
  snprintf(&msg[0], msg.size(), "Error in %s at %s:%d: %s",
           funcName, file, line, m.c_str());
}

static void print_trace (void)
{
  void *array[20];
  size_t size;
  char **strings;
  size_t i;

  size = backtrace (array, 20);
  strings = backtrace_symbols (array, size);

  fprintf (stderr, "Obtained %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
     fprintf (stderr, ">> %s\n", strings[i]);

  free (strings);
}

void abortWithStack() {
  fprintf(stderr, "==============\nBacktrace:\n");
  print_trace();
  fprintf(stderr, "==============\n");
  abort();
}

}
