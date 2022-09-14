#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace impl {

struct C10_API PythonDispatcherTLS {
  static void set_state(SafePyHandle state);
  static SafePyHandle get_state();
  static void reset_state();
};

struct C10_API DisablePythonDispatcher {
  DisablePythonDispatcher() : old_(PythonDispatcherTLS::get_state()) {
    PythonDispatcherTLS::set_state({});
  }
  ~DisablePythonDispatcher() {
    PythonDispatcherTLS::set_state(old_);
  }
  c10::SafePyHandle old_;
};

} // namespace impl
} // namespace c10
