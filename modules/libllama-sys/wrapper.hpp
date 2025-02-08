#include "llama.h"

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
  if (prio == GGML_SCHED_PRIO_NORMAL) {
    return true;
  }

  DWORD p = NORMAL_PRIORITY_CLASS;
  switch (prio) {
  case GGML_SCHED_PRIO_NORMAL:
    p = NORMAL_PRIORITY_CLASS;
    break;
  case GGML_SCHED_PRIO_MEDIUM:
    p = ABOVE_NORMAL_PRIORITY_CLASS;
    break;
  case GGML_SCHED_PRIO_HIGH:
    p = HIGH_PRIORITY_CLASS;
    break;
  case GGML_SCHED_PRIO_REALTIME:
    p = REALTIME_PRIORITY_CLASS;
    break;
  }

  if (!SetPriorityClass(GetCurrentProcess(), p)) {
    return false;
  }

  return true;
}

#else // MacOS and POSIX
#include <sys/resource.h>
#include <sys/types.h>

bool set_process_priority(enum ggml_sched_priority prio) {
  if (prio == GGML_SCHED_PRIO_NORMAL) {
    return true;
  }

  int p = 0;
  switch (prio) {
  case GGML_SCHED_PRIO_NORMAL:
    p = 0;
    break;
  case GGML_SCHED_PRIO_MEDIUM:
    p = -5;
    break;
  case GGML_SCHED_PRIO_HIGH:
    p = -10;
    break;
  case GGML_SCHED_PRIO_REALTIME:
    p = -20;
    break;
  }

  if (!setpriority(PRIO_PROCESS, 0, p)) {
    return false;
  }
  return true;
}

#endif

struct ggml_threadpool *
libllama_init_threadpool(enum ggml_backend_dev_type dev_type,
                         struct ggml_threadpool_params tpp) {
  auto *reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(dev_type));
  auto *ggml_threadpool_new_fn =
      (decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_new");

  set_process_priority(tpp.prio);

  struct ggml_threadpool *threadpool = ggml_threadpool_new_fn(&tpp);
  if (!threadpool) {
    return nullptr;
  }

  return threadpool;
}

int libllama_free_threadpool(enum ggml_backend_dev_type dev_type,
                             struct ggml_threadpool *threadpool) {
  auto *reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(dev_type));
  auto *ggml_threadpool_free_fn =
      (decltype(ggml_threadpool_free) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_free");

  if (nullptr == threadpool) {
    return 1;
  }

  ggml_threadpool_free_fn(threadpool);

  return 0;
}
