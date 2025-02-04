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

int libllama_init_attach_cpu_threadpool(llama_context *ctx,
                                 struct ggml_threadpool_params tpp,
                                 struct ggml_threadpool_params tpp_batch) {
  auto *reg = ggml_backend_dev_backend_reg(
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
  auto *ggml_threadpool_new_fn =
      (decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_new");
  auto *ggml_threadpool_free_fn =
      (decltype(ggml_threadpool_free) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_free");

  set_process_priority(tpp.prio);

  struct ggml_threadpool *threadpool_batch = NULL;
  if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
    threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
    if (!threadpool_batch) {
      return 1;
    }

    // Start the non-batch threadpool in the paused state
    tpp.paused = true;
  }

  struct ggml_threadpool *threadpool = ggml_threadpool_new_fn(&tpp);
  if (!threadpool) {
    return 1;
  }

  llama_attach_threadpool(ctx, threadpool, threadpool_batch);

  return 0;
}
