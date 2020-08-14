#ifndef TVM_SUPPORT_PARALLEL_FOR_H_
#define TVM_SUPPORT_PARALLEL_FOR_H_

#include <deque>
#include <functional>
#include <future>
#include <thread>
#include <mutex>
#include <utility>
#include <vector>

#include <dmlc/logging.h>

namespace tvm {
namespace support {

class ThreadPool {
 public:
  /*!
   * \brief Set the thread number used in this pool.
   * \param n The thread number of this pool.
   */
  void Launch(size_t n = 1) {
    for (std::size_t i = 0; i < n; ++i) {
      threads_.emplace_back([this] { WorkerFunc(); });
    }
  }

  /*!
   * \brief Set the total task number to be executed in this parallel for run batch.
   * \param n The task number of this parallel for run batch.
   */
  void BeginBatch(int n) {
    CHECK(is_finished_) << "Last run batch didn't finished.";
    finish_ct_ = n;
    is_finished_ = n <= 0;
  }

  /*!
   * \brief Add run task to task queue. The task added will be run in thread pool immediately.
   * \param f The task function to be executed.
   * \param args The args of the task function.
   * \return The result of the task function.
   */
  template <typename F, typename... Args, typename R = typename std::result_of<F(Args...)>::type>
  std::future<R> Enqueue(F&& f, Args&&... args) {
    std::packaged_task<R()> p(std::bind(f, args...));

    auto r = p.get_future();
    {
      std::unique_lock<std::mutex> l(m_);
      work_.emplace_back(std::move(p));
    }
    work_signal_.notify_one();
    return r;
  }

  /*! \brief Wait until the parallel for run batch is finished. */
  void WaitBatch() {
    std::unique_lock<std::mutex> l(finish_mutex_);
    if (!is_finished_) {
      finish_signal_.wait(l);
    }
    CHECK(is_finished_);
  }

  /*! \brief Stop the running process. */
  void Abort() {
    CancelPending();
    Join();
  }

  /*! \brief Cancel all the tasks in task queue. */
  void CancelPending() {
    std::unique_lock<std::mutex> l(m_);
    work_.clear();
  }

  /*! \brief Wait until all of the threads are finished. */
  void Join() {
    {
      std::unique_lock<std::mutex> l(m_);
      for (size_t i = 0; i < threads_.size(); ++i) {
        work_.push_back({});
      }
    }
    work_signal_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
    threads_.clear();
  }

  /*!
   * \brief Get the working thread number of this pool.
   * \return The thread number of this pool.
   */
  size_t NumWorkers() { return threads_.size(); }

  static const int REFRESH_EVERY = 128;
  static ThreadPool& Global();

  ~ThreadPool() { Join(); }

 private:
  void WorkerFunc() {
    std::packaged_task<void()> f;
    while (true) {
      {
        std::unique_lock<std::mutex> l(m_);
        if (work_.empty()) {
          work_signal_.wait(l, [&] { return !work_.empty(); });
        }
        f = std::move(work_.front());
        work_.pop_front();
      }
      if (!f.valid()) {
        return;
      }

      f();

      finish_ct_--;
      if (finish_ct_ == 0) {
        std::unique_lock<std::mutex> l(finish_mutex_);
        is_finished_ = true;
        finish_signal_.notify_one();
      }
    }
  }

  std::mutex m_;
  std::condition_variable work_signal_;
  std::deque<std::packaged_task<void()>> work_;
  std::vector<std::thread> threads_;

  bool is_finished_ = true;
  std::mutex finish_mutex_;
  std::atomic<int> finish_ct_;
  std::condition_variable finish_signal_;
};

void parallel_for(int begin, int end, const std::function<void(int)>& f, int step = 1);

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_PARALLEL_FOR_H_
