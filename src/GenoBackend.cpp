#include "GenoBackend.h"
#include "Logger.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stop_token>
#include <thread>

void GenoBackend::stream(std::span<const uint32_t> extractIndex, Callback callback) {
    if (streamActive_.exchange(true, std::memory_order_acq_rel)) {
        LOGGER.e(0, "GenoBackend::stream called concurrently on one backend instance.");
    }
    struct StreamActiveReset {
        std::atomic<bool> &active;
        ~StreamActiveReset() { active.store(false, std::memory_order_release); }
    } streamActiveReset{streamActive_};

    if (const std::size_t cap = initialBlockCapacity(); cap > 0) {
        for (auto &block : blocks_) {
            if (block.buf.size() < cap) {
                block.buf.resize(cap);
            }
        }
    }

    std::mutex queueMutex;
    std::condition_variable_any readyCv;
    std::condition_variable_any freeCv;
    std::deque<int> readySlots;
    std::array<bool, 2> slotInUse{false, false};
    bool done = false;
    std::exception_ptr ioError;

    std::jthread ioThread([&](std::stop_token stopToken) {
        gcta_restore_initial_affinity();
        try {
            auto producer = makeProducer(extractIndex);
            while (!stopToken.stop_requested()) {
                int slot = -1;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    freeCv.wait(lock, stopToken, [&]() {
                        return !slotInUse[0] || !slotInUse[1];
                    });
                    if (stopToken.stop_requested()) {
                        break;
                    }
                    slot = !slotInUse[0] ? 0 : 1;
                    slotInUse[static_cast<std::size_t>(slot)] = true;
                }

                if (!producer->nextBlock(blocks_[static_cast<std::size_t>(slot)])) {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    slotInUse[static_cast<std::size_t>(slot)] = false;
                    done = true;
                    break;
                }

                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    readySlots.push_back(slot);
                }
                readyCv.notify_one();
            }

            {
                std::lock_guard<std::mutex> lock(queueMutex);
                done = true;
            }
            readyCv.notify_all();
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                ioError = std::current_exception();
                done = true;
            }
            readyCv.notify_all();
        }
    });

    try {
        while (true) {
            int slot = -1;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                readyCv.wait(lock, [&]() {
                    return ioError || done || !readySlots.empty();
                });
                if (ioError) {
                    break;
                }
                if (readySlots.empty()) {
                    if (done) {
                        break;
                    }
                    continue;
                }
                slot = readySlots.front();
                readySlots.pop_front();
            }

            callback(blocks_[static_cast<std::size_t>(slot)]);

            {
                std::lock_guard<std::mutex> lock(queueMutex);
                slotInUse[static_cast<std::size_t>(slot)] = false;
            }
            freeCv.notify_one();
        }
    } catch (...) {
        ioThread.request_stop();
        freeCv.notify_all();
        if (ioThread.joinable()) {
            ioThread.join();
        }
        throw;
    }

    if (ioThread.joinable()) {
        ioThread.join();
    }
    if (ioError) {
        std::rethrow_exception(ioError);
    }
}
