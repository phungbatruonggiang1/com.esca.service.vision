// The D-Bus face of the vision service.
//
// It owns a camera client and a detector, and turns frames into Detections
// signals. Nothing here touches pixels on the bus: the frame is read from the
// camera service's shared memory, and only the resulting list of objects is
// published.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <sdbus-c++/sdbus-c++.h>

#include "detector.hpp"

namespace esca::vision {

class VisionService {
public:
    struct Config {
        Detector::Config detector;
        uint32_t every = 3;      ///< run the model on every Nth captured frame
        bool autostart = false;  ///< detect from boot instead of waiting for Start()
    };

    VisionService(sdbus::IConnection &connection, const Config &config);
    ~VisionService();

    /// Load the model. Done before entering the event loop so a bad model
    /// path fails loudly at startup rather than on the first Start().
    bool load(std::string &error);

    /// Begin detecting if the configuration asks for it.
    void autostart();

private:
    // --- D-Bus methods -----------------------------------------------------
    int32_t start();
    int32_t stop();
    int32_t set_threshold(double value);
    int32_t set_every(uint32_t n);
    sdbus::Struct<bool, std::string, double, uint32_t> get_state();

    // --- worker ------------------------------------------------------------
    void run();
    void emit_detections(uint64_t frame_seq, uint64_t timestamp_ns, double inference_ms,
                         const std::vector<Detection> &objects);
    void fault(const std::string &where, const std::string &what);

    bool start_worker(std::string &error);
    void stop_worker();

    sdbus::IConnection &connection_;
    std::unique_ptr<sdbus::IObject> object_;

    Config config_;
    Detector detector_;

    mutable std::mutex mutex_;
    std::thread worker_;
    std::atomic<bool> running_{ false };       ///< asked to run
    std::atomic<bool> worker_alive_{ false };  ///< thread entered and has not returned
    std::atomic<uint32_t> every_{ 3 };
    int clients_ = 0;  ///< Start() calls outstanding; the worker runs while > 0

    // Rolling average, for GetState and for the status line a viewer draws.
    std::atomic<double> avg_ms_{ 0.0 };
    std::atomic<uint32_t> inferences_{ 0 };

    std::string last_error_;
};

} // namespace esca::vision
