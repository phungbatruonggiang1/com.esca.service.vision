#include "service.hpp"

#include <cstdio>

#include "esca/camera/client.hpp"

namespace esca::vision {

namespace {
// Negative returns are errors, as in the camera service; the detail is in
// GetErrorMessage().
constexpr int32_t kOk = 0;
constexpr int32_t kErr = -1;

/// How long to wait for a frame before deciding the camera has gone away.
/// Generous: the camera runs at 30 fps, so anything beyond a few hundred
/// milliseconds already means something is wrong.
constexpr int kFrameTimeoutMs = 2000;
} // namespace

VisionService::VisionService(sdbus::IConnection &connection, const Config &config)
    : connection_(connection), config_(config)
{
    every_ = config_.every == 0 ? 1 : config_.every;

    object_ = sdbus::createObject(connection_, sdbus::ObjectPath{ kObjectPath });

    object_->addVTable(
        sdbus::registerMethod("Start").implementedAs([this] { return start(); }),
        sdbus::registerMethod("Stop").implementedAs([this] { return stop(); }),
        sdbus::registerMethod("SetThreshold")
            .implementedAs([this](const double &value) { return set_threshold(value); }),
        sdbus::registerMethod("SetEvery")
            .implementedAs([this](const uint32_t &n) { return set_every(n); }),
        sdbus::registerMethod("GetState").implementedAs([this] { return get_state(); }),
        sdbus::registerMethod("GetErrorMessage").implementedAs([this] {
            std::lock_guard<std::mutex> lock(mutex_);
            return last_error_;
        }),
        // frame_seq, timestamp_ns, inference_ms, and the objects themselves.
        // The sequence number is the whole point: it is what lets a viewer
        // know which frame these boxes belong to.
        sdbus::registerSignal("Detections")
            .withParameters<uint64_t, uint64_t, double,
                            std::vector<sdbus::Struct<std::string, uint16_t, double, double,
                                                      double, double, double>>>(),
        sdbus::registerSignal("FaultEvent").withParameters<std::string, std::string>())
        .forInterface(kInterface);
}

VisionService::~VisionService()
{
    // Safe to block here, unlike in stop_worker(): the event loop has already
    // left, so a signal the worker is still emitting cannot be waiting on a
    // lock this thread holds.
    running_ = false;
    if (worker_.joinable())
        worker_.join();
}

bool VisionService::load(std::string &error)
{
    if (!detector_.load(config_.detector, error))
        return false;

    std::fprintf(stderr, "esca-visiond: model %s, input %dx%d, XNNPACK %s, %d threads\n",
                 detector_.model_path().c_str(), detector_.input_width(),
                 detector_.input_height(), detector_.accelerated() ? "on" : "off",
                 config_.detector.threads);
    return true;
}

void VisionService::autostart()
{
    if (!config_.autostart)
        return;

    std::string error;
    if (!start_worker(error)) {
        std::fprintf(stderr, "esca-visiond: autostart: %s\n", error.c_str());
        return;
    }
    // Counts as one client, so a viewer calling Stop() does not switch off a
    // detector that was asked to run from boot.
    std::lock_guard<std::mutex> lock(mutex_);
    ++clients_;
}

// --- D-Bus methods ---------------------------------------------------------

int32_t VisionService::start()
{
    std::string error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++clients_;
    }
    if (!start_worker(error)) {
        std::lock_guard<std::mutex> lock(mutex_);
        --clients_;
        last_error_ = error;
        return kErr;
    }
    return kOk;
}

int32_t VisionService::stop()
{
    bool idle = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (clients_ > 0)
            --clients_;
        idle = clients_ == 0;
    }
    // Reference counted: one viewer closing must not blind another. The
    // detector only goes quiet once the last one has left.
    if (idle)
        stop_worker();
    return kOk;
}

int32_t VisionService::set_threshold(double value)
{
    if (value < 0.0 || value > 1.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "threshold must be between 0 and 1";
        return kErr;
    }
    detector_.set_threshold(static_cast<float>(value));
    return kOk;
}

int32_t VisionService::set_every(uint32_t n)
{
    every_ = n == 0 ? 1 : n;
    return kOk;
}

sdbus::Struct<bool, std::string, double, uint32_t> VisionService::get_state()
{
    return { running_.load(), detector_.model_path(), avg_ms_.load(), every_.load() };
}

// --- worker ----------------------------------------------------------------

bool VisionService::start_worker(std::string &error)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_)
        return true;

    // A previous run may still be winding down. Joining it here is safe only
    // because worker_alive_ says it has already returned, so join() does not
    // block -- see stop_worker() for why blocking here would deadlock.
    if (worker_alive_) {
        error = "the previous run has not finished yet; try again in a moment";
        return false;
    }
    if (worker_.joinable())
        worker_.join();

    running_ = true;
    worker_alive_ = true;
    try {
        worker_ = std::thread([this] {
            run();
            worker_alive_ = false;
        });
    } catch (const std::exception &e) {
        running_ = false;
        worker_alive_ = false;
        error = std::string("cannot start worker: ") + e.what();
        return false;
    }
    return true;
}

void VisionService::stop_worker()
{
    // Only raise the flag. The worker notices it within one frame timeout and
    // returns on its own.
    //
    // It must not be joined here. This runs on the bus thread inside a D-Bus
    // method handler, and sdbus-c++ holds the connection's lock for the whole
    // handler; the worker emits Detections on that same connection, so it
    // would block on that lock while the handler blocked on join() -- both
    // threads stuck, the service never answering another call again. That
    // deadlock is exactly what happened the first time this was written.
    //
    // The destructor does join, which is safe: by then the event loop has
    // exited and nothing holds the connection lock.
    running_ = false;
}

void VisionService::run()
{
    using namespace esca::camera;

    CameraClient camera;
    const auto ids = camera.device_list();
    if (ids.empty()) {
        fault("camera", "no cameras (is esca-camerad running?)");
        running_ = false;
        return;
    }

    std::string error;
    if (camera.open(ids.front(), error) < 0) {
        fault("camera", "open: " + error);
        running_ = false;
        return;
    }
    if (!camera.start(error)) {
        fault("camera", "start: " + error);
        running_ = false;
        return;
    }

    const FrameFormat format = camera.frame_format();
    std::fprintf(stderr, "esca-visiond: detecting on %s, %ux%u stride %u\n", ids.front().c_str(),
                 format.width, format.height, format.stride);

    std::vector<uint8_t> frame;
    std::vector<Detection> objects;
    FrameInfo info;
    uint32_t since_last = 0;
    double total_ms = 0.0;
    uint32_t done = 0;
    bool starved = false;

    while (running_) {
        if (!camera.next_frame(frame, info, kFrameTimeoutMs)) {
            // Not fatal on its own: another client may have stopped the
            // camera, and it may come back. Report the first timeout of a
            // run and then stay quiet, or a disconnected sensor fills the
            // journal with one line every two seconds.
            if (!starved) {
                fault("camera", "timed out waiting for a frame");
                starved = true;
            }
            continue;
        }
        starved = false;

        // Dropping frames is deliberate. Capture runs at 30 fps and the model
        // does not; the newest frame is the only one worth looking at, and a
        // backlog would only make the boxes staler.
        if (++since_last < every_.load())
            continue;
        since_last = 0;

        double ms = 0.0;
        if (!detector_.run(frame.data(), format.width, format.height, format.stride, objects, ms,
                           error)) {
            fault("detector", error);
            break;
        }

        total_ms += ms;
        ++done;
        avg_ms_ = total_ms / done;
        inferences_ = done;

        emit_detections(info.sequence, info.timestamp_ns, ms, objects);
    }

    camera.stop();
    camera.close();
    running_ = false;
    std::fprintf(stderr, "esca-visiond: stopped after %u inferences\n", done);
}

void VisionService::emit_detections(uint64_t frame_seq, uint64_t timestamp_ns,
                                    double inference_ms, const std::vector<Detection> &objects)
{
    std::vector<sdbus::Struct<std::string, uint16_t, double, double, double, double, double>> wire;
    wire.reserve(objects.size());
    for (const Detection &d : objects)
        wire.emplace_back(d.label, d.class_id, d.score, d.left, d.top, d.right, d.bottom);

    try {
        // Emitting a signal is thread-safe by design in sdbus-c++, which is
        // what lets this run on the worker thread while main runs the loop.
        object_->emitSignal(sdbus::SignalName{ "Detections" })
            .onInterface(sdbus::InterfaceName{ kInterface })
            .withArguments(frame_seq, timestamp_ns, inference_ms, wire);
    } catch (const sdbus::Error &e) {
        std::fprintf(stderr, "esca-visiond: cannot emit Detections: %s\n", e.getMessage().c_str());
    }
}

void VisionService::fault(const std::string &where, const std::string &what)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = what;
    }
    std::fprintf(stderr, "esca-visiond: %s: %s\n", where.c_str(), what.c_str());

    try {
        object_->emitSignal(sdbus::SignalName{ "FaultEvent" })
            .onInterface(sdbus::InterfaceName{ kInterface })
            .withArguments(where, what);
    } catch (const sdbus::Error &) {
    }
}

} // namespace esca::vision
