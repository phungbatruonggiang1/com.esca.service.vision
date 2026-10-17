// What an application links against to learn what the camera is looking at.
//
// Pixels never cross the bus. The application reads frames from the camera
// service's shared memory itself; this client only brings it the list of
// objects found in them -- a few hundred bytes per frame, which is what D-Bus
// is good at. The two halves are tied together by the camera's frame sequence
// number: both sides read the same ring, so both see the same numbering, and
// no clock needs to be synchronised to know which frame a box belongs to.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace esca::vision {

constexpr const char *kServiceName = "vn.sparc.esca.Vision1";
constexpr const char *kObjectPath = "/vn/sparc/esca/Vision1";
constexpr const char *kInterface = "vn.sparc.esca.Vision1";

/// One object the detector found, in normalised frame coordinates: 0 is the
/// left/top edge and 1 the right/bottom one, whatever the capture resolution.
/// A box may stick slightly outside that range -- SSD extrapolates -- so a
/// drawing client clamps rather than trusting the bounds.
struct Detection {
    std::string label;      ///< human-readable class name, "?" if unmapped
    uint16_t class_id = 0;  ///< index as the model emits it, before label lookup
    double score = 0.0;
    double left = 0.0;
    double top = 0.0;
    double right = 0.0;
    double bottom = 0.0;
};

/// Everything found in one frame, and which frame that was.
struct DetectionSet {
    uint64_t frame_seq = 0;     ///< camera frame these came from; 0 means "nothing yet"
    uint64_t timestamp_ns = 0;  ///< when that frame was captured
    double inference_ms = 0.0;  ///< how long the model took on it
    std::vector<Detection> objects;
};

/// Subscribes to the vision service and keeps the newest result.
///
/// The proxy runs its own bus thread, so a client with its own event loop --
/// a Wayland application, say -- never has to pump D-Bus. It just calls
/// latest() whenever it is about to draw.
class VisionClient {
public:
    VisionClient();
    ~VisionClient();

    VisionClient(const VisionClient &) = delete;
    VisionClient &operator=(const VisionClient &) = delete;

    /// Connect to the service and begin receiving Detections signals.
    /// Returns false if the service is not on the bus.
    bool connect(std::string &error);

    /// Ask the service to run the detector. Reference counted on the service
    /// side: one client stopping does not blind the others.
    bool start(std::string &error);
    void stop();

    bool set_threshold(double value, std::string &error);

    /// Run the model on every Nth captured frame. Capture is 30 fps and
    /// inference is not; 3 is the default and keeps one core free.
    bool set_every(uint32_t n, std::string &error);

    /// The most recent result, as a copy -- the bus thread may replace the
    /// stored one at any moment. frame_seq is 0 until the first arrives.
    DetectionSet latest() const;

    bool state(bool &running, std::string &model, double &avg_ms, std::string &error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace esca::vision
