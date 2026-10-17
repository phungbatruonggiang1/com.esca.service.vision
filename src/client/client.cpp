#include "esca/vision/client.hpp"

#include <chrono>
#include <mutex>

#include <sdbus-c++/sdbus-c++.h>

namespace esca::vision {

namespace {
/// How the objects array travels: label, class, score, then the box.
using WireObject = sdbus::Struct<std::string, uint16_t, double, double, double, double, double>;
} // namespace

struct VisionClient::Impl {
    std::unique_ptr<sdbus::IProxy> proxy;

    mutable std::mutex mutex;
    DetectionSet latest;  ///< written by the bus thread, read by the caller's

    std::string service_error()
    {
        std::string message;
        try {
            proxy->callMethod("GetErrorMessage").onInterface(kInterface).storeResultsTo(message);
        } catch (const sdbus::Error &e) {
            message = e.getMessage();
        }
        return message;
    }
};

VisionClient::VisionClient() : impl_(std::make_unique<Impl>()) {}

VisionClient::~VisionClient() = default;

bool VisionClient::connect(std::string &error)
{
    // Its own connection and its own event loop thread, so an application
    // with a Wayland loop never has to pump D-Bus -- and explicitly the
    // system bus. Letting sd-bus pick the default would land on the session
    // bus for anything the display manager starts, since those have
    // DBUS_SESSION_BUS_ADDRESS set, and the service is not there.
    impl_->proxy = sdbus::createProxy(sdbus::createSystemBusConnection(),
                                      sdbus::ServiceName{ kServiceName },
                                      sdbus::ObjectPath{ kObjectPath });

    try {
        impl_->proxy->uponSignal("Detections")
            .onInterface(kInterface)
            .call([this](const uint64_t &frame_seq, const uint64_t &timestamp_ns,
                         const double &inference_ms, const std::vector<WireObject> &objects) {
                DetectionSet set;
                set.frame_seq = frame_seq;
                set.timestamp_ns = timestamp_ns;
                set.inference_ms = inference_ms;
                set.objects.reserve(objects.size());
                for (const WireObject &o : objects) {
                    Detection d;
                    d.label = o.get<0>();
                    d.class_id = o.get<1>();
                    d.score = o.get<2>();
                    d.left = o.get<3>();
                    d.top = o.get<4>();
                    d.right = o.get<5>();
                    d.bottom = o.get<6>();
                    set.objects.push_back(std::move(d));
                }

                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->latest = std::move(set);
            });
    } catch (const sdbus::Error &e) {
        error = e.getMessage();
        return false;
    }

    // Does the service actually answer? Subscribing to a signal succeeds even
    // when nobody is there to send it, so ask something -- but briefly. The
    // D-Bus default is 25 seconds, and a viewer should fall back to plain
    // video in a moment rather than show nothing while a sick service is
    // timing out.
    sdbus::Struct<bool, std::string, double, uint32_t> probe;
    try {
        impl_->proxy->callMethod("GetState")
            .onInterface(kInterface)
            .withTimeout(std::chrono::seconds(2))
            .storeResultsTo(probe);
    } catch (const sdbus::Error &e) {
        error = e.getMessage();
        return false;
    }
    return true;
}

bool VisionClient::start(std::string &error)
{
    if (!impl_->proxy) {
        error = "not connected";
        return false;
    }

    int32_t rc = -1;
    try {
        impl_->proxy->callMethod("Start").onInterface(kInterface).storeResultsTo(rc);
    } catch (const sdbus::Error &e) {
        error = e.getMessage();
        return false;
    }
    if (rc != 0) {
        error = impl_->service_error();
        return false;
    }
    return true;
}

void VisionClient::stop()
{
    if (!impl_->proxy)
        return;
    try {
        int32_t ignored = 0;
        impl_->proxy->callMethod("Stop").onInterface(kInterface).storeResultsTo(ignored);
    } catch (const sdbus::Error &) {
    }
}

bool VisionClient::set_threshold(double value, std::string &error)
{
    if (!impl_->proxy) {
        error = "not connected";
        return false;
    }

    int32_t rc = -1;
    try {
        impl_->proxy->callMethod("SetThreshold")
            .onInterface(kInterface)
            .withArguments(value)
            .storeResultsTo(rc);
    } catch (const sdbus::Error &e) {
        error = e.getMessage();
        return false;
    }
    if (rc != 0) {
        error = impl_->service_error();
        return false;
    }
    return true;
}

bool VisionClient::set_every(uint32_t n, std::string &error)
{
    if (!impl_->proxy) {
        error = "not connected";
        return false;
    }

    int32_t rc = -1;
    try {
        impl_->proxy->callMethod("SetEvery")
            .onInterface(kInterface)
            .withArguments(n)
            .storeResultsTo(rc);
    } catch (const sdbus::Error &e) {
        error = e.getMessage();
        return false;
    }
    if (rc != 0) {
        error = impl_->service_error();
        return false;
    }
    return true;
}

DetectionSet VisionClient::latest() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->latest;
}

bool VisionClient::state(bool &running, std::string &model, double &avg_ms, std::string &error)
{
    if (!impl_->proxy) {
        error = "not connected";
        return false;
    }

    sdbus::Struct<bool, std::string, double, uint32_t> s;
    try {
        impl_->proxy->callMethod("GetState").onInterface(kInterface).storeResultsTo(s);
    } catch (const sdbus::Error &e) {
        error = e.getMessage();
        return false;
    }

    running = s.get<0>();
    model = s.get<1>();
    avg_ms = s.get<2>();
    return true;
}

} // namespace esca::vision
