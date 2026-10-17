#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <sdbus-c++/sdbus-c++.h>

#include "esca/vision/client.hpp" // kServiceName
#include "service.hpp"

namespace {

sdbus::IConnection *g_connection = nullptr;

void handle_signal(int)
{
    // Leaving the event loop unwinds the service, which stops the worker and
    // releases our handle on the camera. Dying with it held would keep the
    // camera service streaming for a client that no longer exists.
    if (g_connection)
        g_connection->leaveEventLoop();
}

int usage()
{
    std::fprintf(stderr,
                 "usage: esca-visiond [options]\n"
                 "  --model PATH      .tflite detector\n"
                 "  --labels PATH     label list, one per line\n"
                 "  --threshold F     minimum score, default 0.5\n"
                 "  --every N         run the model on every Nth frame (default 3)\n"
                 "  --threads N       interpreter threads (default 3 of 4 cores)\n"
                 "  --no-xnnpack      disable the XNNPACK delegate\n"
                 "  --autostart       detect from startup instead of waiting for Start()\n");
    return 2;
}

} // namespace

int main(int argc, char **argv)
{
    using namespace esca::vision;

    VisionService::Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };

        if (arg == "--model") config.detector.model = next();
        else if (arg == "--labels") config.detector.labels = next();
        else if (arg == "--threshold") config.detector.threshold = std::stof(next());
        else if (arg == "--threads") config.detector.threads = std::stoi(next());
        else if (arg == "--no-xnnpack") config.detector.xnnpack = false;
        else if (arg == "--every") config.every = static_cast<uint32_t>(std::stoul(next()));
        else if (arg == "--autostart") config.autostart = true;
        else return usage();
    }

    try {
        auto connection = sdbus::createSystemBusConnection(sdbus::ServiceName{ kServiceName });
        g_connection = connection.get();

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        VisionService service(*connection, config);

        std::string error;
        if (!service.load(error)) {
            // Fatal, unlike a missing camera: without a model there is
            // nothing this service can ever do, and failing at startup is a
            // clearer signal than answering every Start() with an error.
            std::fprintf(stderr, "esca-visiond: %s\n", error.c_str());
            return 1;
        }
        service.autostart();

        std::fprintf(stderr, "esca-visiond: ready on %s\n", kServiceName);
        connection->enterEventLoop();
        std::fprintf(stderr, "esca-visiond: shutting down\n");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "esca-visiond: fatal: %s\n", e.what());
        return 1;
    }
}
