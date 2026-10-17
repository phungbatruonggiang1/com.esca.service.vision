// Exercises esca-visiond over D-Bus, without a display.
//
// Everything the viewer application does, this does from a terminal: start
// the detector, watch the Detections signals arrive, print the frame sequence
// each one belongs to. If the boxes look wrong on screen, run this first --
// it tells you whether the service or the drawing is at fault.

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "esca/vision/client.hpp"

using namespace esca::vision;

namespace {

int usage()
{
    std::fprintf(stderr,
                 "usage: esca-visionctl state\n"
                 "       esca-visionctl watch [seconds]\n"
                 "       esca-visionctl threshold VALUE\n"
                 "       esca-visionctl every N\n");
    return 2;
}

int cmd_state(VisionClient &client)
{
    bool running = false;
    std::string model;
    double avg_ms = 0.0;
    std::string error;
    if (!client.state(running, model, avg_ms, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("running: %s\nmodel:   %s\naverage: %.1f ms\n", running ? "yes" : "no",
                model.c_str(), avg_ms);
    return 0;
}

int cmd_watch(VisionClient &client, int seconds)
{
    std::string error;
    if (!client.start(error)) {
        std::fprintf(stderr, "start: %s\n", error.c_str());
        return 1;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    uint64_t last_seq = 0;

    while (std::chrono::steady_clock::now() < deadline) {
        const DetectionSet set = client.latest();
        if (set.frame_seq != last_seq) {
            last_seq = set.frame_seq;
            std::printf("seq %-7llu %6.1f ms  %zu object(s)\n",
                        static_cast<unsigned long long>(set.frame_seq), set.inference_ms,
                        set.objects.size());
            for (const Detection &d : set.objects) {
                std::printf("    %-18s %.2f   box [%.2f %.2f %.2f %.2f]\n", d.label.c_str(),
                            d.score, d.left, d.top, d.right, d.bottom);
            }
            std::fflush(stdout);
        }
        // Polling a snapshot rather than taking a callback, because that is
        // exactly what the viewer does before each frame it draws.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    client.stop();
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
        return usage();

    VisionClient client;
    std::string error;
    if (!client.connect(error)) {
        std::fprintf(stderr, "cannot reach %s: %s\n", kServiceName, error.c_str());
        return 1;
    }

    const std::string command = argv[1];

    if (command == "state")
        return cmd_state(client);

    if (command == "watch")
        return cmd_watch(client, argc > 2 ? std::stoi(argv[2]) : 10);

    if (command == "threshold" && argc > 2) {
        if (!client.set_threshold(std::stod(argv[2]), error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        return 0;
    }

    if (command == "every" && argc > 2) {
        if (!client.set_every(static_cast<uint32_t>(std::stoul(argv[2])), error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        return 0;
    }

    return usage();
}
