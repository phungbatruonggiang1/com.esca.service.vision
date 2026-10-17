// Standalone detection benchmark.
//
// This does not talk to esca-visiond: it opens the camera itself and runs the
// same Detector in-process. That is the point -- it measures the model and
// the board without any of the service machinery in the way, which is the
// number worth quoting. Use esca-visionctl to exercise the service.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "detector.hpp"
#include "esca/camera/client.hpp"

using namespace esca::camera;
using esca::vision::Detection;
using esca::vision::Detector;

namespace {

struct Options {
    Detector::Config detector;
    int frames = 50;  ///< how many inferences to run; 0 means forever
    int every = 3;    ///< run the model on every Nth captured frame
};

int usage()
{
    std::fprintf(stderr,
                 "usage: esca-vision-detect [options]\n"
                 "  --model PATH      .tflite detector\n"
                 "  --labels PATH     label list, one per line\n"
                 "  --threshold F     minimum score, default 0.5\n"
                 "  --frames N        inferences to run, 0 = forever (default 50)\n"
                 "  --every N         run on every Nth frame (default 3)\n"
                 "  --threads N       interpreter threads (default 3)\n"
                 "  --no-xnnpack      disable the XNNPACK delegate\n");
    return 2;
}

} // namespace

int main(int argc, char **argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };

        if (arg == "--model") opt.detector.model = next();
        else if (arg == "--labels") opt.detector.labels = next();
        else if (arg == "--threshold") opt.detector.threshold = std::stof(next());
        else if (arg == "--threads") opt.detector.threads = std::stoi(next());
        else if (arg == "--no-xnnpack") opt.detector.xnnpack = false;
        else if (arg == "--frames") opt.frames = std::stoi(next());
        else if (arg == "--every") opt.every = std::max(1, std::stoi(next()));
        else return usage();
    }

    std::string error;
    Detector detector;
    if (!detector.load(opt.detector, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("model %s, input %dx%d, %d threads, XNNPACK %s\n", opt.detector.model.c_str(),
                detector.input_width(), detector.input_height(), opt.detector.threads,
                detector.accelerated() ? "on" : "off");

    CameraClient camera;
    const auto ids = camera.device_list();
    if (ids.empty()) {
        std::fprintf(stderr, "no cameras (is esca-camerad running?)\n");
        return 1;
    }
    if (camera.open(ids.front(), error) < 0) {
        std::fprintf(stderr, "open: %s\n", error.c_str());
        return 1;
    }
    if (!camera.start(error)) {
        std::fprintf(stderr, "start: %s\n", error.c_str());
        return 1;
    }

    const FrameFormat format = camera.frame_format();
    std::printf("camera %s, %ux%u stride %u\n\n", ids.front().c_str(), format.width,
                format.height, format.stride);

    std::vector<uint8_t> frame;
    std::vector<Detection> objects;
    FrameInfo info;
    double total_ms = 0;
    int inferences = 0;
    int skipped = 0;

    while (opt.frames == 0 || inferences < opt.frames) {
        if (!camera.next_frame(frame, info)) {
            std::fprintf(stderr, "timed out waiting for a frame\n");
            break;
        }
        // Dropping frames is the point of --every: inference is slower than
        // capture, and the newest frame is the only one worth looking at.
        if (++skipped < opt.every)
            continue;
        skipped = 0;

        double ms = 0.0;
        if (!detector.run(frame.data(), format.width, format.height, format.stride, objects, ms,
                          error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            break;
        }
        total_ms += ms;
        ++inferences;

        std::printf("seq %-7llu %6.1f ms  %zu object(s)\n",
                    static_cast<unsigned long long>(info.sequence), ms, objects.size());
        for (const Detection &d : objects) {
            std::printf("    %-18s %.2f   box [%.2f %.2f %.2f %.2f]\n", d.label.c_str(), d.score,
                        d.left, d.top, d.right, d.bottom);
        }
    }

    if (inferences > 0) {
        std::printf("\n%d inferences, %.1f ms average (%.1f fps of inference alone)\n",
                    inferences, total_ms / inferences, 1000.0 * inferences / total_ms);
    }

    camera.stop();
    camera.close();
    return 0;
}
