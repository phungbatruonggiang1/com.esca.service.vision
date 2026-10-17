// The model, wrapped so that the daemon and the command-line tool run exactly
// the same inference.
//
// TensorFlow Lite stays behind the pimpl: tflite::Interpreter is an alias for
// a class in an inner namespace, so it cannot be forward declared, and
// including its headers here would pull all of TensorFlow into every
// translation unit that wants a detection.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "esca/vision/client.hpp" // Detection

namespace esca::vision {

class Detector {
public:
    struct Config {
        std::string model = "/usr/share/esca/models/detect.tflite";
        std::string labels = "/usr/share/esca/models/labelmap.txt";
        float threshold = 0.5f;
        int threads = 3;      ///< one less than the four cores, so drawing still gets one
        bool xnnpack = true;
    };

    Detector();
    ~Detector();

    Detector(const Detector &) = delete;
    Detector &operator=(const Detector &) = delete;

    bool load(const Config &config, std::string &error);

    /// Run the model over one planar YUV420 frame as the camera publishes it.
    /// Conversion and resizing to the model's input happen here.
    bool run(const uint8_t *yuv, uint32_t width, uint32_t height, uint32_t stride,
             std::vector<Detection> &out, double &inference_ms, std::string &error);

    void set_threshold(float value);
    float threshold() const;

    int input_width() const;
    int input_height() const;
    bool accelerated() const;
    const std::string &model_path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace esca::vision
