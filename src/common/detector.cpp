#include "detector.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#include "preprocess.hpp"

namespace esca::vision {

namespace {

std::vector<std::string> read_labels(const std::string &path)
{
    std::vector<std::string> labels;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        labels.push_back(line);
    }
    return labels;
}

} // namespace

struct Detector::Impl {
    Detector::Config config;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate *delegate = nullptr;

    std::vector<std::string> labels;
    int label_offset = 0;
    int input_w = 0;
    int input_h = 0;
    std::vector<uint8_t> rgb;

    // SetThreshold arrives on the bus thread while the worker is mid-frame.
    std::atomic<float> threshold{ 0.5f };

    ~Impl()
    {
        // The interpreter refers to the delegate, so it has to go first.
        interpreter.reset();
        if (delegate)
            TfLiteXNNPackDelegateDelete(delegate);
    }
};

Detector::Detector() : impl_(std::make_unique<Impl>()) {}

Detector::~Detector() = default;

bool Detector::load(const Config &config, std::string &error)
{
    impl_->config = config;
    impl_->threshold = config.threshold;

    impl_->model = tflite::FlatBufferModel::BuildFromFile(config.model.c_str());
    if (!impl_->model) {
        error = "cannot load model " + config.model;
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (tflite::InterpreterBuilder(*impl_->model, resolver)(&impl_->interpreter) != kTfLiteOk
        || !impl_->interpreter) {
        error = "cannot build interpreter";
        return false;
    }
    impl_->interpreter->SetNumThreads(config.threads);

    if (config.xnnpack) {
        TfLiteXNNPackDelegateOptions xnn = TfLiteXNNPackDelegateOptionsDefault();
        xnn.num_threads = config.threads;
        impl_->delegate = TfLiteXNNPackDelegateCreate(&xnn);
        // Not every model is fully supported; falling back to the reference
        // kernels is slower but correct, so a refusal is not fatal.
        if (!impl_->delegate
            || impl_->interpreter->ModifyGraphWithDelegate(impl_->delegate) != kTfLiteOk)
            impl_->delegate = nullptr;
    }

    if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
        error = "cannot allocate tensors";
        return false;
    }

    const TfLiteTensor *input = impl_->interpreter->tensor(impl_->interpreter->inputs()[0]);
    if (input->dims->size != 4 || input->type != kTfLiteUInt8) {
        error = "expected a quantised uint8 model with a 4-D input";
        return false;
    }
    impl_->input_h = input->dims->data[1];
    impl_->input_w = input->dims->data[2];
    impl_->rgb.resize(static_cast<size_t>(impl_->input_w) * impl_->input_h * 3);

    impl_->labels = read_labels(config.labels);
    // The COCO labelmap shipped with this model holds 91 lines for 90 classes:
    // line 1 is a "???" placeholder for the background class, which the SSD
    // outputs never name. Class 0 from the model is therefore line 2. Without
    // this offset every detection is reported as the label above the right
    // one, which looks plausible and is always wrong.
    impl_->label_offset = (!impl_->labels.empty() && impl_->labels[0] == "???") ? 1 : 0;
    return true;
}

bool Detector::run(const uint8_t *yuv, uint32_t width, uint32_t height, uint32_t stride,
                   std::vector<Detection> &out, double &inference_ms, std::string &error)
{
    out.clear();
    if (!impl_->interpreter) {
        error = "detector not loaded";
        return false;
    }

    yuv420_to_rgb(yuv, static_cast<int>(width), static_cast<int>(height),
                  static_cast<int>(stride), impl_->rgb.data(), impl_->input_w, impl_->input_h);

    TfLiteTensor *input = impl_->interpreter->tensor(impl_->interpreter->inputs()[0]);
    std::memcpy(input->data.uint8, impl_->rgb.data(), impl_->rgb.size());

    const auto started = std::chrono::steady_clock::now();
    if (impl_->interpreter->Invoke() != kTfLiteOk) {
        error = "inference failed";
        return false;
    }
    inference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();

    // Standard COCO SSD output order: boxes, classes, scores, count.
    const auto &outputs = impl_->interpreter->outputs();
    if (outputs.size() < 4) {
        error = "expected 4 output tensors";
        return false;
    }
    const float *boxes = impl_->interpreter->tensor(outputs[0])->data.f;
    const float *classes = impl_->interpreter->tensor(outputs[1])->data.f;
    const float *scores = impl_->interpreter->tensor(outputs[2])->data.f;
    const int count = static_cast<int>(*impl_->interpreter->tensor(outputs[3])->data.f);

    const float threshold = impl_->threshold.load();
    for (int i = 0; i < count; ++i) {
        if (scores[i] < threshold)
            continue;

        const int class_id = static_cast<int>(classes[i]);
        const int label_index = class_id + impl_->label_offset;

        Detection d;
        d.label = (label_index >= 0 && label_index < static_cast<int>(impl_->labels.size()))
                      ? impl_->labels[label_index]
                      : "?";
        d.class_id = static_cast<uint16_t>(class_id);
        d.score = scores[i];
        // SSD writes each box as top, left, bottom, right.
        d.top = boxes[i * 4 + 0];
        d.left = boxes[i * 4 + 1];
        d.bottom = boxes[i * 4 + 2];
        d.right = boxes[i * 4 + 3];
        out.push_back(std::move(d));
    }
    return true;
}

void Detector::set_threshold(float value) { impl_->threshold = value; }
float Detector::threshold() const { return impl_->threshold.load(); }
int Detector::input_width() const { return impl_->input_w; }
int Detector::input_height() const { return impl_->input_h; }
bool Detector::accelerated() const { return impl_->delegate != nullptr; }
const std::string &Detector::model_path() const { return impl_->config.model; }

} // namespace esca::vision
