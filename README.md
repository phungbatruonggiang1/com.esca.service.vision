# ESCA Vision — what the camera is looking at

A second client of the ESCA camera service. It reads frames from the same
shared memory the viewer draws from, runs an SSD MobileNet detector over every
Nth one, and publishes the result as a D-Bus signal.

    camera (IMX219) → esca-camerad → shared memory ─┬→ esca-visiond ──┐
                                                     │                │ Detections
                                                     └→ esca-camview ←┘

Pixels go through shared memory, descriptions go over the bus. A frame is
1.3 MB and arrives 30 times a second; a list of detections is a few hundred
bytes, ten times a second. Each travels by the route that suits it.

Inference lives in its own process on purpose. It is the slowest and least
predictable part of the system, and when it stalls or dies the picture should
keep moving without it.

## The interface

    service     vn.sparc.esca.Vision1
    object      /vn/sparc/esca/Vision1
    interface   vn.sparc.esca.Vision1

    signal  Detections(t frame_seq, t timestamp_ns, d inference_ms,
                       a(sqddddd) objects)
            each object: label, class_id, score, left, top, right, bottom

    method  Start() → i          reference counted; the detector runs while > 0
            Stop() → i
            SetThreshold(d) → i
            SetEvery(u) → i      run the model on every Nth captured frame
            GetState() → (b running, s model, d avg_ms, u every)
            GetErrorMessage() → s

    signal  FaultEvent(s where, s what)

### frame_seq is the whole point

The service and its clients read the same ring, so they see the same frame
numbering. That one number is what lets a viewer know which frame a box
belongs to, with no clocks to synchronise and nothing to guess.

It matters because the answer always arrives late. Capture is 33 ms apart and
inference takes about 70 ms on this board, so detections reach a viewer two
frames or so after the frame they describe. A viewer can draw the newest boxes
anyway and accept that they lag, hold frames back to match, or buffer and
join on the sequence number. `esca-camview` takes the first option and stops
drawing entirely once the boxes fall more than fifteen frames behind, because
stale boxes are worse than none.

## What it ships

- `esca-visiond` — the service.
- `esca-visionctl` — the service from a terminal: `state`, `watch [seconds]`,
  `threshold VALUE`, `every N`. Run this first when boxes look wrong on
  screen; it tells you whether the service or the drawing is at fault.
- `esca-vision-detect` — a standalone benchmark that opens the camera itself
  and never touches the service, so it measures the model and the board with
  nothing in the way.
- `libesca-visionclient` and `esca/vision/client.hpp` — what an application
  links against. The proxy runs its own bus thread, so an application with a
  Wayland loop never pumps D-Bus; it calls `latest()` before it draws.

## Measured on a Raspberry Pi 4

SSD MobileNet v1 300×300, uint8 quantised, XNNPACK on:

| threads | ms per inference |
|---|---|
| 4 (benchmark, nothing else running) | 54 |
| 3 (the service default, viewer drawing) | 66–72 |

Three threads by default, not four: the fourth core is what keeps the viewer
drawing while the model runs.

## Getting a model

Not shipped — it has its own licence and is 4 MB:

    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
    unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d /usr/share/esca/models/

That gives `detect.tflite` and `labelmap.txt`, and covers the 80 COCO classes:
person, chair, cup, laptop, cell phone, bottle, and so on.

The labelmap has 91 lines for 90 classes — line 1 is a `???` placeholder for
the background class the SSD never emits — so the detector adds one when
looking a class up. Without that every detection is reported as the label
above the right one, which looks plausible and is always wrong.

Any uint8-quantised detector with the four standard SSD outputs works;
`--model` points at it.

## Building

In the Yocto build it is the `esca-vision` recipe in `meta-sparc-esca`. It
needs `libtensorflow-lite`, `sdbus-c++` and the camera service's client
library, all of which that layer provides.
