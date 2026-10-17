// Turning a camera frame into what the model wants.
//
// The camera service publishes YU12 (planar YUV420) at the capture
// resolution; a detector wants small square RGB. Doing that with nearest
// neighbour sampling costs a few milliseconds and no dependencies, which
// matters more here than resampling quality: the model downsamples anyway.

#pragma once

#include <cstdint>
#include <vector>

namespace esca::vision {

/// Converts a planar YUV420 (YU12/I420) frame to tightly packed RGB888,
/// resizing to dst_w x dst_h at the same time.
///
/// @param data    frame as published by the camera service
/// @param width   luma width in pixels
/// @param height  luma height in pixels
/// @param stride  bytes per luma row; chroma rows are stride/2
inline void yuv420_to_rgb(const uint8_t *data, int width, int height, int stride,
                          uint8_t *dst, int dst_w, int dst_h)
{
    const uint8_t *y_plane = data;
    const int chroma_stride = stride / 2;
    const int chroma_height = height / 2;
    const uint8_t *u_plane = y_plane + static_cast<size_t>(stride) * height;
    const uint8_t *v_plane = u_plane + static_cast<size_t>(chroma_stride) * chroma_height;

    for (int dy = 0; dy < dst_h; ++dy) {
        // Nearest neighbour: map the destination pixel back to a source one.
        const int sy = dy * height / dst_h;
        for (int dx = 0; dx < dst_w; ++dx) {
            const int sx = dx * width / dst_w;

            const int y = y_plane[static_cast<size_t>(sy) * stride + sx];
            const int u = u_plane[static_cast<size_t>(sy / 2) * chroma_stride + sx / 2] - 128;
            const int v = v_plane[static_cast<size_t>(sy / 2) * chroma_stride + sx / 2] - 128;

            // BT.601, the conversion libcamera's YUV420 output follows.
            int r = y + ((91881 * v) >> 16);
            int g = y - ((22554 * u + 46802 * v) >> 16);
            int b = y + ((116130 * u) >> 16);

            r = r < 0 ? 0 : (r > 255 ? 255 : r);
            g = g < 0 ? 0 : (g > 255 ? 255 : g);
            b = b < 0 ? 0 : (b > 255 ? 255 : b);

            uint8_t *pixel = dst + (static_cast<size_t>(dy) * dst_w + dx) * 3;
            pixel[0] = static_cast<uint8_t>(r);
            pixel[1] = static_cast<uint8_t>(g);
            pixel[2] = static_cast<uint8_t>(b);
        }
    }
}

/// Bytes a YUV420 frame of this geometry occupies, for sanity checks.
inline size_t yuv420_size(int height, int stride)
{
    return static_cast<size_t>(stride) * height + static_cast<size_t>(stride / 2) * (height / 2) * 2;
}

} // namespace esca::vision
