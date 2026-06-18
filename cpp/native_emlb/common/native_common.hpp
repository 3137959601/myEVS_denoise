#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

namespace myevs_native_emlb {

struct Event3 {
    int32_t x = 0;
    int32_t y = 0;
    int8_t p = 0;
    uint64_t t = 0;
    bool valid = false;
};

inline int8_t norm_pol(int8_t p) noexcept {
    return p > 0 ? int8_t{1} : int8_t{-1};
}

inline uint64_t abs_dt(uint64_t a, uint64_t b) noexcept {
    return a >= b ? (a - b) : (b - a);
}

inline void check_dims(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("width/height must be positive");
    }
}

class NativeBase {
public:
    NativeBase(int width, int height, bool show_on, bool show_off)
        : width_(width), height_(height), show_on_(show_on), show_off_(show_off) {
        check_dims(width, height);
    }

    virtual ~NativeBase() = default;
    virtual void reset() = 0;

protected:
    int width_;
    int height_;
    bool show_on_;
    bool show_off_;

    bool visible(int32_t x, int32_t y, int8_t p) const noexcept {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) {
            return false;
        }
        if (p > 0 && !show_on_) {
            return false;
        }
        if (p < 0 && !show_off_) {
            return false;
        }
        return true;
    }
};

} // namespace myevs_native_emlb
