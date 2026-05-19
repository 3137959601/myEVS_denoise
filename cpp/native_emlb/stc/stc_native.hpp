#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class StcNative : public NativeBase {
public:
    StcNative(int width, int height, uint64_t duration_ticks, int radius, int threshold, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), duration_(duration_ticks), radius_(std::clamp(radius, 0, 8)), threshold_(std::max(0, threshold)) {
        reset();
    }

    void reset() override {
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        last_on_.assign(n, 0);
        last_off_.assign(n, 0);
    }

    py::array_t<uint8_t> accept_batch(
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> t,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> x,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> y,
        py::array_t<int8_t, py::array::c_style | py::array::forcecast> p) {
        auto tb = t.unchecked<1>();
        auto xb = x.unchecked<1>();
        auto yb = y.unchecked<1>();
        auto pb = p.unchecked<1>();
        const auto n = tb.shape(0);
        if (xb.shape(0) != n || yb.shape(0) != n || pb.shape(0) != n) {
            throw std::invalid_argument("t/x/y/p arrays must have the same length");
        }
        py::array_t<uint8_t> out(n);
        auto ob = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
            ob(i) = accept_one(xb(i), yb(i), norm_pol(pb(i)), tb(i)) ? uint8_t{1} : uint8_t{0};
        }
        return out;
    }

private:
    uint64_t duration_;
    int radius_;
    int threshold_;
    std::vector<uint64_t> last_on_;
    std::vector<uint64_t> last_off_;

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) return false;
        auto &last = p > 0 ? last_on_ : last_off_;
        const size_t idx0 = idx(x, y);
        if (threshold_ <= 0) {
            last[idx0] = t;
            return true;
        }
        if (duration_ == 0) {
            last[idx0] = t;
            return false;
        }
        const uint64_t t0 = t > duration_ ? (t - duration_) : 0;
        int cnt = 0;
        const int x0 = std::max(0, x - radius_);
        const int x1 = std::min(width_ - 1, x + radius_);
        const int y0 = std::max(0, y - radius_);
        const int y1 = std::min(height_ - 1, y + radius_);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) continue;
                const uint64_t ts = last[idx(xx, yy)];
                if (ts != 0 && ts >= t0) {
                    ++cnt;
                    if (cnt >= threshold_) {
                        last[idx0] = t;
                        return true;
                    }
                }
            }
        }
        last[idx0] = t;
        return false;
    }
};

} // namespace myevs_native_emlb
