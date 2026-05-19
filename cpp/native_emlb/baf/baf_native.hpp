#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class BafNative : public NativeBase {
public:
    BafNative(int width, int height, uint64_t duration_ticks, int radius, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), duration_(duration_ticks), radius_(std::clamp(radius, 0, 8)) {
        reset();
    }

    void reset() override {
        last_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), 0);
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
    std::vector<uint64_t> last_;

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) return false;
        const size_t idx0 = idx(x, y);
        if (radius_ <= 0 || duration_ == 0) {
            last_[idx0] = t;
            return true;
        }
        const uint64_t t0 = t > duration_ ? (t - duration_) : 0;
        bool has_neighbor = false;
        const int x0 = std::max(0, x - radius_);
        const int x1 = std::min(width_ - 1, x + radius_);
        const int y0 = std::max(0, y - radius_);
        const int y1 = std::min(height_ - 1, y + radius_);
        for (int yy = y0; yy <= y1 && !has_neighbor; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) continue;
                const uint64_t ts = last_[idx(xx, yy)];
                if (ts != 0 && ts >= t0) {
                    has_neighbor = true;
                    break;
                }
            }
        }
        last_[idx0] = t;
        return has_neighbor;
    }
};

} // namespace myevs_native_emlb
