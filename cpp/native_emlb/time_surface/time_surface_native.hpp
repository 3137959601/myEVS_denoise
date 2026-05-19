#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class TimeSurfaceNative : public NativeBase {
public:
    TimeSurfaceNative(int width, int height, uint64_t decay_ticks, int radius, double threshold, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), decay_(std::max<uint64_t>(1, decay_ticks)), radius_(std::max(0, radius)), threshold_(threshold) {
        reset();
    }

    void reset() override {
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        pos_.assign(n, 0);
        neg_.assign(n, 0);
        pos_valid_.assign(n, 0);
        neg_valid_.assign(n, 0);
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
    uint64_t decay_;
    int radius_;
    double threshold_;
    std::vector<uint64_t> pos_;
    std::vector<uint64_t> neg_;
    std::vector<uint8_t> pos_valid_;
    std::vector<uint8_t> neg_valid_;

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) {
            return false;
        }
        auto &surf = p > 0 ? pos_ : neg_;
        auto &valid = p > 0 ? pos_valid_ : neg_valid_;
        int support = 0;
        double sum = 0.0;
        const int x0 = std::max(0, x - radius_);
        const int x1 = std::min(width_ - 1, x + radius_);
        const int y0 = std::max(0, y - radius_);
        const int y1 = std::min(height_ - 1, y + radius_);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                const size_t k = idx(xx, yy);
                if (!valid[k]) continue;
                const double dt = static_cast<double>(abs_dt(t, surf[k]));
                sum += std::exp(-dt / static_cast<double>(decay_));
                ++support;
            }
        }
        const double score = support == 0 ? 0.0 : sum / static_cast<double>(support);
        const size_t k0 = idx(x, y);
        surf[k0] = t;
        valid[k0] = 1;
        return score >= threshold_;
    }
};

} // namespace myevs_native_emlb
