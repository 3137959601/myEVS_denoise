#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class EbfNative : public NativeBase {
public:
    EbfNative(int width, int height, uint64_t tau_ticks, int radius, double threshold, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), tau_(tau_ticks), radius_(std::clamp(radius, 0, 8)), threshold_(threshold) {
        reset();
    }

    void reset() override {
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        ts_.assign(n, 0);
        pol_.assign(n, 0);
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
    uint64_t tau_;
    int radius_;
    double threshold_;
    std::vector<uint64_t> ts_;
    std::vector<int8_t> pol_;

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) return false;
        const size_t idx0 = idx(x, y);
        if (radius_ <= 0 || tau_ == 0) {
            ts_[idx0] = t;
            pol_[idx0] = p;
            return true;
        }
        const double inv_tau = 1.0 / static_cast<double>(tau_);
        double score = 0.0;
        const int x0 = std::max(0, x - radius_);
        const int x1 = std::min(width_ - 1, x + radius_);
        const int y0 = std::max(0, y - radius_);
        const int y1 = std::min(height_ - 1, y + radius_);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) continue;
                const size_t k = idx(xx, yy);
                if (pol_[k] != p || ts_[k] == 0) continue;
                const uint64_t dt = abs_dt(t, ts_[k]);
                if (dt > tau_) continue;
                score += static_cast<double>(tau_ - dt) * inv_tau;
            }
        }
        ts_[idx0] = t;
        pol_[idx0] = p;
        return score > threshold_;
    }
};

} // namespace myevs_native_emlb
