#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class KNoiseNative : public NativeBase {
public:
    KNoiseNative(int width, int height, uint64_t duration_ticks, int threshold, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), duration_(duration_ticks), threshold_(std::max(0, threshold)) {
        reset();
    }

    void reset() override {
        x_cols_.assign(static_cast<size_t>(width_), Event3{});
        y_rows_.assign(static_cast<size_t>(height_), Event3{});
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
    int threshold_;
    std::vector<Event3> x_cols_;
    std::vector<Event3> y_rows_;

    bool close_same(const Event3 &e, uint64_t t, int8_t p) const noexcept {
        return e.valid && e.p == p && abs_dt(t, e.t) <= duration_;
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) {
            return false;
        }
        int support = 0;
        const bool xm = x > 0;
        const bool xp = x < width_ - 1;
        const bool ym = y > 0;
        const bool yp = y < height_ - 1;

        if (xm) {
            const auto &e = x_cols_[static_cast<size_t>(x - 1)];
            if (close_same(e, t, p) && ((ym && e.y == y - 1) || e.y == y || (yp && e.y == y + 1))) ++support;
        }
        {
            const auto &e = x_cols_[static_cast<size_t>(x)];
            if (close_same(e, t, p) && ((ym && e.y == y - 1) || (yp && e.y == y + 1))) ++support;
        }
        if (xp) {
            const auto &e = x_cols_[static_cast<size_t>(x + 1)];
            if (close_same(e, t, p) && ((ym && e.y == y - 1) || e.y == y || (yp && e.y == y + 1))) ++support;
        }
        if (ym) {
            const auto &e = y_rows_[static_cast<size_t>(y - 1)];
            if (close_same(e, t, p) && ((xm && e.x == x - 1) || e.x == x || (xp && e.x == x + 1))) ++support;
        }
        {
            const auto &e = y_rows_[static_cast<size_t>(y)];
            if (close_same(e, t, p) && ((xm && e.x == x - 1) || (xp && e.x == x + 1))) ++support;
        }
        if (yp) {
            const auto &e = y_rows_[static_cast<size_t>(y + 1)];
            if (close_same(e, t, p) && ((xm && e.x == x - 1) || e.x == x || (xp && e.x == x + 1))) ++support;
        }

        Event3 e{x, y, p, t, true};
        x_cols_[static_cast<size_t>(x)] = e;
        y_rows_[static_cast<size_t>(y)] = e;
        return support >= threshold_;
    }
};

} // namespace myevs_native_emlb
