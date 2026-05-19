#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class EventFlowNative : public NativeBase {
public:
    EventFlowNative(int width, int height, uint64_t duration_ticks, int radius, double threshold, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), duration_(duration_ticks), radius_(std::max(1, radius)), threshold_(threshold) {
        reset();
    }

    void reset() override {
        events_.clear();
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
    double threshold_;
    std::deque<Event3> events_;

    static bool solve3(double a[3][3], double b[3], double x[3]) noexcept {
        double m[3][4] = {
            {a[0][0], a[0][1], a[0][2], b[0]},
            {a[1][0], a[1][1], a[1][2], b[1]},
            {a[2][0], a[2][1], a[2][2], b[2]},
        };
        for (int col = 0; col < 3; ++col) {
            int piv = col;
            double best = std::fabs(m[col][col]);
            for (int r = col + 1; r < 3; ++r) {
                const double v = std::fabs(m[r][col]);
                if (v > best) {
                    best = v;
                    piv = r;
                }
            }
            if (best < 1e-12) return false;
            if (piv != col) {
                for (int c = col; c < 4; ++c) std::swap(m[col][c], m[piv][c]);
            }
            const double div = m[col][col];
            for (int c = col; c < 4; ++c) m[col][c] /= div;
            for (int r = 0; r < 3; ++r) {
                if (r == col) continue;
                const double f = m[r][col];
                for (int c = col; c < 4; ++c) m[r][c] -= f * m[col][c];
            }
        }
        x[0] = m[0][3];
        x[1] = m[1][3];
        x[2] = m[2][3];
        return true;
    }

    double fit_flow(int32_t x, int32_t y, uint64_t t) const {
        size_t count = 0;
        double ata[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        double atb[3] = {0.0, 0.0, 0.0};
        for (const auto &e : events_) {
            if (std::abs(x - e.x) > radius_ || std::abs(y - e.y) > radius_) continue;
            const double row[3] = {static_cast<double>(e.x), static_cast<double>(e.y), 1.0};
            const double bv = (static_cast<double>(static_cast<int64_t>(e.t) - static_cast<int64_t>(t))) * 1.0e-3;
            for (int r = 0; r < 3; ++r) {
                atb[r] += row[r] * bv;
                for (int c = 0; c < 3; ++c) {
                    ata[r][c] += row[r] * row[c];
                }
            }
            ++count;
        }
        if (count <= 3) {
            return std::numeric_limits<double>::infinity();
        }
        double sol[3] = {0.0, 0.0, 0.0};
        if (!solve3(ata, atb, sol)) {
            return std::numeric_limits<double>::infinity();
        }
        if (std::fabs(sol[0]) < 1e-12 || std::fabs(sol[1]) < 1e-12) {
            return std::numeric_limits<double>::infinity();
        }
        const double invx = -1.0 / sol[0];
        const double invy = -1.0 / sol[1];
        return std::sqrt(invx * invx + invy * invy);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) {
            return false;
        }
        const double flow = fit_flow(x, y, t);
        const bool keep = flow <= threshold_;
        while (!events_.empty()) {
            if (abs_dt(t, events_.front().t) >= duration_) {
                events_.pop_front();
            } else {
                break;
            }
        }
        events_.push_back(Event3{x, y, p, t, true});
        return keep;
    }
};

} // namespace myevs_native_emlb
