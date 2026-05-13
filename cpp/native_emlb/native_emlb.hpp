#pragma once

#include <algorithm>
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

class YNoiseNative : public NativeBase {
public:
    YNoiseNative(int width, int height, uint64_t duration_ticks, int radius, int threshold, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off), duration_(duration_ticks), radius_(std::max(0, radius)), threshold_(std::max(0, threshold)) {
        reset();
    }

    void reset() override {
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        ts_.assign(n, 0);
        pol_.assign(n, 0);
        valid_.assign(n, 0);
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
    std::vector<uint64_t> ts_;
    std::vector<int8_t> pol_;
    std::vector<uint8_t> valid_;

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) {
            return false;
        }
        int density = 0;
        const int x0 = std::max(0, x - radius_);
        const int x1 = std::min(width_ - 1, x + radius_);
        const int y0 = std::max(0, y - radius_);
        const int y1 = std::min(height_ - 1, y + radius_);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                const size_t k = idx(xx, yy);
                if (!valid_[k]) continue;
                if (abs_dt(t, ts_[k]) > duration_) continue;
                if (pol_[k] != p) continue;
                ++density;
            }
        }
        const size_t k0 = idx(x, y);
        ts_[k0] = t;
        pol_[k0] = p;
        valid_[k0] = 1;
        return density >= threshold_;
    }
};

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

class N149Native : public NativeBase {
public:
    N149Native(int width, int height, uint64_t tau_ticks, int radius, double threshold, bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off),
          tau_(std::max<uint64_t>(1, tau_ticks)),
          radius_(std::clamp(radius, 0, 8)),
          threshold_(threshold),
          sigma_(read_sigma()) {
        reset();
    }

    void reset() override {
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        last_ts_.assign(n, 0);
        last_pol_.assign(n, 0);
        hot_state_.assign(n, 0);
        beta_state_ = 0.0;
        mix_state_ = 0.0;
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

    py::array_t<float> score_batch(
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
        py::array_t<float> out(n);
        auto ob = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
            if (!visible(xb(i), yb(i), norm_pol(pb(i)))) {
                ob(i) = 0.0f;
                continue;
            }
            ob(i) = static_cast<float>(score_one(xb(i), yb(i), norm_pol(pb(i)), tb(i)));
        }
        return out;
    }

private:
    uint64_t tau_;
    int radius_;
    double threshold_;
    double sigma_;
    std::vector<uint64_t> last_ts_;
    std::vector<int8_t> last_pol_;
    std::vector<int32_t> hot_state_;
    double beta_state_ = 0.0;
    double mix_state_ = 0.0;

    static double read_sigma() {
        const char *v = std::getenv("MYEVS_N149_SIGMA");
        if (!v) return 2.5;
        try {
            const double s = std::stod(v);
            return std::isfinite(s) && s > 1e-6 ? s : 2.5;
        } catch (...) {
            return 2.5;
        }
    }

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    bool acc_neighbor(size_t k, uint64_t ti, int8_t pi, double w_space, double &raw_same, double &raw_opp, int &cnt_support) const {
        const int8_t pol_nb = last_pol_[k];
        if (pol_nb != pi && pol_nb != -pi) return false;
        const uint64_t ts = last_ts_[k];
        if (ts == 0 || ti <= ts) return false;
        const uint64_t dt = ti - ts;
        if (dt > tau_) return false;
        const double base_time = 1.0 - static_cast<double>(dt) / static_cast<double>(tau_);
        if (base_time <= 0.0) return false;
        const double wst = base_time * base_time * w_space;
        if (pol_nb == pi) {
            raw_same += wst;
            ++cnt_support;
        } else {
            raw_opp += wst;
        }
        return true;
    }

    double score_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return 0.0;
        const size_t idx0 = idx(x, y);
        int32_t h0 = hot_state_[idx0];
        const uint64_t ts0 = last_ts_[idx0];
        uint64_t dt0_u = tau_;
        if (ts0 != 0) dt0_u = abs_dt(t, ts0);
        const int64_t dt0 = static_cast<int64_t>(std::min<uint64_t>(dt0_u, static_cast<uint64_t>(std::numeric_limits<int32_t>::max())));
        if (dt0 != 0) {
            h0 -= static_cast<int32_t>(std::min<int64_t>(dt0, std::numeric_limits<int32_t>::max()));
            if (h0 < 0) h0 = 0;
        }
        const int64_t inc = static_cast<int64_t>(tau_) - static_cast<int64_t>(dt0_u);
        if (inc > 0) {
            const int64_t hv = static_cast<int64_t>(h0) + inc;
            h0 = static_cast<int32_t>(std::min<int64_t>(hv, std::numeric_limits<int32_t>::max()));
        }

        const int rr = radius_;
        const int x0 = std::max(0, x - rr);
        const int x1 = std::min(width_ - 1, x + rr);
        const int y0 = std::max(0, y - rr);
        const int y1 = std::min(height_ - 1, y + rr);
        const double inv_2sig2 = 1.0 / (2.0 * sigma_ * sigma_);
        double raw_same = 0.0;
        double raw_opp = 0.0;
        int cnt_support = 0;

        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) continue;
                const int dx = xx - x;
                const int dy = yy - y;
                const double w_space = std::exp(-static_cast<double>(dx * dx + dy * dy) * inv_2sig2);
                if (w_space <= 0.0) continue;
                acc_neighbor(idx(xx, yy), t, p, w_space, raw_same, raw_opp, cnt_support);
            }
        }

        last_ts_[idx0] = t;
        last_pol_[idx0] = p;
        hot_state_[idx0] = h0;

        const double tr = std::max<double>(1.0, static_cast<double>(tau_ / 2));
        double u_self = static_cast<double>(h0) / (static_cast<double>(h0) + tr + 1e-6);
        u_self = std::clamp(u_self, 0.0, 1.0);

        const double N = 4096.0;
        beta_state_ += (u_self - beta_state_) / N;
        beta_state_ = std::clamp(beta_state_, 0.0, 1.0);

        const double denom_mix = raw_same + raw_opp;
        double mix = 0.0;
        if (denom_mix > 0.0) {
            mix = raw_opp / (denom_mix + 1e-6);
            mix = std::clamp(mix, 0.0, 1.0);
        }
        mix_state_ += (mix - mix_state_) / N;
        mix_state_ = std::clamp(mix_state_, 0.0, 1.0);

        double alpha_eff = 1.0 - mix_state_;
        if (alpha_eff < 0.0) alpha_eff = 0.0;
        alpha_eff *= alpha_eff;

        const double raw_gated = raw_same + alpha_eff * raw_opp;
        const double base_score = raw_gated / (1.0 + u_self * u_self);
        const int cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1;
        double sfrac = 0.0;
        if (cnt_possible > 0) {
            sfrac = static_cast<double>(cnt_support) / static_cast<double>(cnt_possible);
            sfrac = std::clamp(sfrac, 0.0, 1.0);
        }
        return base_score * (1.0 + beta_state_ * sfrac);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) return false;
        return score_one(x, y, p, t) >= threshold_;
    }
};

} // namespace myevs_native_emlb
