#pragma once

#include "../common/native_common.hpp"

#include <array>

namespace myevs_native_emlb {

class N149RtlFixedNative : public NativeBase {
public:
    N149RtlFixedNative(int width, int height, uint64_t tau_ticks, int radius, uint64_t thr_ticks,
                       bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off),
          tau_(std::max<uint64_t>(1, tau_ticks)),
          radius_(std::clamp(radius, 0, 8)),
          thr_ticks_(thr_ticks) {
        reset();
    }

    void reset() override {
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        last_ts_.assign(n, 0);
        last_pol_.assign(n, 0);
        hot_state_.assign(n, 0);
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

    py::array_t<uint64_t> score_batch(
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
        py::array_t<uint64_t> out(n);
        auto ob = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
            ob(i) = score_one(xb(i), yb(i), norm_pol(pb(i)), tb(i));
        }
        return out;
    }

private:
    static constexpr uint64_t ACC_MASK = (uint64_t{1} << 56) - 1;
    static constexpr uint32_t HOT_UNIT = 32;
    static constexpr uint32_t HOT_DECAY_SHIFT = 9;
    static constexpr std::array<uint16_t, 17> F_LUT = {
        65535, 49152, 43691, 40960, 39322, 38229, 37449, 36864, 36409,
        36045, 35747, 35499, 35289, 35109, 34953, 34816, 34696
    };

    uint64_t tau_;
    int radius_;
    uint64_t thr_ticks_;
    std::vector<uint32_t> last_ts_;
    std::vector<uint8_t> last_pol_;
    std::vector<uint8_t> hot_state_;

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    static uint16_t w_same(int d2) noexcept {
        switch (d2) {
        case 1: return 55664;
        case 2: return 47279;
        case 4: return 34108;
        case 5: return 28970;
        case 8: return 17752;
        default: return 0;
        }
    }

    static uint16_t w_opp(int d2) noexcept {
        switch (d2) {
        case 1: return 2783;
        case 2: return 2364;
        case 4: return 1705;
        case 5: return 1448;
        case 8: return 888;
        default: return 0;
        }
    }

    static int32_t arithmetic_shift_right(int32_t v, int bits) noexcept {
        if (v >= 0) {
            return v >> bits;
        }
        const int32_t mag = -v;
        return -((mag + ((int32_t{1} << bits) - 1)) >> bits);
    }

    static uint16_t interp_f(uint8_t h) noexcept {
        const uint32_t i = static_cast<uint32_t>(h) >> 4;
        const uint32_t frac = static_cast<uint32_t>(h) & 0x0fU;
        const int32_t base = static_cast<int32_t>(F_LUT[i]);
        const int32_t diff = static_cast<int32_t>(F_LUT[i + 1]) - base;
        int32_t v = base + arithmetic_shift_right(diff * static_cast<int32_t>(frac), 4);
        if (v < 0) return 0;
        if (v > 65535) return 65535;
        return static_cast<uint16_t>(v);
    }

    static uint64_t scale_score(uint64_t score_acc, uint16_t f_q16) noexcept {
        const uint64_t hi = score_acc >> 32;
        const uint64_t lo = score_acc & 0xffffffffULL;
        return hi * static_cast<uint64_t>(f_q16) +
               ((lo * static_cast<uint64_t>(f_q16)) >> 32);
    }

    uint64_t score_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) return 0;
        const size_t idx0 = idx(x, y);
        const uint32_t ti = static_cast<uint32_t>(t);
        const uint8_t pi = p > 0 ? uint8_t{1} : uint8_t{0};

        const uint32_t ts0 = last_ts_[idx0];
        const uint32_t dt0 = (ts0 != 0 && ti >= ts0) ? (ti - ts0) : static_cast<uint32_t>(tau_);
        const uint32_t h_sum = static_cast<uint32_t>(hot_state_[idx0]) + HOT_UNIT;
        const uint32_t h_dec = (dt0 + ((uint32_t{1} << HOT_DECAY_SHIFT) - 1)) >> HOT_DECAY_SHIFT;
        uint32_t h_next = h_sum > h_dec ? (h_sum - h_dec) : 0;
        if (h_next > 255) h_next = 255;
        const uint8_t h_new = static_cast<uint8_t>(h_next);
        hot_state_[idx0] = h_new;

        uint64_t score_acc = 0;
        const int x0 = std::max(0, x - radius_);
        const int x1 = std::min(width_ - 1, x + radius_);
        const int y0 = std::max(0, y - radius_);
        const int y1 = std::min(height_ - 1, y + radius_);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) continue;
                const int dx = std::abs(xx - x);
                const int dy = std::abs(yy - y);
                const int d2 = dx * dx + dy * dy;
                const size_t k = idx(xx, yy);
                const uint16_t w = (last_pol_[k] == pi) ? w_same(d2) : w_opp(d2);
                if (w == 0) continue;
                const uint32_t ts = last_ts_[k];
                if (ts == 0 || ti <= ts) continue;
                const uint32_t dt = ti - ts;
                if (dt > tau_) continue;
                const uint32_t tau_m_dt = static_cast<uint32_t>(tau_ - dt);
                const uint32_t w_time_sq = tau_m_dt * tau_m_dt;
                const uint64_t score_inc = static_cast<uint64_t>(w_time_sq) * static_cast<uint64_t>(w);
                score_acc = (score_acc + score_inc) & ACC_MASK;
            }
        }

        last_ts_[idx0] = ti;
        last_pol_[idx0] = pi;
        return scale_score(score_acc, interp_f(h_new));
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        return score_one(x, y, p, t) >= thr_ticks_;
    }
};

} // namespace myevs_native_emlb
