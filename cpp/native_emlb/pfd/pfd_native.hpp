#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class PfdNative : public NativeBase {
public:
    PfdNative(
        int width,
        int height,
        uint64_t duration_ticks,
        int radius,
        double min_neighbors,
        int stage1_var,
        bool mode_b = false,
        bool show_on = true,
        bool show_off = true)
        : NativeBase(width, height, show_on, show_off),
          win_ticks_(duration_ticks == 0 ? uint64_t{1} : duration_ticks),
          radius_(std::max(1, std::min(radius, 8))),
          min_neighbors_(min_neighbors),
          stage1_var_(std::max(1, stage1_var)),
          mode_b_(mode_b),
          fifo_size_(5) {
        const size_t n = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        last_on_.assign(n, 0);
        last_off_.assign(n, 0);
        last_evt_.assign(n, 0);
        last_pol_.assign(n, int8_t{0});
        flip_buf_.assign(n * fifo_size_, 0);
        flip_head_.assign(n, 0);
        flip_count_.assign(n, 0);
    }

    void reset() override {
        std::fill(last_on_.begin(), last_on_.end(), uint64_t{0});
        std::fill(last_off_.begin(), last_off_.end(), uint64_t{0});
        std::fill(last_evt_.begin(), last_evt_.end(), uint64_t{0});
        std::fill(last_pol_.begin(), last_pol_.end(), int8_t{0});
        std::fill(flip_buf_.begin(), flip_buf_.end(), uint64_t{0});
        std::fill(flip_head_.begin(), flip_head_.end(), int32_t{0});
        std::fill(flip_count_.begin(), flip_count_.end(), int32_t{0});
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
    uint64_t win_ticks_;
    int radius_;
    double min_neighbors_;
    int stage1_var_;
    bool mode_b_;
    int fifo_size_;

    std::vector<uint64_t> last_on_;
    std::vector<uint64_t> last_off_;
    std::vector<uint64_t> last_evt_;
    std::vector<int8_t> last_pol_;

    std::vector<uint64_t> flip_buf_; // [pixel * fifo_size + slot]
    std::vector<int32_t> flip_head_;
    std::vector<int32_t> flip_count_;

    inline size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    inline size_t fidx(size_t pix, int32_t slot) const noexcept {
        return pix * static_cast<size_t>(fifo_size_) + static_cast<size_t>(slot);
    }

    void push_flip(size_t pix, uint64_t t) {
        int32_t c = flip_count_[pix];
        int32_t h = flip_head_[pix];
        if (c < fifo_size_) {
            const int32_t pos = (h + c) % fifo_size_;
            flip_buf_[fidx(pix, pos)] = t;
            flip_count_[pix] = c + 1;
            return;
        }
        flip_buf_[fidx(pix, h)] = t;
        flip_head_[pix] = (h + 1) % fifo_size_;
    }

    int count_recent_flips(size_t pix, uint64_t t) const {
        const int32_t c = flip_count_[pix];
        const int32_t h = flip_head_[pix];
        int cnt = 0;
        for (int32_t k = 0; k < c; ++k) {
            const int32_t pos = (h + k) % fifo_size_;
            const uint64_t ft = flip_buf_[fidx(pix, pos)];
            if (ft == 0) {
                continue;
            }
            if (abs_dt(t, ft) <= win_ticks_) {
                ++cnt;
            }
        }
        return cnt;
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) {
            return false;
        }

        const size_t i0 = idx(x, y);

        // Update polarity/flip state first (same order as python op).
        const int8_t lp = last_pol_[i0];
        if (lp != 0 && lp != p) {
            push_flip(i0, t);
        }
        last_pol_[i0] = p;
        last_evt_[i0] = t;

        // Stage-1 same-polarity support.
        std::vector<uint64_t> &same = (p > 0) ? last_on_ : last_off_;
        same[i0] = t;

        const int x0 = std::max(0, x - radius_);
        const int x1 = std::min(width_ - 1, x + radius_);
        const int y0 = std::max(0, y - radius_);
        const int y1 = std::min(height_ - 1, y + radius_);

        int support = 0;
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) {
                    continue;
                }
                const uint64_t ts = same[idx(xx, yy)];
                if (ts == 0) {
                    continue;
                }
                if (abs_dt(t, ts) < win_ticks_) {
                    ++support;
                }
            }
        }
        const int max_var = (2 * radius_ + 1) * (2 * radius_ + 1) - 1;
        const int var = std::max(1, std::min(stage1_var_, max_var));
        if (support < var) {
            return false;
        }

        // Stage-2 polarity-flip consistency.
        const int cur_flip = count_recent_flips(i0, t);
        int neigh_active = 0;
        int neigh_flip_sum = 0;
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) {
                    continue;
                }
                const size_t j = idx(xx, yy);
                const uint64_t te = last_evt_[j];
                if (te != 0 && abs_dt(t, te) <= win_ticks_) {
                    ++neigh_active;
                }
                neigh_flip_sum += count_recent_flips(j, t);
            }
        }

        if (static_cast<double>(neigh_active) <= min_neighbors_) {
            return false;
        }

        const double neigh_flip_mean = static_cast<double>(neigh_flip_sum) / static_cast<double>(neigh_active);
        const double score = mode_b_ ? std::abs(neigh_flip_mean)
                                     : std::abs(static_cast<double>(cur_flip) - neigh_flip_mean);
        return score <= 1.0;
    }
};

} // namespace myevs_native_emlb

