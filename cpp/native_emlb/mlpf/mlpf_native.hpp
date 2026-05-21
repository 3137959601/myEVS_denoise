#pragma once

#include "../common/native_common.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace myevs_native_emlb {

class MlpfNative {
public:
    MlpfNative(
        int width,
        int height,
        uint64_t duration_ticks,
        double threshold,
        int patch,
        py::array_t<float, py::array::c_style | py::array::forcecast> fc1_weight,
        py::array_t<float, py::array::c_style | py::array::forcecast> fc1_bias,
        py::array_t<float, py::array::c_style | py::array::forcecast> fc2_weight,
        py::array_t<float, py::array::c_style | py::array::forcecast> fc2_bias,
        bool output_is_prob = false,
        bool show_on = true,
        bool show_off = true)
        : width_(width),
          height_(height),
          duration_ticks_(duration_ticks > 0 ? duration_ticks : 1),
          threshold_(threshold),
          patch_(patch),
          radius_(patch / 2),
          output_is_prob_(output_is_prob),
          show_on_(show_on),
          show_off_(show_off) {
        if (width_ <= 0 || height_ <= 0) {
            throw std::runtime_error("MlpfNative: width/height must be positive");
        }
        if (patch_ <= 0 || (patch_ % 2) == 0) {
            throw std::runtime_error("MlpfNative: patch must be odd and positive");
        }

        const auto w1 = fc1_weight.unchecked<2>();
        const auto b1 = fc1_bias.unchecked<1>();
        const auto w2 = fc2_weight.unchecked<2>();
        const auto b2 = fc2_bias.unchecked<1>();

        in_dim_ = static_cast<int>(w1.shape(1));
        hidden_dim_ = static_cast<int>(w1.shape(0));
        if (in_dim_ <= 0 || hidden_dim_ <= 0) {
            throw std::runtime_error("MlpfNative: invalid fc1 shape");
        }
        if (static_cast<int>(b1.shape(0)) != hidden_dim_) {
            throw std::runtime_error("MlpfNative: fc1 bias shape mismatch");
        }
        if (static_cast<int>(w2.shape(0)) != 1 || static_cast<int>(w2.shape(1)) != hidden_dim_) {
            throw std::runtime_error("MlpfNative: fc2 weight shape mismatch (expect 1 x hidden)");
        }
        if (static_cast<int>(b2.shape(0)) != 1) {
            throw std::runtime_error("MlpfNative: fc2 bias shape mismatch");
        }

        const int area = patch_ * patch_;
        if (in_dim_ != 2 * area) {
            throw std::runtime_error("MlpfNative: input dim mismatch with patch (expect 2*patch*patch)");
        }

        fc1_w_.assign(w1.data(0, 0), w1.data(0, 0) + static_cast<size_t>(hidden_dim_) * static_cast<size_t>(in_dim_));
        fc1_b_.assign(b1.data(0), b1.data(0) + static_cast<size_t>(hidden_dim_));
        fc2_w_.assign(w2.data(0, 0), w2.data(0, 0) + static_cast<size_t>(hidden_dim_));
        fc2_b_ = b2(0);

        reset();
    }

    void reset() {
        last_ts_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), 0);
        feat_.assign(static_cast<size_t>(in_dim_), 0.0f);
        hidden_.assign(static_cast<size_t>(hidden_dim_), 0.0f);
    }

    py::array_t<uint8_t> accept_batch(
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> t,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> x,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> y,
        py::array_t<int8_t, py::array::c_style | py::array::forcecast> p) {
        auto bt = t.unchecked<1>();
        auto bx = x.unchecked<1>();
        auto by = y.unchecked<1>();
        auto bp = p.unchecked<1>();
        const py::ssize_t n = bt.shape(0);
        if (bx.shape(0) != n || by.shape(0) != n || bp.shape(0) != n) {
            throw std::runtime_error("MlpfNative.accept_batch: input lengths mismatch");
        }

        py::array_t<uint8_t> keep_arr(n);
        auto keep = keep_arr.mutable_unchecked<1>();
        const float inv_win = 1.0f / static_cast<float>(duration_ticks_);
        const int area = patch_ * patch_;

        for (py::ssize_t i = 0; i < n; ++i) {
            const int xi = bx(i);
            const int yi = by(i);
            const uint64_t ti = bt(i);
            const int8_t pi = bp(i);

            if (!visible_local(xi, yi, pi)) {
                keep(i) = static_cast<uint8_t>(0);
                continue;
            }
            if (xi < 0 || yi < 0 || xi >= width_ || yi >= height_) {
                keep(i) = static_cast<uint8_t>(0);
                continue;
            }

            std::fill(feat_.begin(), feat_.end(), 0.0f);
            const float pol = (pi > 0) ? 1.0f : -1.0f;
            int k = 0;
            for (int dy = -radius_; dy <= radius_; ++dy) {
                const int yy = yi + dy;
                for (int dx = -radius_; dx <= radius_; ++dx) {
                    const int xx = xi + dx;
                    if (xx >= 0 && xx < width_ && yy >= 0 && yy < height_) {
                        const size_t idx = static_cast<size_t>(yy) * static_cast<size_t>(width_) + static_cast<size_t>(xx);
                        const uint64_t prev = last_ts_[idx];
                        const uint64_t dt = (ti >= prev) ? (ti - prev) : 0;
                        feat_[static_cast<size_t>(k)] = 1.0f - static_cast<float>(dt) * inv_win;
                        feat_[static_cast<size_t>(k + area)] = pol;
                    }
                    ++k;
                }
            }

            const size_t center_idx = static_cast<size_t>(yi) * static_cast<size_t>(width_) + static_cast<size_t>(xi);
            last_ts_[center_idx] = ti;

            for (int h = 0; h < hidden_dim_; ++h) {
                float s = fc1_b_[static_cast<size_t>(h)];
                const size_t woff = static_cast<size_t>(h) * static_cast<size_t>(in_dim_);
                for (int d = 0; d < in_dim_; ++d) {
                    s += fc1_w_[woff + static_cast<size_t>(d)] * feat_[static_cast<size_t>(d)];
                }
                hidden_[static_cast<size_t>(h)] = (s > 0.0f) ? s : 0.0f;
            }

            float out = fc2_b_;
            for (int h = 0; h < hidden_dim_; ++h) {
                out += fc2_w_[static_cast<size_t>(h)] * hidden_[static_cast<size_t>(h)];
            }
            const float prob = output_is_prob_ ? out : (1.0f / (1.0f + std::exp(-out)));
            keep(i) = static_cast<uint8_t>((prob >= static_cast<float>(threshold_)) ? 1 : 0);
        }
        return keep_arr;
    }

private:
    bool visible_local(int x, int y, int8_t p) const noexcept {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return false;
        if (p > 0 && !show_on_) return false;
        if (p < 0 && !show_off_) return false;
        return true;
    }

    int width_{0};
    int height_{0};
    uint64_t duration_ticks_{1};
    double threshold_{0.5};
    int patch_{7};
    int radius_{3};
    int in_dim_{0};
    int hidden_dim_{0};
    bool output_is_prob_{false};
    bool show_on_{true};
    bool show_off_{true};

    std::vector<uint64_t> last_ts_;
    std::vector<float> fc1_w_;
    std::vector<float> fc1_b_;
    std::vector<float> fc2_w_;
    float fc2_b_{0.0f};
    std::vector<float> feat_;
    std::vector<float> hidden_;
};

}  // namespace myevs_native_emlb
