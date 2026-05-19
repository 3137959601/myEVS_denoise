#pragma once

#include "../common/native_common.hpp"

namespace myevs_native_emlb {

class N149Native : public NativeBase {
public:
    N149Native(int width, int height, uint64_t duration_ticks, int radius, double threshold,
               bool show_on = true, bool show_off = true)
        : NativeBase(width, height, show_on, show_off),
          tau_(std::max<uint64_t>(1, duration_ticks)),
          radius_(std::clamp(radius, 0, 8)),
          threshold_(threshold),
          sigma_(std::max(0.1, read_sigma())),
          hot_bits_(read_hot_bits()),
          hot_mask_(hot_bits_ >= 31 ? 0x7FFFFFFF : ((1 << hot_bits_) - 1)),
          no_hot_(env_flag("MYEVS_N149_NO_HOT")),
          no_beta_(!env_flag("MYEVS_N149_USE_BETA")),
          no_mix_(env_flag("MYEVS_N149_NO_MIX")),
          no_opp_(env_flag("MYEVS_N149_NO_OPP")),
          no_sfrac_(!env_flag("MYEVS_N149_USE_SFRAC")),
          no_spatial_(env_flag("MYEVS_N149_NO_SPATIAL")),
          alpha_form_(read_alpha_form()),
          u_denom_factor_(read_u_denom_factor()),
          b_denom_form_(read_b_denom_form()),
          hot_k_(read_hot_k()),
          hot_binary_(env_flag("MYEVS_N149_HOT_BINARY")),
          hot_lut_enabled_(false),
          hot_lut_bits_(read_hot_lut_bits()),
          wt_linear_(env_flag("MYEVS_N149_WT_LINEAR")),
          simple_a_(env_flag("MYEVS_N149_SIMPLE_A")),
          simple_b_(env_flag("MYEVS_N149_SIMPLE_B")),
          use_ema_(env_flag("MYEVS_N149_USE_EMA")),
          alpha_instant_(!use_ema_),  // v2.2: default instant, EMA is opt-in
          alpha_fixed_(read_alpha_fixed()) {
        if (use_ema_) alpha_instant_ = false;  // EMA overrides instant
        if (simple_b_) { simple_a_ = true; no_hot_ = true; no_mix_ = true; no_opp_ = true; }
        if (simple_a_) { no_hot_ = true; no_mix_ = true; no_opp_ = true; }
        hot_lut_enabled_ = env_flag("MYEVS_N149_HOT_LUT");
        if (hot_lut_enabled_) build_hot_lut();
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
        auto tb = t.unchecked<1>(); auto xb = x.unchecked<1>(); auto yb = y.unchecked<1>(); auto pb = p.unchecked<1>();
        const auto n = tb.shape(0);
        if (xb.shape(0) != n || yb.shape(0) != n || pb.shape(0) != n)
            throw std::invalid_argument("t/x/y/p arrays must have the same length");
        py::array_t<uint8_t> out(n);
        auto ob = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i)
            ob(i) = accept_one(xb(i), yb(i), norm_pol(pb(i)), tb(i)) ? uint8_t{1} : uint8_t{0};
        return out;
    }

private:
    uint64_t tau_; int radius_; double threshold_; double sigma_;
    int hot_bits_; int32_t hot_mask_;
    bool no_hot_, no_beta_, no_mix_, no_opp_, no_sfrac_, no_spatial_;
    int alpha_form_; double u_denom_factor_; int b_denom_form_; double hot_k_;
    bool hot_binary_, hot_lut_enabled_, wt_linear_, simple_a_, simple_b_, alpha_instant_, use_ema_;
    double alpha_fixed_; int hot_lut_bits_;
    std::vector<double> hot_lut_tbl_; int hot_lut_step_;
    std::vector<uint64_t> last_ts_; std::vector<int8_t> last_pol_; std::vector<int32_t> hot_state_;
    double beta_state_ = 0.0; double mix_state_ = 0.0;

    size_t idx(int32_t x, int32_t y) const noexcept {
        return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x);
    }

    static bool env_flag(const char* name) {
        const char *v = std::getenv(name);
        return v && (std::string(v) == "1" || std::string(v) == "true");
    }
    static double read_sigma() {
        const char *v = std::getenv("MYEVS_N149_SIGMA"); if (!v) return 3.0;
        try { double d = std::stod(v); return std::isfinite(d) && d >= 0.1 ? d : 3.0; } catch (...) { return 3.0; }
    }
    static int read_hot_bits() {
        const char *v = std::getenv("MYEVS_N149_HOT_BITS"); if (!v) return 31;
        try { int b = std::stoi(v); return std::clamp(b, 2, 31); } catch (...) { return 31; }
    }
    static int read_alpha_form() {
        const char *v = std::getenv("MYEVS_N149_ALPHA_FORM"); if (!v) return 0;
        std::string s(v); if (s == "lin" || s == "1-m") return 1; if (s == "m2" || s == "1-m2") return 2; return 0;
    }
    static double read_u_denom_factor() {
        const char *v = std::getenv("MYEVS_N149_U_DENOM"); if (!v) return 1.0;
        try { double d = std::stod(v); return std::isfinite(d) && d > 0.01 ? d : 1.0; } catch (...) { return 1.0; }
    }
    static int read_b_denom_form() {
        const char *v = std::getenv("MYEVS_N149_B_DENOM"); if (!v) return 1;
        std::string s(v); if (s == "1+u2") return 0; if (s == "(1+u)2") return 2; return 1;
    }
    static double read_hot_k() {
        const char *v = std::getenv("MYEVS_N149_HOT_K"); if (!v) return 2.0;
        try { double d = std::stod(v); return std::isfinite(d) && d >= 1.0 ? d : 2.0; } catch (...) { return 2.0; }
    }
    static double read_alpha_fixed() {
        const char *v = std::getenv("MYEVS_N149_ALPHA_FIXED"); if (!v) return 0.25;  // v2.2 default
        try { double d = std::stod(v); return std::isfinite(d) && d >= 0.0 ? d : 0.25; } catch (...) { return 0.25; }
    }
    static int read_hot_lut_bits() {
        const char *v = std::getenv("MYEVS_N149_HOT_LUT_BITS"); if (!v) return 4;
        try { int b = std::stoi(v); return std::clamp(b, 2, 8); } catch (...) { return 4; }
    }
    void build_hot_lut() {
        int lut_entries = 1 << hot_lut_bits_, h_max = 1 << hot_bits_, step = h_max >> hot_lut_bits_;
        double tr = std::max<double>(1.0, static_cast<double>(tau_) * u_denom_factor_);
        hot_lut_tbl_.resize(lut_entries + 1);
        for (int i = 0; i <= lut_entries; ++i)
            hot_lut_tbl_[i] = (static_cast<double>(i*step) + tr) / (hot_k_ * static_cast<double>(i*step) + tr);
        hot_lut_step_ = step;
    }

    bool acc_neighbor(size_t k, uint64_t ti, int8_t pi, double w_space, double &raw_same, double &raw_opp, int &cnt_support) const {
        if (simple_b_) { /* all polarities */ }
        else if (simple_a_) { if (last_pol_[k] != pi) return false; }
        else { int8_t pn = last_pol_[k]; if (pn != pi && pn != -pi) return false; }
        uint64_t ts = last_ts_[k];
        if (ts == 0 || ti <= ts) return false;
        uint64_t dt = ti - ts;
        if (dt > tau_) return false;
        double base_time = 1.0 - static_cast<double>(dt) / static_cast<double>(tau_);
        if (base_time <= 0.0) return false;
        double wst = (wt_linear_ ? base_time : base_time * base_time) * w_space;
        if (simple_a_ || simple_b_) raw_same += wst;
        else if (last_pol_[k] == pi) { raw_same += wst; ++cnt_support; }
        else raw_opp += wst;
        return true;
    }

    double score_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) return 0.0;
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return 0.0;
        size_t idx0 = idx(x, y);

        int32_t h0 = hot_state_[idx0];
        uint64_t ts0 = last_ts_[idx0];
        uint64_t dt0_u = tau_; if (ts0 != 0) dt0_u = abs_dt(t, ts0);
        int64_t dt0 = static_cast<int64_t>(std::min<uint64_t>(dt0_u, static_cast<uint64_t>(std::numeric_limits<int32_t>::max())));
        int64_t h_new = static_cast<int64_t>(h0) + static_cast<int64_t>(tau_) - 2 * dt0;
        h0 = static_cast<int32_t>(std::clamp<int64_t>(h_new, 0, std::numeric_limits<int32_t>::max()));
        h0 = (h0 > hot_mask_) ? hot_mask_ : h0;

        int rr = radius_, x0 = std::max(0, x - rr), x1 = std::min(width_ - 1, x + rr);
        int y0 = std::max(0, y - rr), y1 = std::min(height_ - 1, y + rr);
        double inv_2sig2 = 1.0 / (2.0 * sigma_ * sigma_);
        double raw_same = 0.0, raw_opp = 0.0; int cnt_support = 0;
        for (int yy = y0; yy <= y1; ++yy)
            for (int xx = x0; xx <= x1; ++xx) {
                if (xx == x && yy == y) continue;
                double w_space = no_spatial_ ? 1.0 : std::exp(-static_cast<double>((xx-x)*(xx-x) + (yy-y)*(yy-y)) * inv_2sig2);
                if (w_space <= 0.0) continue;
                acc_neighbor(idx(xx, yy), t, p, w_space, raw_same, raw_opp, cnt_support);
            }
        if (no_opp_) raw_opp = 0.0;

        last_ts_[idx0] = t;
        if (!simple_b_) last_pol_[idx0] = p;
        hot_state_[idx0] = h0;

        double tr = std::max<double>(1.0, static_cast<double>(tau_) * u_denom_factor_);
        double discount_num = static_cast<double>(h0) + tr, discount_factor;
        if (hot_lut_enabled_) {
            int hc = h0 & hot_mask_, i = hc >> (hot_bits_ - hot_lut_bits_);
            if (i >= (1 << hot_lut_bits_)) i = (1 << hot_lut_bits_) - 1;
            int frac = hc & (hot_lut_step_ - 1);
            double diff = hot_lut_tbl_[i + 1] - hot_lut_tbl_[i];
            discount_factor = hot_lut_tbl_[i] + (diff * static_cast<double>(frac)) / static_cast<double>(hot_lut_step_);
        } else if (hot_binary_) {
            discount_factor = (static_cast<double>(h0) > tr) ? hot_k_ : 1.0;
        } else {
            discount_factor = discount_num / (hot_k_ * static_cast<double>(h0) + tr);
        }

        double N = 4096.0;
        double u_self = no_hot_ ? 0.0 : std::clamp(static_cast<double>(h0) / discount_num, 0.0, 1.0);
        beta_state_ += (u_self - beta_state_) / N;
        beta_state_ = no_beta_ ? 0.0 : std::clamp(beta_state_, 0.0, 1.0);

        // v2.2: alpha logic 鈥?default instant+fixed(0.25), EMA is opt-in
        double dm = raw_same + raw_opp, mix = 0.0;
        if (dm > 0.0) { mix = raw_opp / (dm + 1e-6); mix = std::clamp(mix, 0.0, 1.0); }
        if (use_ema_ && !simple_a_) {
            mix_state_ += (mix - mix_state_) / N;
            mix_state_ = no_mix_ ? 0.0 : std::clamp(mix_state_, 0.0, 1.0);
        }
        double alpha_eff = alpha_fixed_;  // v2.2: default 0.25, overridable
        if (!use_ema_ && alpha_fixed_ < 0.0) {
            // v2.1 compat: instant mix with formula when no fixed alpha
            double m_eff = mix;
            alpha_eff = 1.0 - m_eff; if (alpha_eff < 0.0) alpha_eff = 0.0;
            if (alpha_form_ == 1) {}
            else if (alpha_form_ == 2) { alpha_eff = 1.0 - m_eff*m_eff; if (alpha_eff < 0.0) alpha_eff = 0.0; }
            else alpha_eff *= alpha_eff;
        }

        double raw_gated = raw_same + alpha_eff * raw_opp;
        double base_score;
        if (b_denom_form_ == 1) base_score = no_hot_ ? raw_gated : (raw_gated * discount_factor);
        else if (b_denom_form_ == 2) { double u = no_hot_ ? 0.0 : std::clamp(h0/discount_num, 0.0, 1.0); double t2 = 1.0+u; base_score = raw_gated/(t2*t2); }
        else { double u = no_hot_ ? 0.0 : std::clamp(h0/discount_num, 0.0, 1.0); base_score = raw_gated/(1.0+u*u); }

        int cnt_possible = (x1-x0+1)*(y1-y0+1)-1;
        double sfrac = 0.0;
        if (cnt_possible > 0) { sfrac = static_cast<double>(cnt_support)/static_cast<double>(cnt_possible); sfrac = std::clamp(sfrac, 0.0, 1.0); }
        return base_score * (1.0 + (no_sfrac_ ? 0.0 : beta_state_) * sfrac);
    }

    bool accept_one(int32_t x, int32_t y, int8_t p, uint64_t t) {
        if (!visible(x, y, p)) return false;
        return score_one(x, y, p, t) >= threshold_;
    }
};

} // namespace myevs_native_emlb
