#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Events {
    std::uint64_t n = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint64_t> t;
    std::vector<std::uint16_t> x;
    std::vector<std::uint16_t> y;
    std::vector<std::int8_t> p;
    std::vector<std::uint8_t> label;
};

struct Offset {
    int dx = 0;
    int dy = 0;
    int delta = 0;
    double w = 1.0;
};

struct OffsetF {
    int dx = 0;
    int dy = 0;
    int delta = 0;
    float w = 1.0f;
};

static void read_exact(std::ifstream& f, char* dst, std::size_t bytes) {
    f.read(dst, static_cast<std::streamsize>(bytes));
    if (!f) {
        throw std::runtime_error("unexpected EOF");
    }
}

static Events read_events(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open input bin: " + path);
    }
    char magic[8];
    Events ev;
    read_exact(f, magic, 8);
    if (std::string(magic, magic + 8) != "MYEVSBIN") {
        throw std::runtime_error("bad input magic");
    }
    read_exact(f, reinterpret_cast<char*>(&ev.n), sizeof(ev.n));
    read_exact(f, reinterpret_cast<char*>(&ev.width), sizeof(ev.width));
    read_exact(f, reinterpret_cast<char*>(&ev.height), sizeof(ev.height));
    ev.t.resize(static_cast<std::size_t>(ev.n));
    ev.x.resize(static_cast<std::size_t>(ev.n));
    ev.y.resize(static_cast<std::size_t>(ev.n));
    ev.p.resize(static_cast<std::size_t>(ev.n));
    ev.label.resize(static_cast<std::size_t>(ev.n));
    read_exact(f, reinterpret_cast<char*>(ev.t.data()), ev.t.size() * sizeof(ev.t[0]));
    read_exact(f, reinterpret_cast<char*>(ev.x.data()), ev.x.size() * sizeof(ev.x[0]));
    read_exact(f, reinterpret_cast<char*>(ev.y.data()), ev.y.size() * sizeof(ev.y[0]));
    read_exact(f, reinterpret_cast<char*>(ev.p.data()), ev.p.size() * sizeof(ev.p[0]));
    read_exact(f, reinterpret_cast<char*>(ev.label.data()), ev.label.size() * sizeof(ev.label[0]));
    return ev;
}

static void write_scores(const std::string& path, const std::vector<float>& scores) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open score output: " + path);
    }
    f.write(reinterpret_cast<const char*>(scores.data()), static_cast<std::streamsize>(scores.size() * sizeof(scores[0])));
}

static std::vector<Offset> build_dense_offsets(int rr, int width, bool uniform_space) {
    const double sigma = 2.8;
    const double inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
    std::vector<Offset> offsets;
    offsets.reserve(static_cast<std::size_t>((2 * rr + 1) * (2 * rr + 1) - 1));
    for (int dy = -rr; dy <= rr; ++dy) {
        for (int dx = -rr; dx <= rr; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int d2 = dx * dx + dy * dy;
            Offset o;
            o.dx = dx;
            o.dy = dy;
            o.delta = dy * width + dx;
            o.w = uniform_space ? 1.0 : std::exp(-static_cast<double>(d2) * inv_2sig2);
            offsets.push_back(o);
        }
    }
    return offsets;
}

static std::vector<OffsetF> build_dense_offsets_f(int rr, int width, bool uniform_space) {
    const float sigma = 2.8f;
    const float inv_2sig2 = 1.0f / (2.0f * sigma * sigma);
    std::vector<OffsetF> offsets;
    offsets.reserve(static_cast<std::size_t>((2 * rr + 1) * (2 * rr + 1) - 1));
    for (int dy = -rr; dy <= rr; ++dy) {
        for (int dx = -rr; dx <= rr; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int d2 = dx * dx + dy * dy;
            OffsetF o;
            o.dx = dx;
            o.dy = dy;
            o.delta = dy * width + dx;
            o.w = uniform_space ? 1.0f : std::exp(-static_cast<float>(d2) * inv_2sig2);
            offsets.push_back(o);
        }
    }
    return offsets;
}

static void score_n176_fast(
    const Events& ev,
    int s,
    int tau_us,
    double tick_ns,
    bool uniform_space,
    std::vector<float>& out
) {
    const int width = static_cast<int>(ev.width);
    const int height = static_cast<int>(ev.height);
    const int rr = std::max(0, (s - 1) / 2);
    const std::size_t npx = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint64_t> last_ts(npx, 0);
    std::vector<std::int8_t> last_pol(npx, 0);
    out.assign(static_cast<std::size_t>(ev.n), 0.0f);

    int tau_ticks = static_cast<int>(std::llround(static_cast<double>(tau_us) * 1000.0 / tick_ns));
    if (tau_ticks <= 0) {
        tau_ticks = 1;
    }
    const double inv_tau = 1.0 / static_cast<double>(tau_ticks);
    const auto offsets = build_dense_offsets(rr, width, uniform_space);

    const double N = 4096.0;
    const double eps = 1e-6;
    const double ks = 4.0 / 5.0;
    const double km = 1.0 / 8.0;
    double b = 0.65;
    double mstate = 0.0;
    double rstate = 0.10;

    for (std::size_t i = 0; i < static_cast<std::size_t>(ev.n); ++i) {
        const int xi = static_cast<int>(ev.x[i]);
        const int yi = static_cast<int>(ev.y[i]);
        if (xi < 0 || xi >= width || yi < 0 || yi >= height) {
            out[i] = 0.0f;
            continue;
        }
        const std::uint64_t ti_u = ev.t[i];
        const int pi = static_cast<int>(ev.p[i]) > 0 ? 1 : -1;
        const int idx0 = yi * width + xi;
        const bool interior = xi >= rr && xi < width - rr && yi >= rr && yi < height - rr;

        double raw_same = 0.0;
        double raw_opp = 0.0;
        int cnt_support = 0;

        if (interior) {
            for (const Offset& o : offsets) {
                const int idx = idx0 + o.delta;
                const int pol_nb = static_cast<int>(last_pol[static_cast<std::size_t>(idx)]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint64_t ts = last_ts[static_cast<std::size_t>(idx)];
                if (ts == 0 || ti_u <= ts) {
                    continue;
                }
                const std::uint64_t dt_u = ti_u - ts;
                if (dt_u > static_cast<std::uint64_t>(tau_ticks)) {
                    continue;
                }
                const double base_time = 1.0 - static_cast<double>(dt_u) * inv_tau;
                if (base_time <= 0.0) {
                    continue;
                }
                const double wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        } else {
            for (const Offset& o : offsets) {
                const int nx = xi + o.dx;
                const int ny = yi + o.dy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }
                const int idx = ny * width + nx;
                const int pol_nb = static_cast<int>(last_pol[static_cast<std::size_t>(idx)]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint64_t ts = last_ts[static_cast<std::size_t>(idx)];
                if (ts == 0 || ti_u <= ts) {
                    continue;
                }
                const std::uint64_t dt_u = ti_u - ts;
                if (dt_u > static_cast<std::uint64_t>(tau_ticks)) {
                    continue;
                }
                const double base_time = 1.0 - static_cast<double>(dt_u) * inv_tau;
                if (base_time <= 0.0) {
                    continue;
                }
                const double wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        }

        const int prev_pol0 = static_cast<int>(last_pol[static_cast<std::size_t>(idx0)]);
        const std::uint64_t ts0 = last_ts[static_cast<std::size_t>(idx0)];
        std::uint64_t dt0 = static_cast<std::uint64_t>(tau_ticks);
        if (ts0 != 0) {
            dt0 = ti_u >= ts0 ? ti_u - ts0 : ts0 - ti_u;
            if (dt0 > static_cast<std::uint64_t>(tau_ticks)) {
                dt0 = static_cast<std::uint64_t>(tau_ticks);
            }
        }

        double mix = 0.0;
        const double denom_mix = raw_same + raw_opp;
        if (denom_mix > 0.0) {
            mix = raw_opp / (denom_mix + eps);
            mix = std::min(1.0, std::max(0.0, mix));
        }
        mstate += (mix - mstate) / N;
        mstate = std::min(1.0, std::max(0.0, mstate));
        double alpha_eff = 1.0 - mstate;
        alpha_eff = std::max(0.0, alpha_eff) * std::max(0.0, alpha_eff);

        double u_lite = 1.0 - static_cast<double>(dt0) * inv_tau;
        u_lite = std::min(1.0, std::max(0.0, u_lite));
        double rhythm_bad = 0.0;
        double rhythm_good = 0.0;
        if (prev_pol0 == -pi) {
            rhythm_bad = u_lite;
        } else if (prev_pol0 == pi) {
            rhythm_good = u_lite;
        }
        rstate += (rhythm_bad - rstate) / N;
        rstate = std::min(1.0, std::max(0.0, rstate));
        double rhythm_pressure = 0.5 * (rhythm_bad + rstate);
        rhythm_pressure = std::min(1.0, std::max(0.0, rhythm_pressure));

        const int cnt_possible = interior ? static_cast<int>(offsets.size()) : ((std::min(width - 1, xi + rr) - std::max(0, xi - rr) + 1) * (std::min(height - 1, yi + rr) - std::max(0, yi - rr) + 1) - 1);
        double sfrac = cnt_possible > 0 ? static_cast<double>(cnt_support) / static_cast<double>(cnt_possible) : 0.0;
        sfrac = std::min(1.0, std::max(0.0, sfrac));
        alpha_eff *= (1.0 - rhythm_pressure / 3.0);
        alpha_eff = std::max(0.0, alpha_eff);

        double relief = 1.0 - ks * sfrac;
        relief = std::min(1.0, std::max(0.25, relief));
        double mix_gain = 1.0 + km * mix;
        mix_gain = std::min(2.0, std::max(0.5, mix_gain));
        double rhythm_scale = 1.0 + 0.5 * rhythm_pressure - 0.25 * rhythm_good;
        rhythm_scale = std::min(1.75, std::max(0.5, rhythm_scale));
        double u_eff = u_lite * relief * mix_gain * rhythm_scale;
        u_eff = std::min(1.0, std::max(0.0, u_eff));
        b += (u_eff - b) / N;
        b = std::min(1.0, std::max(0.0, b));

        const double raw_gated = raw_same + alpha_eff * raw_opp;
        const double base_score = raw_gated / (1.0 + u_eff * u_eff);
        double support_scale = 1.0 + b * sfrac * (1.0 + 0.25 * rhythm_good);
        support_scale = std::min(2.0, std::max(1.0, support_scale));
        out[i] = static_cast<float>(base_score * support_scale);

        last_ts[static_cast<std::size_t>(idx0)] = ti_u;
        last_pol[static_cast<std::size_t>(idx0)] = static_cast<std::int8_t>(pi);
    }
}

static void score_n176_fast32(
    const Events& ev,
    int s,
    int tau_us,
    double tick_ns,
    bool uniform_space,
    std::vector<float>& out
) {
    const int width = static_cast<int>(ev.width);
    const int height = static_cast<int>(ev.height);
    const int rr = std::max(0, (s - 1) / 2);
    const std::size_t npx = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint32_t> last_ts(npx, 0);
    std::vector<std::int8_t> last_pol(npx, 0);
    out.assign(static_cast<std::size_t>(ev.n), 0.0f);

    int tau_ticks = static_cast<int>(std::llround(static_cast<double>(tau_us) * 1000.0 / tick_ns));
    if (tau_ticks <= 0) {
        tau_ticks = 1;
    }
    const std::uint32_t tau_ticks_u = static_cast<std::uint32_t>(tau_ticks);
    const float inv_tau = 1.0f / static_cast<float>(tau_ticks);
    const auto offsets = build_dense_offsets_f(rr, width, uniform_space);

    const float inv_N = 1.0f / 4096.0f;
    const float eps = 1e-6f;
    const float ks = 4.0f / 5.0f;
    const float km = 1.0f / 8.0f;
    float b = 0.65f;
    float mstate = 0.0f;
    float rstate = 0.10f;

    for (std::size_t i = 0; i < static_cast<std::size_t>(ev.n); ++i) {
        const int xi = static_cast<int>(ev.x[i]);
        const int yi = static_cast<int>(ev.y[i]);
        if (xi < 0 || xi >= width || yi < 0 || yi >= height) {
            out[i] = 0.0f;
            continue;
        }
        const std::uint32_t ti_u = static_cast<std::uint32_t>(ev.t[i]);
        const int pi = static_cast<int>(ev.p[i]) > 0 ? 1 : -1;
        const int idx0 = yi * width + xi;
        const bool interior = xi >= rr && xi < width - rr && yi >= rr && yi < height - rr;

        float raw_same = 0.0f;
        float raw_opp = 0.0f;
        int cnt_support = 0;

        if (interior) {
            for (const OffsetF& o : offsets) {
                const int idx = idx0 + o.delta;
                const int pol_nb = static_cast<int>(last_pol[static_cast<std::size_t>(idx)]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint32_t ts = last_ts[static_cast<std::size_t>(idx)];
                if (ts == 0 || ti_u <= ts) {
                    continue;
                }
                const std::uint32_t dt_u = ti_u - ts;
                if (dt_u > tau_ticks_u) {
                    continue;
                }
                const float base_time = 1.0f - static_cast<float>(dt_u) * inv_tau;
                if (base_time <= 0.0f) {
                    continue;
                }
                const float wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        } else {
            for (const OffsetF& o : offsets) {
                const int nx = xi + o.dx;
                const int ny = yi + o.dy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }
                const int idx = ny * width + nx;
                const int pol_nb = static_cast<int>(last_pol[static_cast<std::size_t>(idx)]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint32_t ts = last_ts[static_cast<std::size_t>(idx)];
                if (ts == 0 || ti_u <= ts) {
                    continue;
                }
                const std::uint32_t dt_u = ti_u - ts;
                if (dt_u > tau_ticks_u) {
                    continue;
                }
                const float base_time = 1.0f - static_cast<float>(dt_u) * inv_tau;
                if (base_time <= 0.0f) {
                    continue;
                }
                const float wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        }

        const int prev_pol0 = static_cast<int>(last_pol[static_cast<std::size_t>(idx0)]);
        const std::uint32_t ts0 = last_ts[static_cast<std::size_t>(idx0)];
        std::uint32_t dt0 = tau_ticks_u;
        if (ts0 != 0) {
            dt0 = ti_u >= ts0 ? ti_u - ts0 : ts0 - ti_u;
            if (dt0 > tau_ticks_u) {
                dt0 = tau_ticks_u;
            }
        }

        float mix = 0.0f;
        const float denom_mix = raw_same + raw_opp;
        if (denom_mix > 0.0f) {
            mix = raw_opp / (denom_mix + eps);
            mix = std::min(1.0f, std::max(0.0f, mix));
        }
        mstate += (mix - mstate) * inv_N;
        mstate = std::min(1.0f, std::max(0.0f, mstate));
        float alpha_eff = 1.0f - mstate;
        alpha_eff = std::max(0.0f, alpha_eff) * std::max(0.0f, alpha_eff);

        float u_lite = 1.0f - static_cast<float>(dt0) * inv_tau;
        u_lite = std::min(1.0f, std::max(0.0f, u_lite));
        float rhythm_bad = 0.0f;
        float rhythm_good = 0.0f;
        if (prev_pol0 == -pi) {
            rhythm_bad = u_lite;
        } else if (prev_pol0 == pi) {
            rhythm_good = u_lite;
        }
        rstate += (rhythm_bad - rstate) * inv_N;
        rstate = std::min(1.0f, std::max(0.0f, rstate));
        float rhythm_pressure = 0.5f * (rhythm_bad + rstate);
        rhythm_pressure = std::min(1.0f, std::max(0.0f, rhythm_pressure));

        const int cnt_possible = interior ? static_cast<int>(offsets.size()) : ((std::min(width - 1, xi + rr) - std::max(0, xi - rr) + 1) * (std::min(height - 1, yi + rr) - std::max(0, yi - rr) + 1) - 1);
        float sfrac = cnt_possible > 0 ? static_cast<float>(cnt_support) / static_cast<float>(cnt_possible) : 0.0f;
        sfrac = std::min(1.0f, std::max(0.0f, sfrac));
        alpha_eff *= (1.0f - rhythm_pressure / 3.0f);
        alpha_eff = std::max(0.0f, alpha_eff);

        float relief = 1.0f - ks * sfrac;
        relief = std::min(1.0f, std::max(0.25f, relief));
        float mix_gain = 1.0f + km * mix;
        mix_gain = std::min(2.0f, std::max(0.5f, mix_gain));
        float rhythm_scale = 1.0f + 0.5f * rhythm_pressure - 0.25f * rhythm_good;
        rhythm_scale = std::min(1.75f, std::max(0.5f, rhythm_scale));
        float u_eff = u_lite * relief * mix_gain * rhythm_scale;
        u_eff = std::min(1.0f, std::max(0.0f, u_eff));
        b += (u_eff - b) * inv_N;
        b = std::min(1.0f, std::max(0.0f, b));

        const float raw_gated = raw_same + alpha_eff * raw_opp;
        const float base_score = raw_gated / (1.0f + u_eff * u_eff);
        float support_scale = 1.0f + b * sfrac * (1.0f + 0.25f * rhythm_good);
        support_scale = std::min(2.0f, std::max(1.0f, support_scale));
        out[i] = base_score * support_scale;

        last_ts[static_cast<std::size_t>(idx0)] = ti_u;
        last_pol[static_cast<std::size_t>(idx0)] = static_cast<std::int8_t>(pi);
    }
}

static void score_n176_fast16q(
    const Events& ev,
    int s,
    int tau_us,
    double tick_ns,
    bool uniform_space,
    std::vector<float>& out
) {
    const int width = static_cast<int>(ev.width);
    const int height = static_cast<int>(ev.height);
    const int rr = std::max(0, (s - 1) / 2);
    const std::size_t npx = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint16_t> last_ts(npx, 0);
    std::vector<std::int8_t> last_pol(npx, 0);
    out.assign(static_cast<std::size_t>(ev.n), 0.0f);

    int tau_ticks = static_cast<int>(std::llround(static_cast<double>(tau_us) * 1000.0 / tick_ns));
    if (tau_ticks <= 0) {
        tau_ticks = 1;
    }
    int shift = 0;
    while ((tau_ticks >> shift) > 60000 && shift < 16) {
        ++shift;
    }
    int tau_q = tau_ticks >> shift;
    if (tau_q <= 0) {
        tau_q = 1;
    }
    const std::uint16_t tau_q_u = static_cast<std::uint16_t>(tau_q);
    const float inv_tau_q = 1.0f / static_cast<float>(tau_q);
    const auto offsets = build_dense_offsets_f(rr, width, uniform_space);

    const float inv_N = 1.0f / 4096.0f;
    const float eps = 1e-6f;
    const float ks = 4.0f / 5.0f;
    const float km = 1.0f / 8.0f;
    float b = 0.65f;
    float mstate = 0.0f;
    float rstate = 0.10f;

    for (std::size_t i = 0; i < static_cast<std::size_t>(ev.n); ++i) {
        const int xi = static_cast<int>(ev.x[i]);
        const int yi = static_cast<int>(ev.y[i]);
        if (xi < 0 || xi >= width || yi < 0 || yi >= height) {
            out[i] = 0.0f;
            continue;
        }
        const std::uint16_t ti_q = static_cast<std::uint16_t>(static_cast<std::uint64_t>(ev.t[i]) >> shift);
        const int pi = static_cast<int>(ev.p[i]) > 0 ? 1 : -1;
        const int idx0 = yi * width + xi;
        const bool interior = xi >= rr && xi < width - rr && yi >= rr && yi < height - rr;

        float raw_same = 0.0f;
        float raw_opp = 0.0f;
        int cnt_support = 0;

        if (interior) {
            for (const OffsetF& o : offsets) {
                const int idx = idx0 + o.delta;
                const int pol_nb = static_cast<int>(last_pol[static_cast<std::size_t>(idx)]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint16_t ts = last_ts[static_cast<std::size_t>(idx)];
                if (ts == 0) {
                    continue;
                }
                const std::uint16_t dt_q = static_cast<std::uint16_t>(ti_q - ts);
                if (dt_q == 0 || dt_q > tau_q_u) {
                    continue;
                }
                const float base_time = 1.0f - static_cast<float>(dt_q) * inv_tau_q;
                if (base_time <= 0.0f) {
                    continue;
                }
                const float wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        } else {
            for (const OffsetF& o : offsets) {
                const int nx = xi + o.dx;
                const int ny = yi + o.dy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }
                const int idx = ny * width + nx;
                const int pol_nb = static_cast<int>(last_pol[static_cast<std::size_t>(idx)]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint16_t ts = last_ts[static_cast<std::size_t>(idx)];
                if (ts == 0) {
                    continue;
                }
                const std::uint16_t dt_q = static_cast<std::uint16_t>(ti_q - ts);
                if (dt_q == 0 || dt_q > tau_q_u) {
                    continue;
                }
                const float base_time = 1.0f - static_cast<float>(dt_q) * inv_tau_q;
                if (base_time <= 0.0f) {
                    continue;
                }
                const float wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        }

        const int prev_pol0 = static_cast<int>(last_pol[static_cast<std::size_t>(idx0)]);
        const std::uint16_t ts0 = last_ts[static_cast<std::size_t>(idx0)];
        std::uint16_t dt0 = tau_q_u;
        if (ts0 != 0) {
            dt0 = static_cast<std::uint16_t>(ti_q - ts0);
            if (dt0 > tau_q_u) {
                dt0 = tau_q_u;
            }
        }

        float mix = 0.0f;
        const float denom_mix = raw_same + raw_opp;
        if (denom_mix > 0.0f) {
            mix = raw_opp / (denom_mix + eps);
            mix = std::min(1.0f, std::max(0.0f, mix));
        }
        mstate += (mix - mstate) * inv_N;
        mstate = std::min(1.0f, std::max(0.0f, mstate));
        float alpha_eff = 1.0f - mstate;
        alpha_eff = std::max(0.0f, alpha_eff) * std::max(0.0f, alpha_eff);

        float u_lite = 1.0f - static_cast<float>(dt0) * inv_tau_q;
        u_lite = std::min(1.0f, std::max(0.0f, u_lite));
        float rhythm_bad = 0.0f;
        float rhythm_good = 0.0f;
        if (prev_pol0 == -pi) {
            rhythm_bad = u_lite;
        } else if (prev_pol0 == pi) {
            rhythm_good = u_lite;
        }
        rstate += (rhythm_bad - rstate) * inv_N;
        rstate = std::min(1.0f, std::max(0.0f, rstate));
        float rhythm_pressure = 0.5f * (rhythm_bad + rstate);
        rhythm_pressure = std::min(1.0f, std::max(0.0f, rhythm_pressure));

        const int cnt_possible = interior ? static_cast<int>(offsets.size()) : ((std::min(width - 1, xi + rr) - std::max(0, xi - rr) + 1) * (std::min(height - 1, yi + rr) - std::max(0, yi - rr) + 1) - 1);
        float sfrac = cnt_possible > 0 ? static_cast<float>(cnt_support) / static_cast<float>(cnt_possible) : 0.0f;
        sfrac = std::min(1.0f, std::max(0.0f, sfrac));
        alpha_eff *= (1.0f - rhythm_pressure / 3.0f);
        alpha_eff = std::max(0.0f, alpha_eff);

        float relief = 1.0f - ks * sfrac;
        relief = std::min(1.0f, std::max(0.25f, relief));
        float mix_gain = 1.0f + km * mix;
        mix_gain = std::min(2.0f, std::max(0.5f, mix_gain));
        float rhythm_scale = 1.0f + 0.5f * rhythm_pressure - 0.25f * rhythm_good;
        rhythm_scale = std::min(1.75f, std::max(0.5f, rhythm_scale));
        float u_eff = u_lite * relief * mix_gain * rhythm_scale;
        u_eff = std::min(1.0f, std::max(0.0f, u_eff));
        b += (u_eff - b) * inv_N;
        b = std::min(1.0f, std::max(0.0f, b));

        const float raw_gated = raw_same + alpha_eff * raw_opp;
        const float base_score = raw_gated / (1.0f + u_eff * u_eff);
        float support_scale = 1.0f + b * sfrac * (1.0f + 0.25f * rhythm_good);
        support_scale = std::min(2.0f, std::max(1.0f, support_scale));
        out[i] = base_score * support_scale;

        last_ts[static_cast<std::size_t>(idx0)] = ti_q;
        last_pol[static_cast<std::size_t>(idx0)] = static_cast<std::int8_t>(pi);
    }
}

static void score_n176_fast24q(
    const Events& ev,
    int s,
    int tau_us,
    double tick_ns,
    bool uniform_space,
    std::vector<float>& out
) {
    const int width = static_cast<int>(ev.width);
    const int height = static_cast<int>(ev.height);
    const int rr = std::max(0, (s - 1) / 2);
    const std::size_t npx = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint16_t> last_low(npx, 0);
    std::vector<std::uint8_t> last_epoch(npx, 0);
    std::vector<std::int8_t> last_pol(npx, 0);
    out.assign(static_cast<std::size_t>(ev.n), 0.0f);

    int tau_ticks = static_cast<int>(std::llround(static_cast<double>(tau_us) * 1000.0 / tick_ns));
    if (tau_ticks <= 0) {
        tau_ticks = 1;
    }
    int shift = 0;
    while ((tau_ticks >> shift) > 60000 && shift < 16) {
        ++shift;
    }
    int tau_q = tau_ticks >> shift;
    if (tau_q <= 0) {
        tau_q = 1;
    }
    const std::uint32_t tau_q_u = static_cast<std::uint32_t>(tau_q);
    const float inv_tau_q = 1.0f / static_cast<float>(tau_q);
    const auto offsets = build_dense_offsets_f(rr, width, uniform_space);

    const float inv_N = 1.0f / 4096.0f;
    const float eps = 1e-6f;
    const float ks = 4.0f / 5.0f;
    const float km = 1.0f / 8.0f;
    float b = 0.65f;
    float mstate = 0.0f;
    float rstate = 0.10f;

    for (std::size_t i = 0; i < static_cast<std::size_t>(ev.n); ++i) {
        const int xi = static_cast<int>(ev.x[i]);
        const int yi = static_cast<int>(ev.y[i]);
        if (xi < 0 || xi >= width || yi < 0 || yi >= height) {
            out[i] = 0.0f;
            continue;
        }
        const std::uint32_t ti_q_full = static_cast<std::uint32_t>(static_cast<std::uint64_t>(ev.t[i]) >> shift);
        const std::uint16_t ti_low = static_cast<std::uint16_t>(ti_q_full);
        const std::uint8_t ti_epoch = static_cast<std::uint8_t>(ti_q_full >> 16);
        const std::uint32_t ti24 = (static_cast<std::uint32_t>(ti_epoch) << 16) | static_cast<std::uint32_t>(ti_low);
        const int pi = static_cast<int>(ev.p[i]) > 0 ? 1 : -1;
        const int idx0 = yi * width + xi;
        const bool interior = xi >= rr && xi < width - rr && yi >= rr && yi < height - rr;

        float raw_same = 0.0f;
        float raw_opp = 0.0f;
        int cnt_support = 0;

        if (interior) {
            for (const OffsetF& o : offsets) {
                const int idx = idx0 + o.delta;
                const std::size_t uidx = static_cast<std::size_t>(idx);
                const int pol_nb = static_cast<int>(last_pol[uidx]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint16_t ts_low = last_low[uidx];
                const std::uint8_t ts_epoch = last_epoch[uidx];
                if (ts_low == 0 && ts_epoch == 0) {
                    continue;
                }
                const std::uint32_t ts24 = (static_cast<std::uint32_t>(ts_epoch) << 16) | static_cast<std::uint32_t>(ts_low);
                const std::uint32_t dt_q = (ti24 - ts24) & 0x00FFFFFFu;
                if (dt_q == 0 || dt_q > tau_q_u) {
                    continue;
                }
                const float base_time = 1.0f - static_cast<float>(dt_q) * inv_tau_q;
                if (base_time <= 0.0f) {
                    continue;
                }
                const float wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        } else {
            for (const OffsetF& o : offsets) {
                const int nx = xi + o.dx;
                const int ny = yi + o.dy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }
                const int idx = ny * width + nx;
                const std::size_t uidx = static_cast<std::size_t>(idx);
                const int pol_nb = static_cast<int>(last_pol[uidx]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint16_t ts_low = last_low[uidx];
                const std::uint8_t ts_epoch = last_epoch[uidx];
                if (ts_low == 0 && ts_epoch == 0) {
                    continue;
                }
                const std::uint32_t ts24 = (static_cast<std::uint32_t>(ts_epoch) << 16) | static_cast<std::uint32_t>(ts_low);
                const std::uint32_t dt_q = (ti24 - ts24) & 0x00FFFFFFu;
                if (dt_q == 0 || dt_q > tau_q_u) {
                    continue;
                }
                const float base_time = 1.0f - static_cast<float>(dt_q) * inv_tau_q;
                if (base_time <= 0.0f) {
                    continue;
                }
                const float wst = base_time * base_time * o.w;
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
            }
        }

        const std::size_t idx0u = static_cast<std::size_t>(idx0);
        const int prev_pol0 = static_cast<int>(last_pol[idx0u]);
        const std::uint16_t ts0_low = last_low[idx0u];
        const std::uint8_t ts0_epoch = last_epoch[idx0u];
        std::uint32_t dt0 = tau_q_u;
        if (!(ts0_low == 0 && ts0_epoch == 0)) {
            const std::uint32_t ts024 = (static_cast<std::uint32_t>(ts0_epoch) << 16) | static_cast<std::uint32_t>(ts0_low);
            dt0 = (ti24 - ts024) & 0x00FFFFFFu;
            if (dt0 > tau_q_u) {
                dt0 = tau_q_u;
            }
        }

        float mix = 0.0f;
        const float denom_mix = raw_same + raw_opp;
        if (denom_mix > 0.0f) {
            mix = raw_opp / (denom_mix + eps);
            mix = std::min(1.0f, std::max(0.0f, mix));
        }
        mstate += (mix - mstate) * inv_N;
        mstate = std::min(1.0f, std::max(0.0f, mstate));
        float alpha_eff = 1.0f - mstate;
        alpha_eff = std::max(0.0f, alpha_eff) * std::max(0.0f, alpha_eff);

        float u_lite = 1.0f - static_cast<float>(dt0) * inv_tau_q;
        u_lite = std::min(1.0f, std::max(0.0f, u_lite));
        float rhythm_bad = 0.0f;
        float rhythm_good = 0.0f;
        if (prev_pol0 == -pi) {
            rhythm_bad = u_lite;
        } else if (prev_pol0 == pi) {
            rhythm_good = u_lite;
        }
        rstate += (rhythm_bad - rstate) * inv_N;
        rstate = std::min(1.0f, std::max(0.0f, rstate));
        float rhythm_pressure = 0.5f * (rhythm_bad + rstate);
        rhythm_pressure = std::min(1.0f, std::max(0.0f, rhythm_pressure));

        const int cnt_possible = interior ? static_cast<int>(offsets.size()) : ((std::min(width - 1, xi + rr) - std::max(0, xi - rr) + 1) * (std::min(height - 1, yi + rr) - std::max(0, yi - rr) + 1) - 1);
        float sfrac = cnt_possible > 0 ? static_cast<float>(cnt_support) / static_cast<float>(cnt_possible) : 0.0f;
        sfrac = std::min(1.0f, std::max(0.0f, sfrac));
        alpha_eff *= (1.0f - rhythm_pressure / 3.0f);
        alpha_eff = std::max(0.0f, alpha_eff);

        float relief = 1.0f - ks * sfrac;
        relief = std::min(1.0f, std::max(0.25f, relief));
        float mix_gain = 1.0f + km * mix;
        mix_gain = std::min(2.0f, std::max(0.5f, mix_gain));
        float rhythm_scale = 1.0f + 0.5f * rhythm_pressure - 0.25f * rhythm_good;
        rhythm_scale = std::min(1.75f, std::max(0.5f, rhythm_scale));
        float u_eff = u_lite * relief * mix_gain * rhythm_scale;
        u_eff = std::min(1.0f, std::max(0.0f, u_eff));
        b += (u_eff - b) * inv_N;
        b = std::min(1.0f, std::max(0.0f, b));

        const float raw_gated = raw_same + alpha_eff * raw_opp;
        const float base_score = raw_gated / (1.0f + u_eff * u_eff);
        float support_scale = 1.0f + b * sfrac * (1.0f + 0.25f * rhythm_good);
        support_scale = std::min(2.0f, std::max(1.0f, support_scale));
        out[i] = base_score * support_scale;

        last_low[idx0u] = ti_low;
        last_epoch[idx0u] = ti_epoch;
        last_pol[idx0u] = static_cast<std::int8_t>(pi);
    }
}

static void score_n176(
    const Events& ev,
    int s,
    int tau_us,
    double tick_ns,
    bool uniform_space,
    int ema_tau_us,
    double io_gain,
    std::vector<float>& out
) {
    const int width = static_cast<int>(ev.width);
    const int height = static_cast<int>(ev.height);
    const int rr = std::max(0, (s - 1) / 2);
    const std::size_t npx = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint64_t> last_ts(npx, 0);
    std::vector<std::int8_t> last_pol(npx, 0);
    out.assign(static_cast<std::size_t>(ev.n), 0.0f);

    int tau_ticks = static_cast<int>(std::llround(static_cast<double>(tau_us) * 1000.0 / tick_ns));
    if (tau_ticks <= 0) {
        tau_ticks = 1;
    }
    int ema_tau_ticks = 0;
    if (ema_tau_us > 0) {
        ema_tau_ticks = static_cast<int>(std::llround(static_cast<double>(ema_tau_us) * 1000.0 / tick_ns));
        if (ema_tau_ticks <= 0) {
            ema_tau_ticks = 1;
        }
    }
    io_gain = std::min(0.5, std::max(0.0, io_gain));
    const double inv_tau = 1.0 / static_cast<double>(tau_ticks);
    const double sigma = 2.8;
    const double inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
    const int max_d2 = 2 * rr * rr;
    std::vector<double> space(static_cast<std::size_t>(max_d2 + 1), 1.0);
    if (!uniform_space) {
        for (int d2 = 0; d2 <= max_d2; ++d2) {
            space[static_cast<std::size_t>(d2)] = std::exp(-static_cast<double>(d2) * inv_2sig2);
        }
    }

    const double N = 4096.0;
    const double eps = 1e-6;
    const double ks = 4.0 / 5.0;
    const double km = 1.0 / 8.0;
    double b = 0.65;
    double mstate = 0.0;
    double rstate = 0.10;
    std::uint64_t prev_stream_t = 0;

    for (std::size_t i = 0; i < static_cast<std::size_t>(ev.n); ++i) {
        const int xi = static_cast<int>(ev.x[i]);
        const int yi = static_cast<int>(ev.y[i]);
        if (xi < 0 || xi >= width || yi < 0 || yi >= height) {
            out[i] = 0.0f;
            continue;
        }
        const std::uint64_t ti_u = ev.t[i];
        const int pi = static_cast<int>(ev.p[i]) > 0 ? 1 : -1;
        const int idx0 = yi * width + xi;
        double ema_alpha = 1.0 / N;
        if (ema_tau_ticks > 0 && prev_stream_t > 0) {
            const std::uint64_t dt_stream = ti_u >= prev_stream_t ? ti_u - prev_stream_t : prev_stream_t - ti_u;
            ema_alpha = static_cast<double>(dt_stream) / static_cast<double>(ema_tau_ticks);
            ema_alpha = std::min(1.0, std::max(1.0 / 65536.0, ema_alpha));
        }
        prev_stream_t = ti_u;
        const int x0 = std::max(0, xi - rr);
        const int x1 = std::min(width - 1, xi + rr);
        const int y0 = std::max(0, yi - rr);
        const int y1 = std::min(height - 1, yi + rr);

        double raw_same = 0.0;
        double raw_opp = 0.0;
        double e_inner = 0.0;
        double e_outer = 0.0;
        int cnt_support = 0;
        for (int ny = y0; ny <= y1; ++ny) {
            const int dy = ny - yi;
            for (int nx = x0; nx <= x1; ++nx) {
                const int dx = nx - xi;
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int d2 = dx * dx + dy * dy;
                if (d2 > rr * rr * 2) {
                    continue;
                }
                const int idx = ny * width + nx;
                const int pol_nb = static_cast<int>(last_pol[static_cast<std::size_t>(idx)]);
                if (pol_nb != pi && pol_nb != -pi) {
                    continue;
                }
                const std::uint64_t ts = last_ts[static_cast<std::size_t>(idx)];
                if (ts == 0 || ti_u <= ts) {
                    continue;
                }
                std::uint64_t dt_u = ti_u - ts;
                if (dt_u > static_cast<std::uint64_t>(tau_ticks)) {
                    continue;
                }
                const double base_time = 1.0 - static_cast<double>(dt_u) * inv_tau;
                if (base_time <= 0.0) {
                    continue;
                }
                const double wst = base_time * base_time * space[static_cast<std::size_t>(d2)];
                if (pol_nb == pi) {
                    raw_same += wst;
                    cnt_support += 1;
                } else {
                    raw_opp += wst;
                }
                if (io_gain > 0.0) {
                    if (d2 <= 4) {
                        e_inner += wst;
                    } else {
                        e_outer += wst;
                    }
                }
            }
        }

        const int prev_pol0 = static_cast<int>(last_pol[static_cast<std::size_t>(idx0)]);
        const std::uint64_t ts0 = last_ts[static_cast<std::size_t>(idx0)];
        std::uint64_t dt0 = static_cast<std::uint64_t>(tau_ticks);
        if (ts0 != 0) {
            dt0 = ti_u >= ts0 ? ti_u - ts0 : ts0 - ti_u;
            if (dt0 > static_cast<std::uint64_t>(tau_ticks)) {
                dt0 = static_cast<std::uint64_t>(tau_ticks);
            }
        }

        double mix = 0.0;
        const double denom_mix = raw_same + raw_opp;
        if (denom_mix > 0.0) {
            mix = raw_opp / (denom_mix + eps);
            mix = std::min(1.0, std::max(0.0, mix));
        }
        mstate += (mix - mstate) * ema_alpha;
        mstate = std::min(1.0, std::max(0.0, mstate));
        double alpha_eff = 1.0 - mstate;
        alpha_eff = std::max(0.0, alpha_eff) * std::max(0.0, alpha_eff);

        double u_lite = 1.0 - static_cast<double>(dt0) * inv_tau;
        u_lite = std::min(1.0, std::max(0.0, u_lite));
        double rhythm_bad = 0.0;
        double rhythm_good = 0.0;
        if (prev_pol0 == -pi) {
            rhythm_bad = u_lite;
        } else if (prev_pol0 == pi) {
            rhythm_good = u_lite;
        }
        rstate += (rhythm_bad - rstate) * ema_alpha;
        rstate = std::min(1.0, std::max(0.0, rstate));
        double rhythm_pressure = 0.5 * (rhythm_bad + rstate);
        rhythm_pressure = std::min(1.0, std::max(0.0, rhythm_pressure));

        const int cnt_possible = (x1 - x0 + 1) * (y1 - y0 + 1) - 1;
        double sfrac = cnt_possible > 0 ? static_cast<double>(cnt_support) / static_cast<double>(cnt_possible) : 0.0;
        sfrac = std::min(1.0, std::max(0.0, sfrac));
        alpha_eff *= (1.0 - rhythm_pressure / 3.0);
        alpha_eff = std::max(0.0, alpha_eff);

        double relief = 1.0 - ks * sfrac;
        relief = std::min(1.0, std::max(0.25, relief));
        double mix_gain = 1.0 + km * mix;
        mix_gain = std::min(2.0, std::max(0.5, mix_gain));
        double rhythm_scale = 1.0 + 0.5 * rhythm_pressure - 0.25 * rhythm_good;
        rhythm_scale = std::min(1.75, std::max(0.5, rhythm_scale));
        double u_eff = u_lite * relief * mix_gain * rhythm_scale;
        u_eff = std::min(1.0, std::max(0.0, u_eff));
        b += (u_eff - b) * ema_alpha;
        b = std::min(1.0, std::max(0.0, b));

        const double raw_gated = raw_same + alpha_eff * raw_opp;
        const double base_score = raw_gated / (1.0 + u_eff * u_eff);
        double support_scale = 1.0 + b * sfrac * (1.0 + 0.25 * rhythm_good);
        support_scale = std::min(2.0, std::max(1.0, support_scale));
        double io_scale = 1.0;
        if (io_gain > 0.0) {
            const double e_total = e_inner + e_outer;
            double balance = 0.5;
            if (e_total > eps) {
                const double outer_ratio = std::min(1.0, std::max(0.0, e_outer / (e_total + eps)));
                balance = 4.0 * outer_ratio * (1.0 - outer_ratio);
            }
            io_scale = 1.0 + io_gain * (balance - 0.5);
            io_scale = std::min(1.25, std::max(0.75, io_scale));
        }
        out[i] = static_cast<float>(base_score * support_scale * io_scale);

        last_ts[static_cast<std::size_t>(idx0)] = ti_u;
        last_pol[static_cast<std::size_t>(idx0)] = static_cast<std::int8_t>(pi);
    }
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "usage: ebf_n176_bench <in.bin> <scores.bin> <s> <tau_us> <tick_ns> <repeats> <space_mode> [ema_tau_us] [io_gain]\n"
                  << "space_mode: 0=gaussian original, 1=uniform original, 2=gaussian fast, 3=gaussian fast32, 4=gaussian fast16q, 5=gaussian fast24q\n";
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string scores_path = argv[2];
    const int s = std::stoi(argv[3]);
    const int tau_us = std::stoi(argv[4]);
    const double tick_ns = std::stod(argv[5]);
    const int repeats = std::max(1, std::stoi(argv[6]));
    const int space_mode = std::stoi(argv[7]);
    const bool fast_mode = space_mode == 2;
    const bool fast32_mode = space_mode == 3;
    const bool fast16q_mode = space_mode == 4;
    const bool fast24q_mode = space_mode == 5;
    const bool uniform_space = space_mode == 1;
    const int ema_tau_us = argc >= 9 ? std::stoi(argv[8]) : 0;
    const double io_gain = argc >= 10 ? std::stod(argv[9]) : 0.0;

    try {
        Events ev = read_events(in_path);
        std::vector<float> scores;
        if (fast24q_mode) {
            score_n176_fast24q(ev, s, tau_us, tick_ns, false, scores);
        } else if (fast16q_mode) {
            score_n176_fast16q(ev, s, tau_us, tick_ns, false, scores);
        } else if (fast32_mode) {
            score_n176_fast32(ev, s, tau_us, tick_ns, false, scores);
        } else if (fast_mode) {
            score_n176_fast(ev, s, tau_us, tick_ns, false, scores);
        } else {
            score_n176(ev, s, tau_us, tick_ns, uniform_space, ema_tau_us, io_gain, scores);
        }
        double best = 1e100;
        double sum = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            if (fast24q_mode) {
                score_n176_fast24q(ev, s, tau_us, tick_ns, false, scores);
            } else if (fast16q_mode) {
                score_n176_fast16q(ev, s, tau_us, tick_ns, false, scores);
            } else if (fast32_mode) {
                score_n176_fast32(ev, s, tau_us, tick_ns, false, scores);
            } else if (fast_mode) {
                score_n176_fast(ev, s, tau_us, tick_ns, false, scores);
            } else {
                score_n176(ev, s, tau_us, tick_ns, uniform_space, ema_tau_us, io_gain, scores);
            }
            auto t1 = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            best = std::min(best, sec);
            sum += sec;
        }
        write_scores(scores_path, scores);
        const double mean = sum / static_cast<double>(repeats);
        const double eps_best = static_cast<double>(ev.n) / best;
        const double eps_mean = static_cast<double>(ev.n) / mean;
        std::cout << "events=" << ev.n
                  << " s=" << s
                  << " tau_us=" << tau_us
                  << " uniform_space=" << (uniform_space ? 1 : 0)
                  << " fast_mode=" << (fast_mode ? 1 : 0)
                  << " fast32_mode=" << (fast32_mode ? 1 : 0)
                  << " fast16q_mode=" << (fast16q_mode ? 1 : 0)
                  << " fast24q_mode=" << (fast24q_mode ? 1 : 0)
                  << " ema_tau_us=" << ema_tau_us
                  << " io_gain=" << io_gain
                  << " best_seconds=" << best
                  << " mean_seconds=" << mean
                  << " best_events_per_s=" << eps_best
                  << " mean_events_per_s=" << eps_mean
                  << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
