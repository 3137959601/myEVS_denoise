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
        std::cerr << "usage: ebf_n176_bench <in.bin> <scores.bin> <s> <tau_us> <tick_ns> <repeats> <uniform0or1> [ema_tau_us] [io_gain]\n";
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string scores_path = argv[2];
    const int s = std::stoi(argv[3]);
    const int tau_us = std::stoi(argv[4]);
    const double tick_ns = std::stod(argv[5]);
    const int repeats = std::max(1, std::stoi(argv[6]));
    const bool uniform_space = std::stoi(argv[7]) != 0;
    const int ema_tau_us = argc >= 9 ? std::stoi(argv[8]) : 0;
    const double io_gain = argc >= 10 ? std::stod(argv[9]) : 0.0;

    try {
        Events ev = read_events(in_path);
        std::vector<float> scores;
        score_n176(ev, s, tau_us, tick_ns, uniform_space, ema_tau_us, io_gain, scores);
        double best = 1e100;
        double sum = 0.0;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            score_n176(ev, s, tau_us, tick_ns, uniform_space, ema_tau_us, io_gain, scores);
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
