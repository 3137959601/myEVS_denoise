#include "native_emlb.hpp"

PYBIND11_MODULE(_native_emlb, m) {
    m.doc() = "Native C++ implementations of E-MLB-derived denoise filters for myEVS";

    py::class_<myevs_native_emlb::StcNative>(m, "StcNative")
        .def(py::init<int, int, uint64_t, int, int, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("duration_ticks"), py::arg("radius"), py::arg("threshold"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::StcNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::StcNative::reset);

    py::class_<myevs_native_emlb::StcfOriginalNative>(m, "StcfOriginalNative")
        .def(py::init<int, int, uint64_t, int, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("tau_ticks"), py::arg("k"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::StcfOriginalNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::StcfOriginalNative::reset);

    py::class_<myevs_native_emlb::BafNative>(m, "BafNative")
        .def(py::init<int, int, uint64_t, int, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("duration_ticks"), py::arg("radius"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::BafNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::BafNative::reset);

    py::class_<myevs_native_emlb::EbfNative>(m, "EbfNative")
        .def(py::init<int, int, uint64_t, int, double, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("tau_ticks"), py::arg("radius"), py::arg("threshold"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::EbfNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::EbfNative::reset);

    py::class_<myevs_native_emlb::KNoiseNative>(m, "KNoiseNative")
        .def(py::init<int, int, uint64_t, int, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("duration_ticks"), py::arg("threshold"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::KNoiseNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::KNoiseNative::reset);

    py::class_<myevs_native_emlb::YNoiseNative>(m, "YNoiseNative")
        .def(py::init<int, int, uint64_t, int, int, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("duration_ticks"), py::arg("radius"), py::arg("threshold"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::YNoiseNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::YNoiseNative::reset);

    py::class_<myevs_native_emlb::TimeSurfaceNative>(m, "TimeSurfaceNative")
        .def(py::init<int, int, uint64_t, int, double, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("decay_ticks"), py::arg("radius"), py::arg("threshold"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::TimeSurfaceNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::TimeSurfaceNative::reset);

    py::class_<myevs_native_emlb::EventFlowNative>(m, "EventFlowNative")
        .def(py::init<int, int, uint64_t, int, double, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("duration_ticks"), py::arg("radius"), py::arg("threshold"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::EventFlowNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::EventFlowNative::reset);

    py::class_<myevs_native_emlb::N149Native>(m, "N149Native")
        .def(py::init<int, int, uint64_t, int, double, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("tau_ticks"), py::arg("radius"), py::arg("threshold"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::N149Native::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("score_batch", &myevs_native_emlb::N149Native::score_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::N149Native::reset);

    py::class_<myevs_native_emlb::N149RtlFixedNative>(m, "N149RtlFixedNative")
        .def(py::init<int, int, uint64_t, int, uint64_t, bool, bool>(),
             py::arg("width"), py::arg("height"), py::arg("tau_ticks"), py::arg("radius"), py::arg("thr_ticks"),
             py::arg("show_on") = true, py::arg("show_off") = true)
        .def("accept_batch", &myevs_native_emlb::N149RtlFixedNative::accept_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("score_batch", &myevs_native_emlb::N149RtlFixedNative::score_batch,
             py::arg("t"), py::arg("x"), py::arg("y"), py::arg("p"))
        .def("reset", &myevs_native_emlb::N149RtlFixedNative::reset);

}
