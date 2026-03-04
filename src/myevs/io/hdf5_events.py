from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from ..events import EventBatch, EventStreamMeta


@dataclass(frozen=True)
class Hdf5Info:
    meta: EventStreamMeta
    source_kind: str
    source_time_unit: str


def _import_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except Exception as e:
        raise RuntimeError(
            "HDF5 support requires h5py. Install it in your environment (e.g. conda install -c conda-forge h5py)."
        ) from e


def _to_text(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="ignore")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _to_text(v.item())
    return str(v)


def _parse_geometry_text(geometry: str) -> tuple[int, int] | None:
    m = re.search(r"(\d+)\s*[xX]\s*(\d+)", str(geometry))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _infer_geometry(file_obj, *, width: int | None, height: int | None) -> tuple[int, int]:
    w_arg = int(width) if width is not None else None
    h_arg = int(height) if height is not None else None

    w = None
    h = None

    attrs = file_obj.attrs
    if w is None:
        for k in ("width", "sensor_width"):
            if k in attrs:
                try:
                    w = int(_to_text(attrs[k]))
                    break
                except Exception:
                    pass
    if h is None:
        for k in ("height", "sensor_height"):
            if k in attrs:
                try:
                    h = int(_to_text(attrs[k]))
                    break
                except Exception:
                    pass

    if (w is None or h is None) and "geometry" in attrs:
        gh = _parse_geometry_text(_to_text(attrs["geometry"]))
        if gh is not None:
            gw, ghh = gh
            if w is None:
                w = gw
            if h is None:
                h = ghh

    if w is None and w_arg is not None:
        w = w_arg
    if h is None and h_arg is not None:
        h = h_arg

    if w is None or h is None or w <= 0 or h <= 0:
        raise ValueError(
            "Unable to infer width/height from HDF5 metadata. Please pass --width and --height explicitly."
        )
    return int(w), int(h)


def _normalize_time_unit(v: str) -> str:
    u = str(v).strip().lower()
    aliases = {
        "tick": "tick",
        "ticks": "tick",
        "us": "us",
        "microsecond": "us",
        "microseconds": "us",
        "ns": "ns",
        "nanosecond": "ns",
        "nanoseconds": "ns",
        "ms": "ms",
        "millisecond": "ms",
        "milliseconds": "ms",
    }
    return aliases.get(u, u)


def _detect_time_unit(file_obj, *, source_kind: str) -> str:
    attrs = file_obj.attrs
    for k in ("time_unit", "timestamp_unit", "t_unit", "ts_unit"):
        if k in attrs:
            return _normalize_time_unit(_to_text(attrs[k]))

    # 注释：OpenEB 的 CD 时间戳默认是微秒。
    if source_kind == "openeb_cd":
        return "us"
    return "tick"


def _convert_to_tick(t_raw: np.ndarray, *, src_unit: str, tick_ns: float) -> np.ndarray:
    t_i64 = np.asarray(t_raw, dtype=np.int64)
    t_i64 = np.maximum(t_i64, 0)

    if src_unit == "tick":
        return t_i64.astype(np.uint64, copy=False)
    if src_unit == "us":
        scale = 1000.0 / float(tick_ns)
        return np.rint(t_i64.astype(np.float64) * scale).astype(np.uint64)
    if src_unit == "ns":
        scale = 1.0 / float(tick_ns)
        return np.rint(t_i64.astype(np.float64) * scale).astype(np.uint64)
    if src_unit == "ms":
        scale = 1_000_000.0 / float(tick_ns)
        return np.rint(t_i64.astype(np.float64) * scale).astype(np.uint64)

    raise ValueError(f"Unsupported HDF5 timestamp unit: {src_unit}")


def _select_cd_dataset(file_obj):
    if "/CD/events" in file_obj:
        return file_obj["/CD/events"], "openeb_cd"
    if "/events" in file_obj:
        return file_obj["/events"], "generic_events"
    raise ValueError("Unsupported HDF5 layout: expected /CD/events or /events dataset")


def _is_ecf_plugin_filename(filename: str) -> bool:
    name = str(filename).lower()
    stem, ext = os.path.splitext(name)

    # Typical names:
    # - Windows: H5Zecf.dll / hdf5_ecf_codec.dll
    # - Linux:   libh5zecf.so / libhdf5_ecf_codec.so
    # - macOS:   libh5zecf.dylib / libhdf5_ecf_codec.dylib
    if ("h5zecf" not in stem) and ("ecf_codec" not in stem):
        return False

    if sys.platform.startswith("win"):
        return ext == ".dll"
    if sys.platform == "darwin":
        return ext in (".dylib", ".so")
    return ext == ".so"


def _contains_ecf_plugin(plugin_dir: str) -> bool:
    if not plugin_dir or (not os.path.isdir(plugin_dir)):
        return False
    try:
        names = [n.lower() for n in os.listdir(plugin_dir)]
    except OSError:
        return False
    for n in names:
        if _is_ecf_plugin_filename(n):
            return True
    return False


def _iter_env_plugin_dirs() -> Iterator[str]:
    seen: set[str] = set()

    # Respect already configured plugin search path if present.
    for raw in str(os.environ.get("HDF5_PLUGIN_PATH", "")).split(os.pathsep):
        p = os.path.abspath(raw.strip())
        if not p:
            continue
        k = os.path.normcase(p)
        if k in seen:
            continue
        seen.add(k)
        yield p

    prefixes = [
        os.environ.get("CONDA_PREFIX"),
        os.environ.get("VIRTUAL_ENV"),
        sys.prefix,
    ]
    rel_candidates = (
        os.path.join("Library", "hdf5", "plugin"),       # Windows conda
        os.path.join("lib", "hdf5", "plugin"),           # Linux/macOS common
        os.path.join("lib", "hdf5", "plugins"),          # Some HDF5 builds
        os.path.join("lib64", "hdf5", "plugin"),         # Linux lib64
        os.path.join("lib64", "hdf5", "plugins"),
    )

    for prefix in prefixes:
        if not prefix:
            continue
        for rel in rel_candidates:
            p = os.path.join(str(prefix), rel)
            k = os.path.normcase(os.path.abspath(p))
            if k in seen:
                continue
            seen.add(k)
            yield p


def _iter_project_plugin_dirs() -> Iterator[str]:
    seen: set[str] = set()

    env_hint = os.environ.get("MYEVS_HDF5_PLUGIN_DIR")
    if env_hint:
        p = os.path.abspath(env_hint)
        k = os.path.normcase(p)
        if k not in seen:
            seen.add(k)
            yield p

    # hdf5_events.py -> io -> myevs -> src -> project_root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    rel_candidates = (
        os.path.join("Library", "hdf5", "plugin"),
        os.path.join("library", "hdf5", "plugin"),
        os.path.join("openeb_Library", "hdf5", "plugin"),
        os.path.join("openeb_library", "hdf5", "plugin"),
    )

    for base in (os.getcwd(), project_root):
        for rel in rel_candidates:
            p = os.path.abspath(os.path.join(base, rel))
            k = os.path.normcase(p)
            if k in seen:
                continue
            seen.add(k)
            yield p


def _copy_plugin_dir(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dst_dir, name)
        need_copy = True
        if os.path.exists(dst):
            try:
                need_copy = (
                    os.path.getsize(src) != os.path.getsize(dst)
                    or int(os.path.getmtime(src)) > int(os.path.getmtime(dst))
                )
            except OSError:
                need_copy = True
        if need_copy:
            shutil.copy2(src, dst)


def _bootstrap_project_plugin_into_env() -> str | None:
    src_dir = None
    for p in _iter_project_plugin_dirs():
        if _contains_ecf_plugin(p):
            src_dir = p
            break

    if src_dir is None:
        return None

    src_norm = os.path.normcase(os.path.abspath(src_dir))
    for env_dir in _iter_env_plugin_dirs():
        env_norm = os.path.normcase(os.path.abspath(env_dir))
        try:
            if env_norm != src_norm:
                _copy_plugin_dir(src_dir, env_dir)
            if _contains_ecf_plugin(env_dir):
                return env_dir
        except OSError:
            continue

    # Fallback: even if we cannot write env dir, use project plugin dir directly.
    return src_dir


def _resolve_plugin_path(user_path: str | None) -> str | None:
    if user_path:
        return str(user_path)

    if os.environ.get("HDF5_PLUGIN_PATH"):
        return None

    bootstrapped = _bootstrap_project_plugin_into_env()
    if bootstrapped and _contains_ecf_plugin(bootstrapped):
        return bootstrapped

    for c in _iter_env_plugin_dirs():
        if _contains_ecf_plugin(c):
            return c

    for c in _iter_project_plugin_dirs():
        if _contains_ecf_plugin(c):
            return c
    return None


def read_hdf5(
    path: str,
    *,
    width: int | None = None,
    height: int | None = None,
    batch_events: int = 1_000_000,
    tick_ns: float = 12.5,
    hdf5_plugin_path: str | None = None,
) -> Tuple[Hdf5Info, Iterator[EventBatch]]:
    """Read events from HDF5 and return myEVS batches.

    Supported layouts:
    - OpenEB/Prophesee: `/CD/events` structured dataset with fields x,y,p,t (timestamp in us)
    - Generic myEVS: `/events` structured dataset with fields x,y,p,t

    Returned timestamps are normalized to myEVS ticks for consistent downstream denoise/view/stats behavior.
    """

    resolved_plugin_path = _resolve_plugin_path(hdf5_plugin_path)
    if resolved_plugin_path:
        # 注释：OpenEB HDF5 的 CD 数据常用 ECF 压缩，需要 HDF5 插件目录。
        os.environ["HDF5_PLUGIN_PATH"] = str(resolved_plugin_path)

    h5py = _import_h5py()
    f = h5py.File(path, "r")
    try:
        ds, source_kind = _select_cd_dataset(f)
        if ds.dtype.names is None:
            raise ValueError("HDF5 events dataset must be a structured dataset with fields x,y,p,t")

        fields = {str(name).lower(): str(name) for name in ds.dtype.names}
        for required in ("x", "y", "p", "t"):
            if required not in fields:
                raise ValueError(f"HDF5 events dataset missing required field: {required}")

        w, h = _infer_geometry(f, width=width, height=height)
        src_unit = _detect_time_unit(f, source_kind=source_kind)

        meta = EventStreamMeta(width=w, height=h, time_unit="tick")
        info = Hdf5Info(meta=meta, source_kind=source_kind, source_time_unit=src_unit)

        field_x = fields["x"]
        field_y = fields["y"]
        field_p = fields["p"]
        field_t = fields["t"]

        def gen() -> Iterator[EventBatch]:
            try:
                total = int(ds.shape[0])
                step = max(1, int(batch_events))
                for start in range(0, total, step):
                    end = min(total, start + step)
                    try:
                        part = ds[start:end]
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to read HDF5 event chunk. If this is an OpenEB compressed HDF5 file, "
                            "ensure HDF5 ECF plugin is discoverable (set HDF5_PLUGIN_PATH or use --hdf5-plugin-path)."
                        ) from e

                    x = np.asarray(part[field_x], dtype=np.uint16)
                    y = np.asarray(part[field_y], dtype=np.uint16)
                    p_raw = np.asarray(part[field_p])
                    p = np.where(p_raw > 0, 1, -1).astype(np.int8)
                    t = _convert_to_tick(np.asarray(part[field_t]), src_unit=src_unit, tick_ns=float(tick_ns))

                    if t.shape[0] > 0:
                        yield EventBatch(t=t, x=x, y=y, p=p)
            finally:
                f.close()

        return info, gen()
    except Exception:
        f.close()
        raise


def write_hdf5(
    path: str,
    meta: EventStreamMeta,
    batches: Iterator[EventBatch],
    *,
    tick_ns: float = 12.5,
    chunk_events: int = 262_144,
) -> None:
    """Write event stream into myEVS HDF5.

    Layout intentionally stays close to OpenEB (`/CD/events`) to ease interoperability,
    while storing timestamps directly in myEVS ticks.
    """

    h5py = _import_h5py()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    cd_dtype = np.dtype(
        [
            ("x", np.uint16),
            ("y", np.uint16),
            ("p", np.int16),
            ("t", np.int64),
        ]
    )
    idx_dtype = np.dtype(
        [
            ("id", np.uint64),
            ("ts", np.int64),
        ]
    )
    ext_event_dtype = np.dtype(
        [
            ("p", np.int16),
            ("t", np.int64),
            ("id", np.int16),
        ]
    )

    with h5py.File(path, "w") as f:
        # 注释：根属性用于标记 myEVS 自身时间基准（tick），避免与 OpenEB(us) 语义混淆。
        f.attrs["format"] = "myevs_hdf5_v1"
        f.attrs["geometry"] = f"{int(meta.width)}x{int(meta.height)}"
        f.attrs["time_unit"] = "tick"
        f.attrs["tick_ns"] = str(float(tick_ns))

        g_cd = f.require_group("CD")
        g_ext = f.require_group("EXT_TRIGGER")

        ds = g_cd.create_dataset(
            "events",
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1, int(chunk_events)),),
            dtype=cd_dtype,
        )

        # 注释：保留 indexes 数据集占位，便于与 OpenEB 目录结构一致。
        ds_idx = g_cd.create_dataset(
            "indexes",
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1, int(min(chunk_events, 4096))),),
            dtype=idx_dtype,
        )
        ds_idx.attrs["offset"] = "0"

        g_ext.create_dataset("events", shape=(0,), maxshape=(None,), dtype=ext_event_dtype)
        ds_ext_idx = g_ext.create_dataset("indexes", shape=(0,), maxshape=(None,), dtype=idx_dtype)
        ds_ext_idx.attrs["offset"] = "0"

        offset = 0
        for b in batches:
            if len(b) == 0:
                continue

            n = int(len(b))
            out = np.empty((n,), dtype=cd_dtype)
            out["x"] = np.asarray(b.x, dtype=np.uint16)
            out["y"] = np.asarray(b.y, dtype=np.uint16)
            out["p"] = (np.asarray(b.p, dtype=np.int8) > 0).astype(np.int16)
            out["t"] = np.asarray(b.t, dtype=np.uint64).astype(np.int64)

            ds.resize((offset + n,))
            ds[offset : offset + n] = out
            offset += n
