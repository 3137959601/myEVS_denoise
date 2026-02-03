from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from ..events import EventBatch, EventStreamMeta


@dataclass(frozen=True)
class UsbRawInfo:
    meta: EventStreamMeta
    endian: str = "little"


def _popcount8(v: int) -> int:
    return int(bin(v & 0xFF).count("1"))


def read_usb_raw_evt3(path: str, width: int, height: int, batch_events: int = 1_000_000) -> Tuple[UsbRawInfo, Iterator[EventBatch]]:
    """Parse EVT3-like 32-bit word stream according to your Qt EvtParserWorker::parseWordColumn().

    Word types:
      - W0: type=0xE: {4'he,2'b0,TS[31:24],4'h8,TS[23:12]}
      - W1: type=0x6: {4'h6,TS[11:0],4'h0,2'b0,X[9:0]}
      - Wk: type=0x3 and tag5==0x5: sub=(w>>25)&7, startY=(w>>16)&0x1FF, mask=w&0xFF
           sub==0b100 => ON else OFF

    Timestamp expansion:
      Qt code extends 32-bit TS to 64-bit with epoch on wrap.
    """

    meta = EventStreamMeta(width=width, height=height, time_unit="tick")
    info = UsbRawInfo(meta=meta)

    def gen() -> Iterator[EventBatch]:
        # Parser state (mirrors Qt)
        have_ts_hi_mid = False
        ts_hi8 = 0
        ts_mid12 = 0
        has_col = False
        cur_x = 0
        ts_epoch = 0
        last_ts32 = 0
        have_last_ts = False
        cur_ts64 = 0

        t_buf = []
        x_buf = []
        y_buf = []
        p_buf = []

        with open(path, "rb") as f:
            while True:
                data = f.read(4 * 1024 * 1024)
                if not data:
                    break
                n_words = len(data) // 4
                words = np.frombuffer(data[: n_words * 4], dtype=np.uint32)
                for w in words:
                    w = int(w)
                    wtype = (w >> 28) & 0xF

                    if wtype == 0xE:
                        # W0: [31:28]=E, [15:12]=8 tag (Qt validates this)
                        tag8 = (w >> 12) & 0xF
                        if tag8 != 0x8:
                            have_ts_hi_mid = False
                            continue

                        # TS hi/mid
                        ts_hi8 = (w >> 16) & 0xFF
                        ts_mid12 = (w & 0xFFF)
                        have_ts_hi_mid = True
                        continue

                    if wtype == 0x6:
                        if not have_ts_hi_mid:
                            continue
                        ts_lo12 = (w >> 16) & 0xFFF
                        x = w & 0x3FF

                        # 32-bit timestamp assembly; top 2 bits invalid -> mask to 30-bit in high byte (Qt behavior)
                        ts32 = ((ts_hi8 & 0x3F) << 24) | (ts_mid12 << 12) | ts_lo12
                        have_ts_hi_mid = False

                        # Extend to monotonic 64-bit in case of wrap-around.
                        # Qt only treats it as wrap when the backwards jump is large (avoid false wraps on reordering).
                        if have_last_ts:
                            if ts32 < last_ts32 and (last_ts32 - ts32) > 0x80000000:
                                ts_epoch += (1 << 32)
                        last_ts32 = ts32
                        have_last_ts = True

                        cur_x = x
                        cur_ts64 = ts_epoch + ts32
                        has_col = True
                        continue

                    if wtype == 0x3:
                        tag5 = (w >> 12) & 0xF
                        if tag5 != 0x5 or not has_col:
                            continue
                        sub = (w >> 25) & 0x7
                        is_on = (sub == 0b100)
                        start_y = (w >> 16) & 0x1FF
                        mask = w & 0xFF

                        if cur_x >= width:
                            continue
                        for bit in range(8):
                            if (mask >> bit) & 0x1:
                                yy = start_y + bit
                                if yy >= height:
                                    continue
                                t_buf.append(cur_ts64)
                                x_buf.append(cur_x)
                                y_buf.append(yy)
                                p_buf.append(1 if is_on else -1)
                                if len(t_buf) >= batch_events:
                                    yield EventBatch(
                                        t=np.asarray(t_buf, dtype=np.uint64),
                                        x=np.asarray(x_buf, dtype=np.uint16),
                                        y=np.asarray(y_buf, dtype=np.uint16),
                                        p=np.asarray(p_buf, dtype=np.int8),
                                    )
                                    t_buf.clear(); x_buf.clear(); y_buf.clear(); p_buf.clear()
                        continue

        if t_buf:
            yield EventBatch(
                t=np.asarray(t_buf, dtype=np.uint64),
                x=np.asarray(x_buf, dtype=np.uint16),
                y=np.asarray(y_buf, dtype=np.uint16),
                p=np.asarray(p_buf, dtype=np.int8),
            )

    return info, gen()
