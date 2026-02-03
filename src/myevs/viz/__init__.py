from __future__ import annotations

from typing import Iterator, Literal

from ..events import EventBatch, EventStreamMeta

__all__ = ["view_stream"]


def view_stream(
	meta: EventStreamMeta,
	batches: Iterator[EventBatch],
	*,
	mode: Literal["fps", "events"] = "fps",
	fps: float = 60.0,
	events_per_frame: int = 200_000,
	tick_us: float = 1.0,
	color: Literal["gray", "onoff", "onoff_rb"] = "onoff",
	scheme_id: int = 0,
	window_name: str = "myEVS",
	key_delay_ms: int = 1,
	raw_step: int = 10,
	deadzone: int = 3,
	binary: bool = False,
	hold: bool = True,
	show_on: bool = True,
	show_off: bool = True,
	realtime: bool = False,
	out_video: str | None = None,
	video_fps: float | None = None,
	no_gui: bool = False,
) -> None:
	from .viewer import view_stream as _impl

	return _impl(
		meta,
		batches,
		mode=mode,
		fps=fps,
		events_per_frame=events_per_frame,
		tick_us=tick_us,
		color=color,
		scheme_id=scheme_id,
		window_name=window_name,
		key_delay_ms=key_delay_ms,
		raw_step=raw_step,
		deadzone=deadzone,
		binary=binary,
		hold=hold,
		show_on=show_on,
		show_off=show_off,
		realtime=realtime,
		out_video=out_video,
		video_fps=video_fps,
		no_gui=no_gui,
	)
