# Media-Clock Video Sync Design

## Goal

Keep the source video smooth at its browser-native frame rate while making the
annotated Product cam show a frame from the same source timestamp, including
after playback is paused or seeked.

## Problem

`st.video` owns an HTML5 video element in the browser, but the existing
Product cam loop estimates its source position from `time.perf_counter()` in
the Docker container. Video load/autoplay delay, browser scheduling, and
seeking make those clocks independent. A lag bound applied to the server clock
therefore cannot guarantee that the two panes show the same moment.

## Architecture

Add a small Streamlit custom component whose frontend owns the source HTML5
`<video>` element. It reports a value containing `currentTime`, play state,
and a monotonic revision whenever the browser video time changes materially.
The dashboard uses that media time as the source of truth for Product cam.

The Product cam worker maps the requested source time to a frame index. It
discards decoded frames before that index with `grab()`, decodes the requested
frame, runs normal inference, and renders it with a visible `Source: MM:SS.xx`
caption. It never advances while playback is paused. On a forward seek it
jumps to the requested source frame; on a backward seek it reopens the capture
and seeks to the requested frame before inference.

## Component Contract

The Python wrapper accepts a video URL/path and returns either `None` before
the component is ready or a mapping:

```python
{
    "time_s": 11.24,
    "is_playing": True,
    "revision": 17,
}
```

The frontend sends updates on `loadedmetadata`, `play`, `pause`, `seeked`, and
at most once every 100 ms while playing. The `revision` increments for every
event so the backend can distinguish a seek from a repeated time report.

## Product Stream Contract

`process_product_stream` is replaced by a single-step operation that receives
the latest browser media state. It returns the rendered frame timestamp and
the next capture position. The dashboard only invokes it for a new media
revision and only while the source is playing. For the same video timestamp,
the raw and annotated panes therefore refer to the same source moment, with at
most one frame of rounding error.

## Error Handling

If the component has not supplied a timestamp, the dashboard shows
“Source video — loading…” and does not process frames. Invalid or negative
timestamps are ignored. If OpenCV cannot seek/read the requested source frame,
the Product cam shows a concise error while the source player remains usable.

## Testing

Unit tests cover conversion from media seconds to a clamped frame index,
forward dropping, backward seek/reopen behavior, no processing while paused,
and a timestamp overlay/format helper. Component packaging is verified by a
test that requires its built HTML entrypoint. Docker tests exercise the same
Python unit tests; manual browser verification confirms play, pause, and seek
alignment using the visible source timestamps.

## Scope

This changes only Upload Video playback. Webcam remains a true live server
stream and continues to use its existing path. Product inference FPS remains
measured and displayed; it is not presented as source FPS.
