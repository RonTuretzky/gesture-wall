# Gesture Wall — web version (projector-ready)

A browser port of the Stack A pipeline you can **project on a wall and test**.
Everything runs client-side: webcam → MediaPipe Tasks **PoseLandmarker** (in the
browser, WebGL/GPU) → mirror → homography → 1-Euro smoothing → dwell-to-select.
The selection logic (`OneEuroFilter`, `Zone`, `DwellSelector`, `Homography`) is a
direct port of the Python modules, so behaviour matches the desktop app.

**Why the browser?** Camera access in a browser is a one-click in-page prompt and
`localhost` is a secure context — so it sidesteps the macOS native-app camera
permission problem that blocked `python3 run.py --source pose`.

## Run

```bash
./web/serve.sh                 # serves http://localhost:8000
# or:  python3 -m http.server 8000 -d web
```

Open **http://localhost:8000** in Chrome (best WebGL support), then:

1. Click **Start camera** and allow access when the browser asks.
2. Raise a hand above your shoulder → a cursor appears at your wrist.
3. Hold the cursor over a tile ~0.8 s → the ring fills and the tile toggles.
4. Click **Fullscreen** (or press `f`) and project. The control panel auto-hides;
   hover the top edge of the screen to bring it back.

> No camera handy? Click **Mouse test** to drive the exact same pipeline with the
> mouse — move over a tile and hold still.

## Projector calibration

The cursor maps your wrist position in the camera image onto the wall. If the
camera isn't square-on, run **Calibrate** (button or `c`). The flow is fully
interactive:

- The corner you should point at **pulses and is labelled** (e.g. `TOP-LEFT`);
  the remaining corners are dim/numbered and captured ones show a green ✓.
- A line connects your live cursor to the active target, and the **reach quad**
  draws itself as you go.
- **Hold steady at the corner** and the ring around your cursor fills — at full
  it captures automatically (hands-free, so you don't need to reach the
  keyboard). You can still press **SPACE** to capture immediately.

After all 4 corners the homography is saved to `localStorage` and reused.
**Reset calib** returns to identity (1:1) mapping. You can also run calibration in
**Mouse test** mode to preview the flow without a camera.

## Controls

| Key | Action |
|-----|--------|
| `r` | reset all tile selections |
| `c` | run corner calibration |
| `f` | toggle fullscreen |
| `SPACE` | capture the current corner (during calibration) |
| `Esc` | cancel calibration |

Panel sliders: grid **rows/cols**, **dwell** time, 1-Euro **smooth** (min-cutoff)
and **beta**, plus **Mirror / Filter / Preview** toggles — same tuning knobs as
the Python CLI flags.

## Notes

- The pose model (~6 MB) and MediaPipe WASM load from a CDN on first run, so the
  first launch needs internet. Subsequent loads are cached by the browser.
- Requires `localhost` or `https://` — opening `index.html` via `file://` will
  block the camera.
- Chrome gives the most reliable WebGL/GPU path; Safari works but is slower.
