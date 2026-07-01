# Vibersyn integration (2-wall setup)

This wires the **Vibersyn** idea projector into the gesture wall's two-wall setup:

- **Wall A** — the gesture control surface (`web/wall.html`). Coarse dwell-to-select
  tiles, exactly as before. Opened with `?vibersyn=<url>`, each tile also drives the
  Vibersyn projector.
- **Wall B** — the Vibersyn projector (`web/vibersyn.html`), which embeds the running
  Vibersyn web UI so it can be fullscreened on its own projector like any wall page.

Vibersyn is a separate service (a Bun-served React app with an HTTP API). Because
this repo assigns projectors *manually* (open a URL, fullscreen it — `display` in
`room.json` is decorative, see README), Vibersyn slots in simply as "the page for
wall B". Nothing in the Python pipeline changes.

## Run it

```
./run-2wall-vibersyn.sh          # prints the two URLs + the services to start
```

Which is:

1. Gesture server (this repo): `python -m gesturewall.server --config room.json`
   (serves `web/` on :8000, fusion WS on :8770).
2. Vibersyn (its own repo), with CORS allowing this web origin so wall A's dwells
   can POST cross-origin:
   ```
   VIBERSYN_CORS_ORIGIN=http://localhost:8000 VIBERSYN_PORT=8788 bun run start
   ```
3. Open each URL fullscreen on its projector:
   - Wall A: `http://localhost:8000/wall.html?wall=A&server=ws://localhost:8770&rows=2&cols=3&vibersyn=http://localhost:8788`
   - Wall B: `http://localhost:8000/vibersyn.html?src=http://localhost:8788/?live=1`

## Gesture → Vibersyn bridge

`web/vibersyn-bridge.js` maps a completed **dwell** (the wall's deliberate,
Midas-touch-resistant "select" gesture) to a Vibersyn HTTP action. It is **opt-in
and non-breaking**: `wall.js` calls `window.__vibersynBridge.onDwell(event, wall)`
at its dwell seam, but the bridge does nothing unless the wall is opened with
`?vibersyn=<url>`.

Default zone → action map (a 2×3 control grid; zone ids are `r{row}c{col}`):

| tile   | action         | kind    | Vibersyn endpoint            |
|--------|----------------|---------|------------------------------|
| `r0c0` | Idea Capture   | toggle  | `POST /api/capture`          |
| `r0c1` | Build idea     | oneshot | `POST /api/suggestion/accept`|
| `r0c2` | Auto-Build     | toggle  | `POST /api/auto-accept`      |
| `r1c2` | Emergency stop | oneshot | `POST /api/emergency-stop`   |

- **toggle** tiles send `{ on: <dwell-selected> }`; **oneshot** tiles fire only on
  the select edge (never on deselect).
- Override the map with `&vibersynmap=r0c0:emergency,r0c1:capture` (actions:
  `capture`, `accept`/`build`, `autobuild`, `emergency`).

**Idea Capture mode** is Vibersyn's explicit "start the creation loop" mode — the
`r0c0` tile toggles it, so someone at the wall can start/stop idea capture with a
gesture, with the Vibersyn projector (wall B) showing the captured idea building.

## Tests

Pure bridge logic is checked headless (matches the repo's `_*_check.mjs` convention):

```
node web/_vibersyn_bridge_check.mjs
node --check web/vibersyn-bridge.js
```
