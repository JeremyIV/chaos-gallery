# Chaos gallery

**Live: https://jeremyiv.github.io/chaos-gallery/**

Interactive chaos and fractal demos. The unifying idea: take a fully deterministic system,
treat every pixel as an initial condition, and color it by its fate. Near the basin
boundaries, fate is exquisitely sensitive to the starting point — so the boundaries are
fractal, and you can zoom in more or less forever.

## Demos

- **[Plinko fractal](https://jeremyiv.github.io/chaos-gallery/#plinko)** — every pixel is a
  drop point, colored by the slot where the ball finally lands. Ball flight is solved
  *exactly* in a WebGL fragment shader (quartic root-finding per peg collision), so the
  fractal survives zooms up to 100,000×. Also lives standalone at
  [plinko-fractal-2](https://github.com/JeremyIV/plinko-fractal-2).
- **[Magnetic pendulum](https://jeremyiv.github.io/chaos-gallery/#pendulum)** — every pixel
  is a release point of a damped pendulum over three magnets, colored by which magnet
  captures it; brightness is time to capture. One full simulation per pixel, rendered
  progressively across Web Workers. Scroll to zoom, drag to pan, hover to trace a
  trajectory, click to pin it.

## Adding a demo

1. Create `your-demo/index.html` — self-contained, no build step, no dependencies.
2. Add an entry to the `DEMOS` array at the top of the root `index.html`.

Each demo is its own standalone page; the root page is just a switcher. Routing is
hash-based, so `/#your-demo` is a shareable link straight to a demo.
