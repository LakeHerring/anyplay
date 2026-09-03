# AnyPlay — Autonomous Game AI

A neural agent that plays **any game** it can see and control on Linux —
built first around *Shadow Dungeon* (GOG, run through Proton), the first
reference testbed. It **learns by playing** — not by imitating a human. It
interacts with the game, observes the consequences of its own actions, and
improves its policy from the reward signal.

> **The AI's own actions are the training actions. Human input is optional and
> is never treated as the "correct" action by default.**

The design goal: build a **reliable environment and observation system first**,
and keep the **policy and the training algorithm completely modular**, so the
same visual/game environment becomes the foundation for evolutionary strategies,
reinforcement learning, and future experimental learning methods — without
rewriting the capture infrastructure.

---

## The core loop

```
GAME
  ↓
Visual Observation
  ↓
AI Policy
  ↓
Action
  ↓
GAME
  ↓
Reward / Outcome
  ↓
Learning
  ↓
Improved Policy   (back to Visual Observation)
```

The agent's loop is closed by *environment feedback*, not by a human
demonstration.

---

## General by design: any game, not one game

The target is an agent that can play **any game it can see and input** —
any window it can capture and any keyboard/mouse actions it can inject.
Shadow Dungeon is the **first reference testbed** (2D top-down, simple HUD,
cheap to parse, forgiving to play), not the product. A future game is a
**thin environment adapter**, not a rewrite:

| Game-agnostic (built once) | Game-specific (the adapter, per game) |
|---|---|
| capture (X11 / Wayland portal) | **reward function** (the most game-specific part) |
| observation layer, temporal buffer | **terminal-state detection** (win/loss/quit) |
| policy contract + model families | **action vocabulary** for the Qwen teacher (mapping to raw keys/axes) |
| trainers (RL / ES / BC / …) | curriculum / objectives (optional) |
| experience format + dataset tooling | game launch / window targeting (already mostly automatic) |
| action-space discovery (keys/axes discovered, not hardcoded) | |

So "new game" means: capture works out of the box, the action space is
discovered automatically from recording, and you implement a reward +
terminal-state adapter (plus, if using the VL teacher, that game's action
vocabulary). The capture pipeline, observation layer, policy architecture,
and trainers never change.

---

## Guiding principles

1. **Reliable capture first.** Do not optimize the visual pipeline prematurely.
   A stable CPU pipeline is more valuable than an unreliable zero-copy GPU
   pipeline. GPU transfer is optimized only after reliable capture is proven.
2. **Modular by contract.** The environment, the policy, and the trainer are
   separate components. Swap the policy or the trainer; the environment and the
   capture never change.
3. **Develop each layer independently.** Never debug Portal, PipeWire,
   GStreamer, C, Python, PyTorch and ROCm as one lump — test each layer on its
   own (see the nine tests below).
4. **Priority order** — do not sacrifice 1–4 for 7:
   correctness → reliability → observability → synchronization → latency →
   throughput → GPU optimization.

---

## Architecture

```
┌──────────────────────┐
│        GAME          │
└──────────┬───────────┘
           │
           ▼
    Visual Capture
    Portal / PipeWire
    GStreamer / C
           │
           ▼
    Observation Layer
    Resize / Normalize
    Temporal Buffer
           │
           ▼
        POLICY            (action)
           │
           ▼
         GAME             (the action is applied)
           │
           ▼
        REWARD            (the environment scores the outcome)
           │
           ▼
      EXPERIENCE          (obs, action, reward, next_obs, done)
           │
           ▼
       TRAINER            (how the policy changes)
           │
           ▼
    Improved Policy       (back into the loop)
```

### The modular contract

The whole system hangs off three decoupled roles. The environment is
**algorithm-agnostic**: it does not know *how* the policy is trained, and the
policy does not know *what* the trainer does.

| Component | Provides | Knows about |
|-----------|----------|-------------|
| **Environment** | `observation`, `reward`, `terminal state` | the game, capture, reward |
| **Policy** | `action` | the observation |
| **Trainer** | *how the policy changes* | the experience stream |

Because of this split, the **same environment** should eventually support any
of these trainers, unchanged:

* Evolutionary Strategies (population of policies)
* Reinforcement Learning
* Policy Gradient / Actor-Critic / PPO
* DQN (value-based)
* behavioral cloning (human data, optional)
* other experimental algorithms

Only the policy/trainer changes between these. The visual capture system stays
identical.

---

## The fundamental unit: the *experience*

The critical relationship is **not** "video + human input." It is the causal
chain through the environment:

```
observation_t  →  action_t  →  environment  →  reward_t  →  observation_{t+1}
```

The system therefore records a synchronized experience:

```
(
    observation_t,
    action_t,
    reward_t,
    observation_{t+1},
    done
)
+ metadata: timestamp, frame_id, episode_id, environment_id,
            input_state, game_state, policy_version
```

Every observation carries a **timestamp / frame id**; every action carries a
**timestamp**. That is what lets the environment close the loop and produce
`reward_t` and `observation_{t+1}`. One experience stream supports supervised,
behavioral cloning, RL, offline RL, evolutionary strategies, and dataset
analysis without re-recording.

---

## Reward is the learning signal

Because this is autonomous learning, **reward design is central**. The agent
improves from signals such as:

* game progress / objectives completed
* survival / health
* successful outcomes (e.g. room cleared, item collected)
* failures (death, stuck, regression)
* resource usage / efficiency

**Reward hacking must be considered during environment design** — a reward that
is easy to game will teach the wrong behavior. Reward is derived from the game
state (parsed from the observation and/or the environment), and its design is a
first-class part of the Environment, separate from the Trainer.

The reward function is also the **most game-specific component** in the whole
system — it lives in the per-game environment adapter (see
*General by design*). Everything downstream of it (experience stream,
trainers, policy architecture) is game-agnostic.

---

## Temporal vision

A single frame is usually not enough. The observation system supports
sequences so the agent can infer motion and temporal context (e.g. *enemy
stationary* vs. *enemy moving toward you*):

```
frame[t-3]  frame[t-2]  frame[t-1]  frame[t]  →  POLICY  →  ACTION
```

The temporal architecture is a **swappable** choice, not hardcoded:
frame stacking, recurrent networks, temporal CNNs, transformers, or
state-space models. The current policy uses a small CNN per frame plus a GRU
over the window.

---

## Evolutionary-strategy compatibility

ES is a first-class training mode, not an afterthought:

```
Population
   ↓
┌───────────────┐
│               │
Policy A      Policy B …
│               │
↓               ↓
GAME            GAME
│               │
↓               ↓
Fitness A     Fitness B
   └──────┬──────┘
          ↓
      Selection
          ↓
       Mutation
          ↓
    New population
```

Each policy plays the **same** environment (the visual capture is unchanged);
fitness comes from the reward signal. The project ships a legacy ES scaffold in
`anyplay/evolution/` (currently not wired to the visual policy —
it becomes the ES trainer under the modular contract).

---

## Human input is optional

Human input can be captured for:

* debugging and human-vs-AI comparison
* gameplay analysis
* optional behavioral-cloning experiments

But:

```
human action ≠ correct action
```

The autonomous agent must be capable of learning **without** human
demonstrations. Captured human input is data for analysis and optional BC, not
the default supervision signal.

---

## Visual pipeline

```
Game → XDG Desktop Portal → PipeWire → GStreamer → C++ capture core
     → Python → observation preprocessing → PyTorch / ROCm → model
```

**Ownership:**

* **C++ core / GStreamer** owns: screen capture, PipeWire integration, media
  transport, frame acquisition, timing, input capture/injection.
* **Python** owns: observation preprocessing, NumPy, PyTorch, ML models,
  training, evaluation, dataset management, experiment management.

Python never does low-level PipeWire/GStreamer capture; the C++ core never
knows about the network.

### Two backends, one contract

The capture layer's only job is to hand Python real, correctly-shaped RGB frames
with timestamps.

* **X11** — `ffmpeg x11grab` (recording), `mss` screenshots (play), GStreamer
  `ximagesrc` (daemon).
* **Wayland** — `xdg-desktop-portal` ScreenCast → PipeWire → GStreamer → the
  C++ daemon, published into a POSIX shared-memory ring that Python reads as a
  zero-copy NumPy view (`native_capture.py`). This is the Wayland path and shows
  a portal consent dialog. It is **damage-driven**: frames arrive when the
  screen changes, so a static screen legitimately yields zero frames.

Per-frame contract: `frame_id, timestamp, presentation_timestamp, width,
height, format (RGB)`. Input events carry `timestamp, event_type, key/button,
pressed/released`.

### Capture and inference are separate

```
                 CAPTURE
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
         RECORDER      INFERENCE
             │             │
             ▼             ▼
          DATASET        POLICY
```

* Capture at the useful/native resolution and a high rate
  (`1920×1080 @ 60 FPS`); the **recorder** preserves that high-rate data for
  analysis and training.
* The agent **infers at 15–30 FPS** on a **shallow / latest-frame queue**, so it
  never acts on stale frames.
* Downsize and decimate **after** capture, in Python
  (`observation: 640×360 or 128×96`).

### The capture daemons

Capture runs in a **C++ core** (`core/`, build with `make -C core`):

* `core/src/main.cpp` → `core/build/anyplay-capture` — X11 GStreamer daemon:
  pipeline → shared-memory frame ring + control socket. With
  `--keyboard`/`--pointer` it also runs the **evdev input ring** (raw evdev
  events published into a second shm ring); `--input-only` skips the video
  pipeline entirely.
* `core/src/portal_main.cpp` → `core/build/anyplay-portal-capture` — Wayland
  xdg-desktop-portal + PipeWire daemon, same ring/socket/input protocol.
* `core/test/` — headless regression tests for the rings
  (`make -C core test`).

`native/capture_daemon.c` and `native/portal_capture.c` are the original C
daemons, kept as the reference implementation. `native_capture.py` is a thin
zero-copy client over the frame ring — the returned frame is a shared-memory
view, so copy it immediately if it must outlive the call; `input_ring.py` is
the matching client for the input-event ring.

---

## Develop the pipeline incrementally

Each layer is a standalone, independently testable milestone; if one fails,
debug *only* that layer.

```
TEST 1   Portal → PipeWire          (get a stream/fd from xdg-desktop-portal)
TEST 2   PipeWire → GStreamer       (pipewiresrc negotiates, reaches PLAYING)
TEST 3   GStreamer → C appsink      (frames arrive at the C appsink)
TEST 4   C → valid video frames     (dims/format/timestamps correct, count > 0)
TEST 5   C → Python                 (shared-memory ring → NumPy view)
TEST 6   Python → NumPy             (reshape/dtype correct, real pixels)
TEST 7   NumPy → PyTorch            (tensor on device, no copy surprises)
TEST 8   PyTorch → ROCm             (CUDA/ROCm inference runs)
TEST 9   Full real-time pipeline    (live, low-latency, end to end)
```

The first milestone is a deliberately boring capture program that does
**nothing ML-related**: obtain a ScreenCast stream, connect to the PipeWire
stream, receive frames through GStreamer, convert to a known raw format,
deliver through appsink, verify dimensions/format/timestamps, and count frames
and dropped frames. Diagnostic output looks like:

```
CAPTURE OK
resolution: 1920x1080
format: RGB
fps: 60.0
frames: 18432
dropped: 0
latency: X ms
```

Only after this is reliable do ROCm/PyTorch enter the picture.

---

## Data / session format

A capture session lives in `datasets/<session>/`:

```
datasets/<session>/
    video.mp4          # 60 FPS master recording — keep forever
    inputs.jsonl       # raw evdev events (keyboard + mouse), t = sec since start,
                       # each line tagged with the source device path
    meta.json          # clocks, geometry, device info, alignment offset
    action_space.json  # discovered buttons/axes + axis bounds (from build)
    checkpoints/       # policy checkpoints (from train)
```

`meta.json` stores the offset between the input and video clocks so input events
map exactly onto video frames. The action space is **discovered, not
hardcoded**: every key code and axis that appears becomes an action dimension;
axis values are normalized to `[-1, 1]` from the observed min/max.

The explicit experience stream (`reward`, `done`, and per-experience metadata)
is the canonical unit the autonomous pipeline will emit from this session
(Phase 7).

---

## The model

The **policy** is one swappable component in the contract:

* **FrameEncoder** (`training/models/encoder.py`) — a 4-layer CNN → 256-d
  feature per frame, tanh-bounded. Small and fast by design (it runs at 30 FPS
  during play).
* **Policy** (`training/models/policy.py`) — GRU over the frame-feature window
  → last hidden state → button logits (sigmoid at play) + axis values (tanh,
  denormalized at play).

The **current trainer** is behavioral cloning (supervised imitation):
`BCEWithLogits` on buttons (pos-weighted by key duty) + `MSE` on axes. It is a
useful baseline and one of the interchangeable trainers — the target trainer is
reward-driven, acting on the experience stream. Checkpoints embed the action
space so play can map outputs back to real key codes.

---

## Development phases / roadmap

```
Phase 1   Capture              Portal → PipeWire → GStreamer → C   (in progress)
Phase 2   Frame validation     resolution / format / rate / counts / dropped /
                               latency (the diagnostic program above)
Phase 3   Python bridge        C → Python → NumPy (shared memory)
Phase 4   Observation          resize / normalize / temporal buffering
Phase 5   PyTorch              NumPy → PyTorch
Phase 6   ROCm                 inference + training on AMD GPU
Phase 7   Environment + reward reward signal, experience stream (obs, action,
                               reward, next_obs, done), terminal state
Phase 8   Trainer (modular)    RL / ES / PG / actor-critic / PPO / DQN against
                               the experience stream
Phase 9   Optimization         DMA-BUF / GPU preprocess / zero-copy / pooling —
                               only after Phase 1–8 are reliable
```

* **X11** capture is mature; **Wayland** (portal → PipeWire → C daemon) is the
  active focus, with the daemon and shared-memory bridge in place.
* The **Environment / Policy / Trainer** split (reward-driven, autonomous) is
  the target; the codebase currently runs behavioral cloning on top of the same
  capture foundation.

---

## Requirements & setup

* **Kubuntu, Wayland** (X11 also supported) — the game runs via Proton.
* `ffmpeg` in PATH (X11 capture).
* GStreamer (`gstreamer-1.0`) with `decodebin`, `videoconvert`, `videoscale`,
  `pipewiresrc`; PipeWire; `xdg-desktop-portal` (Wayland capture).
* Python 3.12+ (tested on 3.14) in a venv.
* GPU (optional but recommended): AMD → the **PyTorch ROCm** build
  (`torch.cuda.is_available()` is True under ROCm). The GPU is used for
  neural-network compute, **not** for capture, unless profiling shows capture
  is the bottleneck.
* Input recording and UInput injection read/write `/dev/input`. Add your user
  to the `input` group once, then re-login:

```bash
sudo usermod -aG input $USER
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# AMD GPU (e.g. RX 7900 XTX / gfx1100) — ROCm build of PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/rocm
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Build the C++ capture core (prebuilt binaries are included in `core/build/`,
rebuild if you change the C++):

```bash
make -C core
make -C core test   # headless regression tests (frame + input rings)
```

The legacy C daemons in `native/` build with `./native/build.sh`.

---

## Usage

```bash
# 0. What can it see?
python main.py devices

# 1. Record a session (60 FPS master video + raw input log).
#    X11 / auto-detect (default): the game window is located by title; --launch
#    starts it via its .desktop/Lutris shortcut if it isn't running:
python main.py capture --session run1 --duration 300
#    (omit --duration to stop with Ctrl-C; --region x,y,w,h overrides the
#     auto-detected window; --wait-window sets how long to wait)
#
#    Wayland games (no X11 window -- e.g. Proton / GE-Proton titles): use the
#    portal backend. A KDE "Share Screen" window picker opens; pick the game
#    window and it is recorded at its native resolution (auto-detected):
python main.py capture --session run1 --duration 300 --backend portal

# 2. Discover which keys/axes the recording actually uses:
python main.py build --session run1

# 3. Train (current baseline: behavioral cloning; the reward-driven trainer is
#    the target):
python main.py train --session run1 --epochs 10

# 4. Play — the agent drives the game through a virtual keyboard.
#    --source picks the capture backend: mss (X11), native (GStreamer daemon),
#    or portal (xdg-desktop-portal + PipeWire — real pixels on Wayland):
python main.py play --checkpoint datasets/run1/checkpoints/policy_best.pt --source portal

# 4b. Self-test the whole inference path without screen/UInput:
python main.py play --checkpoint datasets/run1/checkpoints/policy_best.pt --smoke

# 5. Ask the Qwen VL model about a screenshot (auto-starts the llama.cpp
#    server on :8198; vl-serve / vl-stop manage it manually):
python main.py vl "what is happening and what next?" --image shots/now.png

# 6. Run the closed-loop agent: capture -> temporal obs -> Qwen -> action
#    controller -> input. --source portal is the Wayland path (window
#    picker); without --real-input actions are logged only (dry run);
#    --dataset appends training records; --daemon points at the C++ core:
python main.py agent --source portal --model --decision-interval 0.5

# 7. Health checks (no game required):
python main.py obs-check --source test      # observation buffer, synthetic frames
python main.py core-status --sock <socket>  # per-stage latency + drops dashboard
```

---

## Repo layout

```
main.py                  # CLI: devices / capture / build / train / play / agent /
                         #      vl / vl-serve / vl-stop / core-status / obs-check
anyplay/
    capture/             # C→Python bridge + capture backends
        native_capture.py    # zero-copy client over the daemon's shared-memory ring
        capture.py           # run_capture(): 60 FPS video + input log
        video_capturer.py    # ffmpeg x11grab (X11)
        video_capturer_gst.py# GStreamer capture (X11)
        video_capturer_portal.py # portal/PipeWire window capture (Wayland, any game)
        input_ring.py        # zero-copy client over the C++ input-event ring
        input_recorder.py    # evdev event logger (legacy Python input path)
        window.py            # X11 window auto-detect by title
    actions/               # action space (vocabulary), strict VL output parser,
                           # action controller (validate / duration / conflicts)
    agent/                 # multi-rate agent loop (temporal obs → Qwen → input)
    obs/                   # timestamped temporal observation buffer
    vl/                    # Qwen3.5-4B VL client (llama.cpp server, OpenAI API)
    training/
        data/            # Session + frame decode + action alignment + dataset
        models/          # FrameEncoder (CNN) + Policy (GRU) — the policy component
        training.py      # current trainer: behavioral cloning
    game_integration/    # live play: capture → policy → UInput virtual device
    neural_networks/     # legacy from-scratch NN scaffold (kept)
    evolution/           # legacy ES scaffold — future ES trainer (kept)
    utils/               # config, logger, visualizer
core/
    Makefile               # g++ build of both C++ daemons
    include/anyplay/       # shm_protocol.h (frame ring), input_ring_protocol.h
    src/                   # main.cpp (X11), portal_main.cpp (Wayland), frame ring,
                           # evdev input ring, control socket
    build/                 # anyplay-capture, anyplay-portal-capture
    test/                  # headless regression tests (regression_*.py)
native/                    # legacy C daemons (reference implementation)
    capture_daemon.c     # GStreamer pipeline → shared-memory ring
    portal_capture.c     # xdg-desktop-portal + PipeWire source
    build.sh             # build both
tools/
    portal_pw_fd.c       # obtain the PipeWire stream/fd from the portal (TEST 1)
    portal_probe.py      # portal/PipeWire diagnostics (TEST 1–2)
    pw_probe.py          # PipeWire object/property probe
    bench_capture.py     # frame-rate / latency benchmark
datasets/<session>/      # recorded sessions (video, inputs, meta, checkpoints)
```

## Notes

* The **capture/observation foundation is stable by design** — the policy and
  trainer are meant to be swapped without touching it.
* Record **clean** runs if you use human data: the agent can learn from what you
  do, including idling, but human action is *data*, not ground truth.
* 30 FPS / 128×96 training is the starting point — the 60 FPS master lets you
  re-train at higher rates or with longer windows later, without re-recording.
* **Keyboard and mouse are both recorded** (auto-detected: every keyboard and
  every pointer/mouse node — motion, buttons, wheel). Mouse buttons (EV_KEY)
  are button actions; relative mouse motion (EV_REL) and wheel are per-frame
  motion/axis actions. Old single-device sessions still load fine.


---

# Geometric Reasoning and Spatial World Model

A major optimization is to separate **visual perception** from **spatial reasoning**.

The LLM should not be required to repeatedly reason directly over every pixel in
the game frame. Instead, the perception layer should progressively transform
raw visual observations into a compact, temporal, geometric representation of
the game state.

The goal is:

```text
Raw Pixels
    ↓
Fast Vision / Object Detection
    ↓
Tracking
    ↓
Geometry + Temporal Analysis
    ↓
Spatial World State
    ↓
Qwen / RL Policy
    ↓
Action
```

This does not replace visual input. It creates a structured representation that
preserves the information most relevant to decision-making.

## Why geometric reasoning matters

A conventional VLM might receive a screenshot and determine:

> "There is an enemy to the right."

A geometric representation can instead provide:

```text
Enemy:
    distance: 8.4 m
    bearing: +12°
    velocity: toward player
    closing_speed: 2.1 m/s
    threat: 0.87

Wall:
    blocks_direct_path: true

Cover:
    distance: 3.2 m
    bearing: -45°
    protection: 0.92
```

The model can therefore reason about **relationships and changes in the
environment**, rather than repeatedly rediscovering basic spatial facts from
raw pixels.

### Design rationale: geometry + magnitude → scale-invariant state

A deeper reason the spatial representation pays off: **order-of-magnitude
reasoning is a coarse form of geometric reasoning**. What matters is rarely
the absolute value — it is the *relationship* between quantities: direction,
distance, ratio — and how those relationships transform.

Orders of magnitude are naturally logarithmic: 10⁶ → 10⁷ → 10⁹ is +1, +2
orders, and each ×10 step occupies the same "distance" in log space. A model
can reason *"the world is roughly two orders of magnitude larger than the
country,"* and — more usefully here — learn that **×10 is one transformation
regardless of scale**. That scale invariance is what lets a representation
learned in one room (or at one capture resolution) transfer to another: the
policy learns structural rules — *"enemy is close, HP ratio 20:1 → an
aggressive attack is favorable"* — instead of memorizing absolute pixel and
health values.

```text
              REASONING
                 │
        ┌────────┴────────┐
        │                 │
   GEOMETRY           MAGNITUDE
   relationships      relative scale
   directions         ratios
   distances          powers
   transformations     hierarchy
        │                 │
        └────────┬────────┘
                 │
            STRUCTURE
                 │
          generalization
```

**Implications for AnyPlay** — the spatial state is expressed *relative to the
player and to game scale*, not in absolute pixels:

* **`spatial_state` schema** — normalized fields: distance as a fraction of
  view/room radius, closing speed as view-crossing time, HP as ratios, threat
  as a probability. Absolute screen coordinates are kept only for low-level
  control and debugging.
* **Qwen prompt** — receives ratios, not absolutes ("enemy HP is 5% of
  yours"), which is both token-cheap and scale-invariant.
* **Student encoder** — consumes the same ratio/log features, so a policy
  trained in one room or at one resolution needs no re-normalization for
  another.

## Spatial state

The observation layer should be capable of maintaining a structured world
state containing, where available:

- player position and orientation
- object positions and bounding geometry
- object velocity and direction
- relative distance and bearing
- obstacles and traversable areas
- line-of-sight / path obstruction
- targets and objectives
- predicted trajectories
- object identity and tracking history
- confidence and last-seen timestamps

Example:

```json
{
    "player": {
        "position": [42.1, 17.3],
        "heading": 91.0,
        "health": 73
    },
    "enemies": [
        {
            "distance": 8.4,
            "bearing": 12.0,
            "closing_speed": 2.1,
            "threat": 0.87
        }
    ],
    "cover": [
        {
            "distance": 3.2,
            "bearing": -45.0,
            "protection": 0.92
        }
    ],
    "objective": {
        "bearing": 120.0,
        "distance": 31.5
    }
}
```

The exact schema is game-dependent, but the underlying spatial concepts should
remain reusable.

## Relative coordinate systems

Where practical, represent objects relative to the player rather than relying
only on absolute screen coordinates.

For example:

```text
                 NORTH
                   ↑
                   │
            Enemy │
              ↖   │
                \ │
WEST ───────── PLAYER ───────── EAST
                   │
                   ↓
                 SOUTH
```

This lets the policy reason directly about:

- left / right
- front / behind
- distance
- bearing
- approaching / retreating
- nearby cover
- reachable objectives

Screen-space coordinates can still be retained for low-level control and
debugging.

## Temporal geometric reasoning

Geometry becomes substantially more useful when combined with temporal state.

Instead of repeatedly sending:

```text
Frame 1: enemy=(580,305)
Frame 2: enemy=(579,305)
Frame 3: enemy=(578,305)
```

the system should maintain:

```text
Enemy_1:
    position
    velocity
    acceleration
    heading
    predicted_trajectory
    threat_level
    last_seen
```

The LLM can then receive meaningful changes:

```text
Enemy_1:
    velocity changed → -2.1 m/s
    trajectory intersects player in ~1.8 s

Door_1:
    state changed → closed → open
```

This provides temporal memory without requiring the LLM to repeatedly infer
motion from a large sequence of nearly identical images.

## Multi-rate architecture

The geometric layer should operate at a higher rate than the LLM:

```text
Capture                  60 FPS
    ↓
Fast perception       10–60 FPS
    ↓
Tracking + geometry   10–60 FPS
    ↓
World-state updates   continuous
    ↓
Qwen reasoning         2–5 Hz initially
    ↓
RL / low-level policy  30–60 Hz
```

These rates are independently configurable and must be benchmarked.

Qwen inference must never block capture or low-level control. The system should
use an asynchronous observation/world-state queue and prefer the latest valid
state over stale observations.

## LLM as reasoning layer, not physics engine

The architecture should avoid making Qwen simultaneously perform:

```text
vision + object tracking + geometry + physics + control
```

Instead:

```text
Fast deterministic systems:
    capture
    preprocessing
    object detection
    tracking
    geometry
    temporal state

ML / LLM:
    interpretation
    planning
    strategy
    high-level decisions

RL / control:
    fast action selection
    precise low-level control
```

This separation reduces the amount of information the LLM must process and makes
the system easier to profile, debug, and replace.

## Semantic compression

The purpose of geometric state is not merely to reduce token count. It is to
remove irrelevant visual information while preserving decision-relevant
information.

Conceptually:

```text
~2 million pixels
       ↓
objects / features / tracking
       ↓
~tens or hundreds of meaningful values
       ↓
LLM reasoning
```

The exact reduction depends on the game and perception system. The benchmark
must measure whether the compact representation actually improves:

- inference latency
- throughput
- decision quality
- stale-decision rate
- memory usage
- training efficiency

Do not assume a speedup without measuring it.

## Hybrid visual + geometric observations

Geometric state should **complement**, not necessarily replace, visual input.

For difficult or uncertain situations, Qwen can receive both:

```text
Current frame
    +
Temporal context
    +
Spatial world state
    ↓
Qwen
```

This allows the VLM to inspect visual details that the fast perception system
failed to represent while still providing an explicit spatial model.

A useful confidence mechanism is:

```text
High-confidence geometry
    → compact state is sufficient

Low-confidence / novel situation
    → provide richer visual context
    → invoke Qwen
    → update world model
```

This supports a future **adaptive reasoning budget** where expensive visual
reasoning is used primarily when it is valuable.

## Persistent world model

The long-term architecture should maintain a persistent world model across
frames and decisions:

```text
                 ┌─────────────────────┐
                 │    World Model       │
                 │                     │
                 │ objects             │
                 │ positions           │
                 │ velocity            │
                 │ relationships       │
                 │ history             │
                 │ predictions         │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        Fast RL Policy                Qwen
        30–60 Hz                    2–5 Hz
              │                           │
              └─────────────┬─────────────┘
                            ▼
                         Action
```

Qwen can therefore become a high-level planner and supervisory model instead of
being required for every control step.

## Qwen teacher → geometric/RL student

The teacher architecture should record not only screenshots and actions, but
the geometric state available when each decision was made:

```text
observation
spatial_state
action
reward
next_observation
next_spatial_state
done
timestamp
episode_id
policy_version
```

This creates a richer training dataset:

```text
visual situation
      +
spatial state
      ↓
teacher decision
      ↓
environment outcome
```

A smaller policy can then learn to reproduce useful decisions while being
trained further using environment reward:

```text
             Qwen3.5-4B VL
                    │
                    ▼
          Teacher demonstrations
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Visual dataset       Spatial dataset
          │                   │
          └─────────┬─────────┘
                    ▼
              RL / Policy
                    │
                    ▼
             Fast gameplay
```

The eventual objective is for Qwen to be needed less frequently as the student
policy becomes capable of handling familiar situations autonomously.

## Implementation boundary

The geometric reasoning system should be treated as part of the **Observation /
World Model layer**, not hardcoded into Qwen.

Recommended conceptual interfaces:

```text
Capture
  → Frame

Perception
  → Objects / Features

Tracker
  → Persistent Objects

Geometry
  → Relations / Distances / Motion / Predictions

World Model
  → Spatial State

Policy / Qwen
  → Decision

Action Controller
  → Validated Input
```

This keeps the architecture compatible with different vision models, RL
algorithms, and games.

## Development priority

Geometric reasoning should be introduced incrementally:

1. Detect player and major objects.
2. Track object identity across frames.
3. Add relative position, distance, and bearing.
4. Add velocity and motion estimation.
5. Add obstacle/path relationships.
6. Maintain persistent temporal world state.
7. Feed compact spatial state to Qwen.
8. Compare Qwen using raw vision versus vision + geometry.
9. Train an RL/student policy using the combined representation.
10. Benchmark whether the geometric representation improves latency,
   throughput, decision quality, and sample efficiency.

The important experimental principle is:

> **Measure the benefit of geometric reasoning rather than assuming it.**

The expected advantage is that expensive models spend their compute on
**reasoning and planning**, while high-frequency perception and spatial
bookkeeping are handled by cheaper, deterministic or specialized systems.

---

# Recommended C++ + Python Runtime Architecture

The runtime should use a **hybrid C++ + Python architecture**.

## C++ — Real-time game/runtime layer

C++ owns performance-sensitive and OS-facing functionality:

- PipeWire / XDG Desktop Portal screen capture
- GStreamer
- 60 FPS frame capture
- Frame/ring buffers
- Timing and synchronization
- Input capture/injection
- Game/environment interface
- Low-latency IPC/shared memory

C++ is preferred over C for the application/runtime layer because the project benefits from RAII, threading, resource management, stronger abstractions, and maintainable buffer/input abstractions. Native GStreamer/PipeWire APIs can still be used directly.

## Python — AI/ML layer

Python owns the intelligence and experimentation layer:

- Observation construction
- Vision preprocessing
- PyTorch + ROCm
- Qwen3.5-4B VL inference
- RL algorithms
- Training
- Dataset generation
- Evaluation and experiments

Do **not** move the ML stack to C++ merely for performance. PyTorch and the underlying GPU kernels already provide native/GPU execution while Python keeps model and training development flexible.

## Runtime data flow

```text
                         GAME
                           │
                    PipeWire / GStreamer
                           │
                           ▼
                  ┌──────────────────┐
                  │      C++         │
                  │ Runtime Layer    │
                  │                  │
                  │ Capture          │
                  │ Frame Buffer     │
                  │ Timing           │
                  │ Input            │
                  │ Environment      │
                  └────────┬─────────┘
                           │
                    Shared memory / IPC
                           │
                           ▼
                  ┌──────────────────┐
                  │     Python       │
                  │    AI / ML       │
                  │                  │
                  │ Observation      │
                  │ Qwen3.5-4B VL    │
                  │ RL Policy        │
                  │ Training         │
                  └────────┬─────────┘
                           │
                       Action
                           │
                           ▼
                         C++
                           │
                     Input Controller
                           │
                           ▼
                         GAME
```

The boundary should remain modular so Qwen, the RL algorithm, or the policy can be replaced without rewriting the capture/environment infrastructure.

---

# Qwen3.5-4B VL + RL Decision Architecture

Qwen3.5-4B VL should initially be treated as a **high-level visual reasoning/teacher model**, not as the 60 FPS low-level controller.

## Multi-rate pipeline

```text
Game capture:             60 FPS
Low-level control:        30–60 FPS
Fast perception:          10–30 FPS
Qwen reasoning:            2–5 decisions/sec initially
```

These rates must be configurable and independently benchmarked.

Capture must never block because Qwen is performing inference. Use an asynchronous/latest-frame observation queue so the agent does not act on stale frames.

## Temporal observations

Do not rely exclusively on a single screenshot. Maintain a temporal observation window, for example:

```text
frame[t-400ms]
frame[t-200ms]
frame[t]
      │
      ▼
Qwen / Policy
      │
      ▼
    Action
```

This allows the model to infer movement and changes in the environment.

The temporal mechanism remains swappable: frame stacking, GRU/RNN, temporal CNN, transformer, state-space model, etc.

## Qwen action interface

Use a strict, **per-game** action vocabulary rather than unrestricted
natural-language commands. The vocabulary is defined by the game's environment
adapter (see *General by design*): it is a small, curated set of game-level
concepts that map down to the discovered raw keys/axes. Changing game =
changing this vocabulary, not changing the parser, controller, or input
backend.

Example (Shadow Dungeon, the first reference environment):

```text
MOVE_LEFT
MOVE_RIGHT
MOVE_FORWARD
MOVE_BACK
JUMP
ATTACK
DEFEND
INTERACT
WAIT
DODGE_LEFT
DODGE_RIGHT
```

Qwen should return structured output such as:

```json
{
    "action": "DODGE_LEFT",
    "confidence": 0.87
}
```

During development, optional reasoning may be logged for debugging, but production control should minimize unnecessary generated text.

Never connect unrestricted Qwen output directly to keyboard/mouse APIs.

Use:

```text
Qwen
 ↓
Action Parser
 ↓
Action Controller
 ↓
Input Backend
 ↓
GAME
```

The action controller validates actions, handles durations, prevents conflicting inputs, rate-limits commands, and guarantees correct key/button release.

---

# Qwen as Teacher → RL Student

The long-term goal is to avoid requiring a 4B VLM for every low-level decision.

Initially:

```text
Observation
    ↓
Qwen3.5-4B VL
    ↓
Action
    ↓
Game
    ↓
Reward
```

Record the complete interaction:

```text
observation
action
reward
next_observation
done
timestamp
episode_id
policy_version
```

This creates training data describing:

```text
visual situation → decision → outcome
```

A smaller/faster policy can then learn from these demonstrations and environment rewards:

```text
Qwen3.5-4B VL
      │
      ▼
Teacher demonstrations
      │
      ▼
Policy dataset
      │
      ▼
Small vision/RL policy
      │
      ▼
Real-time gameplay
```

Qwen can eventually become a supervisory model that is invoked less frequently or only for unfamiliar situations.

The final architecture should therefore aim toward:

```text
GAME
 ↓
60 FPS Capture
 ↓
Observation Buffer
 ├───────────────┐
 │               │
 ▼               ▼
Fast            RL Policy
Perception          │
 │                  ▼
 └──────────────→ Action
                    │
                    ▼
               Input System
                    │
                    ▼
                   GAME
```

while Qwen primarily supports training, supervision, exploration, and difficult/high-level decisions.

---

# Concrete Implementation Order

1. **Capture:** Portal → PipeWire → GStreamer → C++/native layer.
2. **Frame validation:** resolution, format, FPS, timestamps, dropped frames, latency.
3. **C++ → Python bridge:** shared-memory/ring-buffer transport.
4. **Observation:** resize, normalize, temporal buffering.
5. **PyTorch:** NumPy → tensors.
6. **ROCm:** verify AMD GPU inference/training.
7. **Environment:** action execution, reward, terminal state.
8. **Qwen integration:** temporal visual observations → constrained structured actions.
9. **Closed loop:** observation → Qwen/policy → action → game → reward.
10. **Dataset:** persist synchronized experiences.
11. **RL/ES trainers:** plug interchangeable trainers into the same environment contract.
12. **Student policy:** train a smaller/faster policy from Qwen data plus reward.
13. **Optimization:** only after correctness and reliability; then consider zero-copy/DMA-BUF/GPU preprocessing/pooling.

The core rule remains:

> **Correctness → reliability → observability → synchronization → latency → throughput → GPU optimization.**

Do not sacrifice the earlier properties for premature optimization.

# Cross-Entropy, Probability Geometry, and Geometric Reasoning

## Core Concept

Cross-entropy should be viewed as more than a standard classification loss. It provides a way to measure **how far a model's probability distribution is from the desired distribution**, and it naturally connects to geometric and order-of-magnitude reasoning through logarithmic space.

### Cross-Entropy

For a correct class with predicted probability \(p\):

$$
L=-\log(p)
$$

Lower probability assigned to the correct outcome produces a larger penalty.

```text
p(correct)    Loss
1.00          0
0.90          0.105
0.50          0.693
0.10          2.303
0.01          4.605
```

Because cross-entropy operates logarithmically, multiplicative probability differences become additive differences in loss.

For example:

```text
0.90 → 0.09
```

is a 10× reduction in probability, corresponding to an additive \(\log(10)\) increase in loss.

---

## Connection to Order of Magnitude

Logarithms provide a common mathematical bridge between probability and magnitude.

```text
Magnitude:
10 → 100 → 1000
      ×10    ×10

Log space:
1  →   2  →   3
      +1    +1
```

Likewise:

```text
Probability:
0.9 → 0.09

Log probability:
log(0.9) → log(0.09)
              ↓
           −log(10)
```

Therefore, logarithmic representations allow AI systems to reason about **relative scale rather than only absolute values**.

---

## Geometric Interpretation

Probability distributions can be treated as points in a mathematical probability space.

The model produces:

```text
P(model) = [P(cat), P(dog), P(bird), ...]
```

The target represents the desired distribution.

Cross-entropy measures the incompatibility between the model's prediction and the target distribution.

This can be conceptually combined with other geometric measurements:

```text
Prediction
    │
    ├── Probability geometry
    │      └── Cross-entropy
    │
    ├── Spatial geometry
    │      └── Distance / angle / position error
    │
    ├── Temporal geometry
    │      └── Velocity / trajectory / timing error
    │
    └── Magnitude geometry
           └── Ratios / scale / orders of magnitude
```

---

## Unified Learning Objective

A geometry-aware AI can combine multiple error types:

$$
L =
\lambda_1L_{classification}
+\lambda_2L_{position}
+\lambda_3L_{velocity}
+\lambda_4L_{magnitude}
+\lambda_5L_{temporal}
$$

Where:

* \(L_{classification}\) → cross-entropy / probabilistic error
* \(L_{position}\) → spatial distance error
* \(L_{velocity}\) → motion prediction error
* \(L_{magnitude}\) → scale/ratio error
* \(L_{temporal}\) → timing or sequence error
* \(\lambda_i\) → weights controlling the importance of each component

The objective is therefore not simply:

> "Was the prediction correct?"

but:

> **"How wrong was the prediction, in which dimension, and at what scale?"**

---

## Application to Game AI / VLM Systems

For a visual game AI:

```text
SCREEN / VIDEO
      ↓
VISION
      ↓
OBJECTS + FEATURES
      ↓
STRUCTURED STATE
      ↓
┌──────────────┬──────────────┬──────────────┐
│   Spatial    │  Magnitude   │ Probability  │
│   Geometry   │   Geometry   │    Space     │
├──────────────┼──────────────┼──────────────┤
│ position     │ health ratio │ action odds  │
│ distance     │ scale        │ confidence   │
│ direction    │ relative HP  │ uncertainty  │
│ velocity     │ orders       │ prediction   │
└──────────────┴──────────────┴──────────────┘
      ↓
GEOMETRY-AWARE REASONING
      ↓
ACTION
      ↓
REWARD / LOSS
      ↓
LEARNING
```

Example:

```text
Player:
position = (120,450)
health = 800

Enemy:
position = (140,470)
health = 40

Derived:
distance ≈ 28
relative direction = NE
health ratio = 20:1
enemy health = 5% of player health
```

The AI can simultaneously reason about:

* **where** the enemy is,
* **how far** away it is,
* **how it is moving**,
* **how much stronger/weaker** each entity is,
* **how confident** the model is,
* and **which action has the highest expected value**.

---

## Key Design Principle

Do not force the reasoning model to rediscover every relationship from raw observations.

Prefer:

```text
RAW DATA
   ↓
PERCEPTION
   ↓
RELATIONSHIPS
   ↓
GEOMETRIC REPRESENTATION
   ↓
SCALE / MAGNITUDE REPRESENTATION
   ↓
PROBABILITY / UNCERTAINTY
   ↓
REASONING
   ↓
ACTION
```

The overall goal is to move from **isolated values and tokens** toward **structured, scale-aware, probabilistic relationships**.

Cross-entropy can therefore serve as the probabilistic component of a broader geometric learning framework rather than being treated as an isolated classification loss.

