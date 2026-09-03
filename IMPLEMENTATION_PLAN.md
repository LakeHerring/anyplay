# Implementation Plan — Qwen3.5-4B VL Teacher + RL Student

Maps the target architecture (VL teacher → constrained actions → experience
stream → RL student) onto the existing codebase, with the revised language
boundary:

> **C++ is the game environment/runtime; Python is the AI laboratory.**
>
> ```
> C++ CORE:  capture · frame buffer · timing · input capture ·
>            input injection · IPC/shm · environment state
> PYTHON:    observation · Qwen3.5-4B VL · PyTorch/ROCm · RL ·
>            training · reward · datasets · experiments
>
> C++ → Python : raw observations (shm, no JSON, no copies if possible)
> Python → C++ : action decisions (small IPC messages)
> ```

Qwen lives in Python only. No image bytes ever cross the IPC boundary as
serialized data — frames travel in the shared-memory ring, control messages
are tiny JSON.

> **Scope:** the system targets *any* capturable, key/mouse-driven game.
> Shadow Dungeon is only the **first reference environment** — its reward
> function, terminal-state detection, and VL action vocabulary are per-game
> adapters; capture, observation, policy, and trainers are game-agnostic.

## 0. Current state vs. target

| Target component | Status |
|---|---|
| 60 FPS capture (portal/PipeWire → GStreamer → **C** → shm ring → NumPy) | **[have, C]** `native/capture_daemon.c`, `capture/native_capture.py` |
| Input capture (evdev → `inputs.jsonl`) | **[have, Python]** `capture/input_recorder.py` → moves to C++ core |
| Input injection (UInput virtual device) | **[have, Python]** `game_controller.py` → execution moves to C++ core |
| Qwen3.5-4B VL (GGUF + mmproj via llama.cpp, OpenAI API) | **[have]** `vl/vl_model.py` — single image, free-text out |
| Small CNN+GRU policy + BC trainer + live play | **[have]** `training/`, `game_integration/` |
| **C++ core** (unified env runtime: capture + input + timing + IPC) | **[new]** — port/extend the working C daemon, keep shm protocol stable |
| Timestamped temporal observation buffer w/ input state | **[new, Python]** |
| Multi-rate architecture (60 cap / 30 policy / 2–5 Hz VL) | **[new, Python loop + C++ rates]** |
| Constrained action vocabulary + structured VL output + parser | **[new, Python]** |
| Action controller (duration, conflict rules) + C++ execution backend | **[new, both]** |
| Reward extraction | **[new, Python]** |
| Experience stream + teacher dataset | **[new, Python]** |
| RL trainer | **[new, Python]** |
| Spatial world model (detection → tracking → geometry → scale-relative state) | **[new, Python]** — P-wm |

**Progress (2026-09-03):** P0–P3 done. P-core: step 1 (C++ daemons, X11 +
portal) and step 2 (evdev input ring in both daemons + Python `InputEventRing`
client, dev_id-tagged slots) done — regression test
(`core/test/regression_input.py`) passing, `main.py input-check` CLI added;
step 3 (C++ uinput executor) next. P4: single-image VL decision works (free-text);
strict-JSON hardening pending. P5: agent loop runs live; 5 teacher sessions
recorded (`datasets/vespera2`–`vespera5`); the 10-minute exit criterion is
outstanding.

**Migration principle:** the existing C daemon's shm ring protocol
(magic/version/dims/slots header, slot ring, `alive` flag) is already proven
with a zero-copy Python client. The C++ core keeps that protocol *compatible*,
extends it (input-event ring, action-command region, per-stage timing), and
is built behind the same daemon socket. Python's `native_capture.py` keeps
working during the port; nothing is rewritten blind.

## 1. New layout

### C++ core (`core/`, CMake)

```
core/
    CMakeLists.txt
    include/
        shm_protocol.h        # THE contract: ring header, frame slot, event
                              # slot, action command, timing fields — shared
                              # verbatim with Python (generated or hand-synced)
        frame.h               # struct Frame { ts, w, h, format, data ptr }
    src/
        main.cpp              # daemon entry: parse config, start threads
        capture/
            portal_source.cpp # xdg-desktop-portal + PipeWire (port of
            x11_source.cpp    #   portal_capture.c / ximagesrc path)
            gst_pipeline.cpp  # GStreamer native C API from C++ (no wrapper
                              #   libs), appsink callback → frame buffer
        frame/
            ring.cpp          # lock-free SPSC shm ring (extends current one:
                              #   adds timestamped history, drop counter,
                              #   per-frame latency stamp)
        input/
            evdev_reader.cpp  # read /dev/input/*, normalize events, push into
                              #   input-event shm ring (replaces Python
                              #   input_recorder for live play)
            uinput_writer.cpp # virtual device, executes action commands
                              #   (replaces Python GameController execution)
        action/
            executor.cpp      # receives ActionCmd over IPC, schedules
                              #   press/hold/release, enforces exclusivity +
                              #   duration, guarantees release-all on shutdown
        timing/
            clock.cpp         # monotonic clock, deadline scheduler,
                              #   per-stage latency stamps into the ring
        ipc/
            control.cpp       # small unix socket: JSON commands
                              #   {start_action, release_all, set_vocab,
                              #    get_state, episode events} — metadata only,
                              #   never pixels
        env/
            episode.cpp       # episode lifecycle: start/stop, death/respawn
                              #   hooks, re-acquire portal stream, counters
    build.sh
```

**Ownership boundary (hard rule):** the C++ core contains no model, no
prompt, no reward, no policy logic. It receives `ActionCmd` and emits raw
frames + input events. It *can* know the key layout of the virtual device
(execution detail) but never the semantics of an action name beyond what the
Python side configures via `set_vocab` (name → key sequence + duration).

### Python (`anyplay/`)

```
anyplay/
    capture/
        native_capture.py     # [extend] same shm client; add: input-event
                              #   ring reader, action IPC client (thin)
    agent/                    # [new]
        observation.py        # Frame/Observation dataclasses + ObservationBuffer
        action_space.py       # GameAction vocabulary (YAML), compiles to
                              #   C++ set_vocab payload
        action_controller.py  # policy-rate Action(name, duration) → IPC
                              #   (validates, rate-limits, dedupes)
        reward.py             # reward extractor (pure functions on game state)
        experience.py         # Experience dataclass + ExperienceLogger
        loop.py               # multi-rate agent loop (P5)
        metrics.py            # per-stage latency + dropped-frame tracking
    teacher/                  # [new]
        prompt.py             # system prompt + vocabulary injection
        client.py             # VLModel wrapper: multi-image obs → structured action
        parser.py             # strict JSON parser, rejects unknown actions
        collect.py            # teacher data collection driver
    student/                  # [new]
        distill.py            # BC on teacher dataset (reuses training.models)
        rl/
            env.py            # Gym-like env over episodes / live core
            ppo.py            # PPO (first RL trainer)
            buffers.py
    perception/               # [new, P-wm] detection → tracking → geometry →
                              #   scale-relative spatial state (10–60 Hz)
    (existing: vl/, training/, game_integration/, evolution/, utils/)
```

`game_integration/game_controller.py` becomes a thin adapter over the C++
executor during transition (its `write_action` interface stays, so
`main.py play` works before and after the port).

CLI additions in `main.py`:

```
python main.py core-status                  # daemon health, ring stats, latency
python main.py obs-check                     # P1: buffer extraction test
python main.py input-check                   # P-core: evdev ring + injection test
python main.py vl-decide --image shots/x.png
python main.py agent --policy teacher --actions left,right,wait
python main.py agent --policy student --checkpoint ...
python main.py teacher-collect --duration 1800
python main.py reward-check --session run1
python main.py train-rl --algo ppo
python main.py eval --policy X
```

## 2. Phases

### P0 — Instrumentation [new]
`agent/metrics.py` + C++ timing stamps: every stage logs
`stage, ts, duration_ms` to `metrics.jsonl`. Stages: `capture` (source →
ring), `transfer` (shm → NumPy), `preprocess`, `vl_request`, `vl_parse`,
`action_ipc`, `input_apply`. Dropped-frame counter from the ring (free).
**Exit:** `play --smoke` emits per-stage latencies; end-to-end capture
latency (source timestamp → Python receipt) measured.

### P-core — C++ core (parallel track, unblocks nothing in Python) [new]
Incremental, each step keeps the old path working:

1. **CMake scaffold + port the daemon.** Move `capture_daemon.c` +
   `portal_capture.c` logic into `core/` (C++ classes, GStreamer native C
   API). Shm protocol unchanged; `native_capture.py` runs against the new
   binary as the regression test (frame count, dims, fps, zero-copy).
2. **Input-event ring.** `evdev_reader.cpp` reads the configured keyboard +
   pointer nodes, writes normalized events (`ts, type, code, value`) into a
   second shm ring. Python client reads them (replaces the live
   `InputRecorder` thread; the offline `inputs.jsonl` writer stays in Python
   for recording sessions).
3. **Action executor.** `uinput_writer.cpp` + `executor.cpp`: `set_vocab`
   loads name→keys+duration+exclusivity; `start_action`/`release_all`
   commands; guaranteed release on SIGTERM/shutdown.
4. **Episode hooks.** `env/episode.cpp`: counters (frames, events, actions,
   drops), death/respawn re-acquire of the portal stream (risk #3),
   `core-status` reporting.

**Exit:** `main.py input-check` — record 10 s of real key presses via the
C++ ring (compared against a parallel `evtest`), then have the core inject a
scripted sequence and verify it with `evtest` within ±1 ms of schedule. Old
Python input path still works (feature flag) until P5 completes.

### P1 — Timestamped temporal observation buffer [new, Python]
`agent/observation.py`:

```python
@dataclass
class Frame:        ts: float; id: int; rgb: np.ndarray  # (H,W,3) uint8
@dataclass
class Observation:
    ts: float
    frames: list[Frame]           # e.g. t-400ms, t-200ms, t
    input_state: dict             # currently pressed keys/buttons (from
                                  # C++ event ring)
    last_action: str | None
    game_state: dict | None       # optional reliable state

class ObservationBuffer:
    def __init__(self, frame_count=3, offsets_ms=(400, 200, 0)): ...
    def push(self, ts, frame) -> None
    def sample(self, ts=None) -> Observation
```

All rates in `config.py` (`ObsConfig(frame_count, offsets_ms, cap_fps,
policy_fps, vl_hz)`) — nothing hardcoded.
**Exit:** `main.py obs-check` dumps each Observation (frames side by side +
input state) with measured inter-frame gaps; no Qwen involved.

### P2 — Constrained action space + parser [new, Python]
`agent/action_space.py`: per-game YAML:

```yaml
# games/shadow_dungeon.yaml
actions:
  LEFT:    {keys: [KEY_A], duration_ms: 0}          # hold while active
  RIGHT:   {keys: [KEY_D], duration_ms: 0}
  JUMP:    {keys: [KEY_SPACE], duration_ms: 120}
  ATTACK:  {keys: [KEY_J], duration_ms: 90}
  WAIT:    {keys: []}
  DODGE_LEFT: {keys: [KEY_SHIFT_L, KEY_A], duration_ms: 180}
exclusive_groups: [[JUMP, ATTACK], ...]
```

This file is the *single source of truth*: Python parses it for validation,
and `set_vocab` compiles it into the C++ executor.
`vl/parser.py`: strict parser — expects `{"action": ..., "confidence": ...}`,
rejects unknown actions, one retry with the error in the prompt, then
`WAIT`. Unit-tested against 20+ malformed outputs.
**Exit:** `vl-decide` returns only vocabulary actions; 0 invalid actions
reach the executor in a 100-call fuzz test.

### P3 — Action controller [new, both]
Python `action_controller.py` (policy-rate, validates + rate-limits +
dedupes) → C++ executor (timing, exclusivity, release-all). The C++ side is
the only thing that touches `/dev/uinput` from now on.
**Exit:** scripted sequence through the full Python→IPC→C++ path; `evtest`
log shows exact press/release timing; Ctrl-C and crash paths both release
all keys.

### P4 — VL decision client (offline) [extend, Python]
`teacher/client.py` wraps `VLModel.ask()`:
- multi-image: extend `VLModel` with `images: list[Path]` (N `image_url`
  parts), or one horizontal strip image (token-cost benchmark both)
- frames downscaled to ~448px JPEG
- system prompt fixes vocabulary + JSON-only output
- measures: latency, VRAM, tokens in/out, action distribution
**Exit:** `vl-decide` on 50 curated screenshots; latency histogram +
consistency report (same image 5×, action agreement).

### P5 — Closed-loop agent (first live milestone) [new]
`agent/loop.py`, multi-rate:

```
C++ core (its own threads, RT-friendly):
    60 Hz capture → shm frame ring
    evdev → shm event ring
Python:
    thread A:  ring reader → buffer.push(ts, frame)
    thread B:  event ring → input_state
    thread C:  buffer.sample() → policy.decide(obs) → controller.apply()
               TeacherPolicy: VL at 2–5 Hz, holds last action between calls
               StudentPolicy: existing CNN+GRU predict()
```

Start with `actions: [LEFT, RIGHT, WAIT]`. VL calls never block capture or
input (different processes — Rule 7 is now *structurally* guaranteed).
**Exit:** 10-minute live session, game never stalls, `metrics.jsonl` clean,
character visibly moves, 0 crashes, old Python input path retired.

### P6 — Reward system [new, Python]
`agent/reward.py`: pure functions, testable against recorded sessions.
The reward function is the **per-game adapter** — the interface is
game-agnostic, each game supplies its own reward extractor. First
implementation, Shadow Dungeon (640×480 HUD): pixel-parse health bar,
score digits, death
screen (template/color detection first — cheap; VL *only* for what pixels
can't read, e.g. objective text).
Rewards: `+1` room cleared, `+0.1`/s survival, `-5` death, `-0.01`
no-progress (>10 s same position), `+0.05` item collected.
**Exit:** `reward-check --session vespera5` produces a per-frame reward curve
matching manual inspection at every room-clear and death.

### P-wm — Spatial world model (geometric + magnitude state) [new, Python, parallel from P5]
Observation-layer extension (README: *Geometric Reasoning and Spatial World
Model* + *Design rationale: geometry + magnitude → scale-invariant state*).
Perception → tracking → geometry → **scale-relative** spatial state (distance
as fraction of view radius, closing speed as view-crossing time, HP/threat as
ratios; absolute pixels for debug only). Runs at 10–60 Hz, independent of the
Qwen rate; Qwen receives the compact state, with richer visual context only
when geometry confidence is low (adaptive reasoning budget).

Incremental, each step measured before the next (per README *Development
priority*):

1. Detect player + major objects (Shadow Dungeon: player, enemies, doors,
   pickups — template/color detection first; a small detector only if needed).
2. Track identity across frames.
3. Relative position / distance / bearing (player-relative, 2D top-down).
4. Velocity + motion estimation.
5. Obstacle / path relationships.
6. Persistent temporal state (last-seen, predicted trajectory).
7. Compact `spatial_state` into the Qwen prompt.
8. A/B: Qwen raw-vision vs vision+geometry — decision quality + latency.
9. Student policy on the combined visual + spatial representation.
10. Benchmark: latency, throughput, decision quality, sample efficiency.
    **If no measured benefit, stop — "measure, don't assume."**

**P7 coupling:** the experience record gains optional `spatial_state` +
`next_spatial_state` fields (null until step 7 provides them), so teacher
datasets collected after step 7 carry them without a format change.
**Exit:** steps 1–7 on live + recorded frames; step-8 report exists and the
phase is kept or killed on evidence.

### P7 — Experience stream + teacher dataset [new, Python]
The agent loop emits `(obs_id, action, reward, next_obs_id, done, ts,
episode_id, policy)` as JSONL. Frames stored **once** per episode
(`frames.mp4` at 30–60 fps fed from the ring); `obs_id → (video, frame_index)`,
never pixel copies in JSONL.

```
datasets/teacher/<run_id>/
    episodes/<ep>/frames.mp4
    episodes/<ep>/experience.jsonl
    meta.json          # clocks, vocab version, reward version
    quality.json       # invalid-action rate, parse failures, reward spikes
```

`teacher/collect.py`: N episodes, auto-restart on death (core re-acquires
portal stream), quality report.
**Exit:** 50 episodes; every row decodes to valid frame pair + vocab action +
finite reward; <2% parse failures.

### P8 — Student: distill, then RL [new, Python]
1. `student/distill.py`: reuse FrameEncoder+GRU (architecture already
   matches); BC on teacher experience, reward-filtered trajectories first.
2. `student/rl/ppo.py`: PPO; env (`env.py`) runs offline (buffered episodes)
   or live (P5 loop with StudentPolicy).
3. **Evaluation is gameplay, not loss**: replay harness runs
   teacher / student-BC / student-RL over the same 20 recorded + N live
   episodes → survival time, rooms cleared, deaths. `main.py eval`.
**Exit:** student-RL ≥ student-BC on live eval; both within 2× of teacher
survival; student ≥10× faster than teacher (measured).

### P9 — Reduce Qwen to supervision [new, Python]
Hybrid policy: student acts at full rate; VL at low duty cycle (e.g. 0.5 Hz)
or on trigger (reward below baseline for T s, stuck-detection, student
confidence < τ). VL returns corrective action/goal; student executes;
corrections logged as extra training data.
**Exit:** full episode with ≤20% VL duty at equal-or-better eval; removing VL
degrades ≤10%.

## 3. Cross-cutting rules → enforcement points

| Rule | Enforcement point |
|---|---|
| 1. capture ≠ inference rate | C++ core captures at its own rate; Python consumes the ring async |
| 2. no per-frame disk I/O | shm ring; `frames.mp4` at 30 fps only for episodes, never per transition |
| 3. no free LLM → input | `vl/parser.py` + vocabulary whitelist; C++ executor *only* knows `set_vocab` names — a path that bypasses Python cannot exist |
| 4. measure all latencies | C++ timing stamps in ring header + `agent/metrics.py`; `core-status` |
| 5. model-swappable | `Policy.decide(obs) -> Action` protocol; teacher/student interchangeable |
| 6. perception/reasoning/policy/execution split | buffer / VL / policy / (IPC → C++ executor) |
| 7. async, game never stalls | VL runs in a separate process; capture thread in C++ has zero Python deps |
| 8. (new) no pixels over IPC | frames live in shm; control socket carries JSON metadata only |
| 9. (new) C++ owns execution timing, Python owns policy timing | durations/exclusivity in C++ executor; cadence in Python loop |

## 4. Language table (locked)

| Component | Language |
|---|---|
| PipeWire/portal capture | C++ (native GStreamer/GLib C APIs, no wrapper libs) |
| Frame/ring buffer, timing | C++ |
| Input capture (evdev) + injection (uinput) | C++ |
| IPC (shm data + JSON control socket) | C++ ↔ Python |
| Environment state / episode lifecycle | C++ |
| Image preprocessing, observation building | Python (move to C++ only if profiling proves it a bottleneck) |
| Spatial world model (detection/tracking/geometry, P-wm) | Python (move to C++ only if profiling proves it a bottleneck) |
| Qwen3.5-4B VL, PyTorch/ROCm, RL, training | Python |
| Datasets, rewards, experiments | Python |

## 5. Sequencing & effort

```
P0 (1 d) ─┬→ P1 (2 d) → P2 (2 d) → P4 (3 d) ─────────────┐
P-core (5–7 d, parallel) ─────────────────────────────────┤→ P5 (3 d)
                                                            │
                              P6 (3 d) ─┬→ P7 (2 d) → P8 (1–2 w) → P9 (3 d)
                                        │
                                        └─ P-wm (3–5 d, starts after P5, parallel)
```

P-core runs in parallel with P0–P4; P5 is where the tracks merge (first loop
that uses the C++ executor end-to-end). The old Python input path stays as a
feature-flagged fallback until P5 exit. P-wm starts after P5 (it needs a live
loop to measure against), runs in parallel with P6/P7, and its step-7 output
(`spatial_state`) feeds P7's datasets and the P8/P9 student representations.

## 6. Risks & open questions

1. **C→C++ port regression.** Mitigated by keeping the shm protocol
   byte-identical and using `native_capture.py` + `bench_capture.py` as the
   standing regression suite. Old binaries stay in `native/` during P-core.
2. **uinput from C++.** Straightforward (raw `/dev/uinput` ioctl or
   libevdev); needs the `input` group as today. Verify EV_REL mouse injection
   parity with the Python controller (it clones physical device caps — the
   C++ executor must replicate `_build_uinput` behavior, including
   axis-bounds denormalization, or keep axes Python-side initially and move
   only keys+buttons first).
3. **VL latency on ROCm.** llama.cpp + Qwen3.5-4B Q4_K_M on the 7900 XTX —
   benchmark in P4 before trusting the 2–5 Hz target. Fallbacks: strip-image
   encoding, 2 frames.
4. **Reward signal quality.** HUD is small at 640×480. Decide in P6 which
   pixel-parsed subset is reliable; death-screen + room-transition detection
   are the robust core; VL only for objective text.
5. **Portal stream lifetime.** Game restart/respawn may need window
   re-selection. `env/episode.cpp` owns re-acquisition (P-core step 4);
   verify early in P5.
6. **Teacher consistency.** 4B VLM at temp 0.1 still wavers — the P4
   stability check and P7 quality gate (<2% invalid) are the countermeasures;
   if too inconsistent, shrink the vocabulary before scaling collection.
7. **Clock alignment.** Portal-session `meta.json` carries
   `recorder_start_monotonic`; the C++ core should stamp its own clock
   reference in the ring header so Python can align frame-ts, event-ts, and
   reward-ts without guessing.
