"""Imitation-learning training loop.

Trains a Policy on one capture session: the network watches a window of
frames and must reproduce the player's controller state (buttons + axes)
for the last frame of the window.

Loss = button_weight * BCEWithLogits(button_logits, buttons)
     + axis_weight   * MSE(axes_pred, axes)

Checkpoints are written to ``<session>/checkpoints/`` with the model state
and the action space (so play can denormalize outputs and map buttons).
"""

import time
from pathlib import Path

import torch


def train_session(session_dir, epochs=10, window=4, batch_size=16, lr=1e-3,
                  train_fps=30, width=128, height=96, button_weight=1.0,
                  axis_weight=1.0, motion_weight=1.0, device="cuda",
                  action_space=None, cache_frames=True):
    from .data.dataset import VideoDataset  # local import: torch is heavy
    from .models.policy import Policy

    session_dir = Path(session_dir)
    t0 = time.perf_counter()
    ds = VideoDataset(
        session_dir, train_fps=train_fps, width=width, height=height,
        window=window, action_space=action_space, cache_frames=cache_frames,
    )
    if ds.cached_frames is not None:
        print(f"frame cache: {ds.cached_frames} base frames in "
              f"{time.perf_counter() - t0:.1f}s (decoded once, reused every epoch)")
    else:
        print(f"frame cache: unavailable, streaming per epoch "
              f"({time.perf_counter() - t0:.1f}s setup)")
    n_buttons = len(ds.action_space["buttons"])
    n_axes = len(ds.action_space["axes"])
    n_motion = len(ds.action_space.get("motion", []))
    if n_buttons == 0 and n_axes == 0 and n_motion == 0:
        raise ValueError("no actions found in this session's input log")

    if device != "cpu" and not torch.cuda.is_available():
        print(f"device {device} unavailable, falling back to cpu")
        device = "cpu"
    dev = torch.device(device)

    model = Policy(n_buttons, n_axes, n_motion).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Buttons are sparse (pressed < 50% of frames), so weight positives by
    # (neg/pos) to prevent the "always zero" collapse.
    pos_weight = None
    if n_buttons:
        duty = ds.session.button_duty(ds.action_space["buttons"])
        pos_weight = torch.tensor(
            [min(20.0, max(0.5, (1.0 - p) / (p + 1e-3)))
             for p in duty.values()],
            dtype=torch.float32,
        ).to(dev)
        print(f"button duty: {dict(zip(ds.action_space['buttons'],
               [round(v, 2) for v in duty.values()]))}")

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = torch.nn.MSELoss()

    checkpoint_dir = session_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"session : {session_dir.name}")
    print(f"actions : {n_buttons} buttons, {n_axes} axes, {n_motion} motion")
    print(f"model   : {n_params / 1e6:.2f}M params | device: {dev} | "
          f"window={window} @ {train_fps} FPS, {width}x{height}")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running, n_samples = 0.0, 0
        b_frames, b_buttons, b_axes, b_motion = [], [], [], []

        def step(frames, buttons, axes, motion):
            nonlocal running, n_samples
            frames, buttons, axes, motion = (x.to(dev) for x in
                                             (frames, buttons, axes, motion))
            out = model(frames)
            # Guard empty channels: mse_loss on a (B, 0) tensor is NaN and
            # would poison the whole loss. Sessions with no axes (or no
            # buttons, or no motion) must still train.
            loss = None
            if out["buttons"].numel():
                loss = button_weight * bce(out["buttons"], buttons)
            if out["axes"].numel():
                term = axis_weight * mse(out["axes"], axes)
                loss = term if loss is None else loss + term
            if out.get("motion") is not None and out["motion"].numel():
                term = motion_weight * mse(out["motion"], motion)
                loss = term if loss is None else loss + term
            if loss is None:
                return  # nothing trainable in this batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * frames.size(0)
            n_samples += frames.size(0)

        for sample in ds:
            b_frames.append(sample["frames"])
            b_buttons.append(sample["buttons"])
            b_axes.append(sample["axes"])
            b_motion.append(sample["motion"])
            if len(b_frames) >= batch_size:
                step(torch.stack(b_frames), torch.stack(b_buttons),
                     torch.stack(b_axes), torch.stack(b_motion))
                b_frames, b_buttons, b_axes, b_motion = [], [], [], []
        if b_frames:
            step(torch.stack(b_frames), torch.stack(b_buttons),
                 torch.stack(b_axes), torch.stack(b_motion))

        avg = running / max(n_samples, 1)
        print(f"epoch {epoch:3d}/{epochs}  loss={avg:.4f}  samples={n_samples}")
        cfg = {
            "window": window,
            "train_fps": train_fps,
            "width": width,
            "height": height,
        }
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(model, ds.action_space, checkpoint_dir / "policy_best.pt",
                            best_loss, cfg)

    save_checkpoint(model, ds.action_space, checkpoint_dir / "policy_last.pt",
                    best_loss, cfg)
    print(f"best loss: {best_loss:.4f}")
    print(f"checkpoints: {checkpoint_dir / 'policy_best.pt'}")
    return {"best_loss": best_loss, "checkpoint": checkpoint_dir / "policy_best.pt"}


def save_checkpoint(model, action_space, path, loss, config=None):
    torch.save(
        {
            "model_state": model.state_dict(),
            "action_space": action_space,
            "n_buttons": model.n_buttons,
            "n_axes": model.n_axes,
            "n_motion": getattr(model, "n_motion", 0),
            "loss": float(loss),
            "config": config or {},
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint -> (model, action_space, meta with loss/config)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    from .models.policy import Policy

    model = Policy(ckpt["n_buttons"], ckpt["n_axes"],
                   ckpt.get("n_motion", 0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {"loss": ckpt.get("loss"), "config": ckpt.get("config", {})}
    return model, ckpt["action_space"], meta
