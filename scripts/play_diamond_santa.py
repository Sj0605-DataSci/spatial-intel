"""Neural game emulator: play Santa's Gift Journey where EVERY frame is rendered
by the trained DIAMOND world model (not the game engine).

You press an arrow key -> the action + current frame go into DIAMOND's EDM sampler
-> it generates the next frame -> that frame becomes the new state (closed loop).
This is a GameNGen-style neural engine: no rules code, the model IS the game.

Honest scope: the model was trained on only 6 transitions near cells (1,1)/(1,2)/
(1,3). It plays those learned paths faithfully; steer off them and it smears /
hallucinates. The HUD flags when your move is a LEARNED transition vs. unseen.

Controls:
  arrows      move (up/down/left/right)
  R           reset to the start frame
  [ / ]       fewer / more denoise steps (speed <-> quality)
  H           toggle: feed model's own prediction back (closed loop) vs re-seed
  Esc / Q     quit

    python scripts/play_diamond_santa.py --ckpt runs/diamond_santa/denoiser_final.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import pygame
from PIL import Image

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from train_diamond_santa import make_denoiser, load_img, ACTIONS, DATA

# --- the learned transition graph (cell -> {action: next_cell}), from the 6 moves ---
LEARNED = {
    (1, 1): {"right": (2, 1), "down": (1, 2)},
    (1, 2): {"right": (2, 2), "up": (1, 1), "down": (1, 3)},
    (1, 3): {"right": (2, 3)},
}
KEY_TO_ACTION = {
    pygame.K_UP: "up", pygame.K_DOWN: "down",
    pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
}
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# theme
BG = (14, 22, 32); PANEL = (22, 33, 46); INK = (231, 238, 245)
MUTED = (147, 164, 182); BADGE = (198, 40, 40)
HIT = (46, 125, 91); MISS = (192, 135, 58); SLATE = (95, 114, 133)


def to_uint8(chw):
    return chw.clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()


def frame_to_surface(arr64, view_px):
    """(H,W,3) uint8 -> pygame Surface upscaled to view_px (Lanczos for viewing)."""
    im = Image.fromarray(arr64).resize((view_px, view_px), Image.LANCZOS)
    return pygame.image.frombuffer(np.asarray(im).tobytes(), (view_px, view_px), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/diamond_santa/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--view", type=int, default=512, help="on-screen frame size")
    ap.add_argument("--steps", type=int, default=5, help="denoise steps (3-5 = real-time)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_denoiser(args.res, dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev))
    net.eval()

    def make_sampler(steps):
        return DiffusionSampler(net, DiffusionSamplerConfig(
            num_steps_denoising=steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))
    sampler = make_sampler(args.steps)
    steps = args.steps

    # start frame = the (1,1) before-frame
    man = json.loads((DATA / "manifest.json").read_text())
    start_row = next(r for r in man["test"] if tuple(r["start"]) == (1, 1))
    start_frame = load_img(DATA / start_row["input"], args.res, dev)  # (C,H,W)

    def reset():
        return start_frame.clone(), (1, 1), None, 0.0, True
    cur, cell, last_act, last_ms, closed_loop = reset()
    history = [(cell, "start")]

    @torch.no_grad()
    def step(frame, action):
        prev = frame.unsqueeze(0).unsqueeze(0)                 # (1,1,C,H,W)
        act = torch.tensor([[ACTIONS[action]]], device=dev)
        torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.time()
        pred, _ = sampler.sample(prev, act)                    # (1,C,H,W)
        torch.cuda.synchronize() if dev == "cuda" else None
        return pred[0], (time.time() - t0) * 1000

    pygame.init()
    W = args.view + 320
    screen = pygame.display.set_mode((W, args.view + 40))
    pygame.display.set_caption("DIAMOND · Santa neural engine")
    font = pygame.font.SysFont("dejavusansmono", 15)
    big = pygame.font.SysFont("dejavusans", 22, bold=True)
    small = pygame.font.SysFont("dejavusansmono", 12)
    clock = pygame.time.Clock()

    cur_arr = to_uint8(cur)
    running = True
    while running:
        moved = None
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_r:
                    cur, cell, last_act, last_ms, closed_loop = reset()
                    cur_arr = to_uint8(cur); history = [(cell, "start")]
                elif e.key == pygame.K_h:
                    closed_loop = not closed_loop
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    steps = max(1, steps + (1 if e.key == pygame.K_RIGHTBRACKET else -1))
                    sampler = make_sampler(steps)
                elif e.key in KEY_TO_ACTION:
                    moved = KEY_TO_ACTION[e.key]

        if moved:
            learned = moved in LEARNED.get(cell, {})
            src = cur if closed_loop else start_frame  # closed loop feeds own output
            new_frame, last_ms = step(src if closed_loop else cur, moved)
            cur = new_frame; cur_arr = to_uint8(cur); last_act = moved
            # update believed cell via the learned graph (falls back to delta)
            if learned:
                cell = LEARNED[cell][moved]
            else:
                dx, dy = DELTA[moved]
                cell = (max(1, min(8, cell[0] + dx)), max(1, min(8, cell[1] + dy)))
            history.append((cell, moved))
            history = history[-8:]

        # ---- draw ----
        screen.fill(BG)
        surf = frame_to_surface(cur_arr, args.view)
        screen.blit(surf, (20, 20))
        pygame.draw.rect(screen, SLATE, (20, 20, args.view, args.view), 1)

        x = args.view + 40
        def line(txt, y, color=INK, f=font):
            screen.blit(f.render(txt, True, color), (x, y))

        line("NEURAL ENGINE", 24, SLATE, small)
        line("Santa · DIAMOND", 40, INK, big)
        line(f"believed cell  ({cell[0]},{cell[1]})", 90, INK)
        line(f"last action    {last_act or '—'}", 112, INK)
        avail = ", ".join(LEARNED.get(cell, {}).keys()) or "none"
        line("learned from here:", 146, SLATE, small)
        line(avail, 162, HIT if avail != "none" else MISS)
        # perf
        fps = 1000 / last_ms if last_ms else 0
        line(f"denoise steps  {steps}   [ / ] adjust", 200, MUTED, small)
        line(f"gen time       {last_ms:5.1f} ms", 220, INK)
        line(f"model fps      {fps:5.1f}", 240, HIT if fps >= 24 else MUTED)
        line(f"loop  {'closed (own output)' if closed_loop else 'open (re-seed)'}", 262, MUTED, small)
        # warn if off learned paths
        if last_act and last_act not in LEARNED.get(history[-2][0] if len(history) > 1 else cell, {}):
            line("⚠ unseen move — model may smear", 288, MISS, small)
        # path history
        line("path:", 320, SLATE, small)
        for i, (c, a) in enumerate(history):
            line(f"  {a:6} -> ({c[0]},{c[1]})", 336 + i * 16, MUTED, small)
        # controls
        cy = args.view - 40
        for j, t in enumerate(["arrows move   R reset", "[ ] steps   H loop   Esc quit"]):
            line(t, cy + j * 18, SLATE, small)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
