"""Hospital Neural Game Engine (self-contained).

The MDP tracks game state; the world model (rich+LPIPS conv-UNet + Qwen VAE) RENDERS
the board frame on the GPU; pygame draws the HUD/effects and handles input.

  arrow keys  = move (W/A/S/D -> up/left/down/right)
  space       = act on object in current cell (treat patient / etc.)
  R           = reset,   Esc = quit

Frame flow per move:
  key -> MDP computes next state (px,py, HUD ints) -> composite CLEAN input frame
  (actor@cell + objects + HUD strip) -> model renders next board -> pygame shows it
  with the deterministic HUD/effects overlay. Inputs are always clean composites
  (never the model's own output) -> no drift.

Runs on any CUDA GPU. ~1.5-2.4 s per rendered frame (turn-based, not real-time).

    python engine.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
# bundled DIAMOND diffusion code (self-contained, no external repo needed)
sys.path.insert(0, str(HERE / "model" / "diamond_src"))

import numpy as np
import torch
import pygame
from PIL import Image

from latent_wm import load_vae, encode, decode
from rich_wm import make_rich_denoiser, RichFeatures, build_heatmap, load_geom

ASSETS = HERE / "assets"
GAME = "hospital"
CKPT = HERE / "model" / "denoiser_final.pt"
SIGMA = 0.612
DENOISE = 20            # 20 steps ~1.5s/frame (30 = 2.4s, crisper)
VIEW = 560


# ---- MDP: hospital rules (from RDDL, minimal) --------------------------------
class HospitalMDP:
    """Tracks player cell + HUD counters. Rules derived from the RDDL."""
    def __init__(self):
        self.grid = json.load(open(ASSETS / "meta.json"))["grid"]  # [cols, rows]
        self.reset()

    def reset(self):
        self.px, self.py = 1, 1
        # ints = [patient_treated, staff_energy, resource_stock, total_score]
        self.ints = [0, 0, 0, 0]
        # patient cells (RDDL): patient_1@(2,2), patient_2@(3,2)
        self.patient_cells = {(2, 2), (3, 2)}
        self.done = False

    def step(self, key):
        """key in W/A/S/D/Space. Returns (action_id, effects_dict) or None if noop."""
        cols, rows = self.grid
        moved = None
        if key == "W" and self.py < rows:
            moved = ([self.px, self.py], [self.px, self.py + 1]); self.py += 1
        elif key == "S" and self.py > 1:
            moved = ([self.px, self.py], [self.px, self.py - 1]); self.py -= 1
        elif key == "A" and self.px > 1:
            moved = ([self.px, self.py], [self.px - 1, self.py]); self.px -= 1
        elif key == "D" and self.px < cols:
            moved = ([self.px, self.py], [self.px + 1, self.py]); self.px += 1
        elif key == "Space" and (self.px, self.py) in self.patient_cells:
            if self.ints[3] < 8:                      # treat -> +1 treated, +1 score
                before = list(self.ints)
                self.ints[0] += 1; self.ints[3] += 1
                if self.ints[3] >= 8:
                    self.done = True
                return 4, {"moved_player": None, "target": "patient_1",
                           "int_changes": {"patient_treated": [before[0], self.ints[0]],
                                           "total_score": [before[3], self.ints[3]]},
                           "completed_object": "patient_1", "hidden_object": None,
                           "bool_flips": []}
            return None
        else:
            return None                               # blocked / invalid
        act = {"W": 0, "S": 1, "A": 2, "D": 3}[key]
        return act, {"moved_player": {"from": moved[0], "to": moved[1]},
                     "int_changes": {}, "completed_object": None,
                     "hidden_object": None, "bool_flips": []}


# ---- Compositor: clean input frame from MDP state ----------------------------
import re


def rddl_positions():
    inst = next((ASSETS / "source").glob("*instance*.rddl")).read_text()
    pos = {}
    for m in re.finditer(r"(\w+?)-([xy])\((\w+),\s*[xy](\d+)\)", inst):
        _, axis, obj, idx = m.group(1), m.group(2), m.group(3), int(m.group(4))
        pos.setdefault(obj, {})[axis] = idx
    return {o: (xy["x"], xy["y"]) for o, xy in pos.items() if "x" in xy and "y" in xy}


class Compositor:
    def __init__(self):
        self.geom = load_geom_local()
        self.pos = rddl_positions()
        self.ref = Image.open(ASSETS / "state_ref.png").convert("RGBA")
        self.W, self.H = self.ref.size
        self.hud_h = 82
        self.ground = Image.open(ASSETS / "themed_ground.png").convert("RGBA")
        self.cell_px = int(abs(self.geom["cell"][0]) * 0.62)

    def cell_center(self, col, row):
        ox, oy = self.geom["origin"]; cw, ch = self.geom["cell"]
        return ox + (col - 1) * cw, oy + (row - 1) * ch

    def _paste(self, canvas, sprite, cx, cy):
        sp = Image.open(sprite).convert("RGBA")
        s = self.cell_px / sp.width
        w, h = int(sp.width * s), int(sp.height * s)
        sp = sp.resize((w, h), Image.LANCZOS)
        canvas.alpha_composite(sp, (int(cx - w / 2), int(cy - h * 0.6)))

    def frame(self, px, py, ints, hidden=()):
        canvas = Image.new("RGBA", (self.W, self.H))
        # HUD strip matching the counters
        hp = ASSETS / "hud_strips" / ("hud_" + "_".join(map(str, ints)) + ".png")
        hud = Image.open(hp).convert("RGBA") if hp.exists() else self.ref.crop((0, 0, self.W, self.hud_h)).convert("RGBA")
        canvas.paste(hud, (0, 0))
        canvas.paste(self.ground.resize((self.W, self.H - self.hud_h)).convert("RGBA"), (0, self.hud_h))
        player_obj = next((o for o in self.pos if "player" in o), None)
        for obj, (c, r) in self.pos.items():
            if obj == player_obj or obj in hidden:
                continue
            sp = ASSETS / "sprites" / f"{obj}.png"
            if sp.exists():
                self._paste(canvas, sp, *self.cell_center(c, r))
        if player_obj:
            self._paste(canvas, ASSETS / "sprites" / f"{player_obj}.png", *self.cell_center(px, py))
        return canvas.convert("RGB")


def load_geom_local():
    return json.loads((ASSETS / "geometry.json").read_text())


# ---- Model wrapper -----------------------------------------------------------
class Renderer:
    def __init__(self, device):
        self.device = device
        self.vae = load_vae(device)
        self.rf = RichFeatures.__new__(RichFeatures)   # build vocab from local transitions
        self._init_rf()
        self.geom = json.loads((ASSETS / "geometry.json").read_text())
        self.net = make_rich_denoiser(device, 5, len(self.rf.targets), len(self.rf.hud_keys),
                                      cond_channels=512, sigma_data=SIGMA)
        self.net.load_state_dict(torch.load(CKPT, map_location=device)); self.net.eval()
        from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
        self.sampler = DiffusionSampler(self.net, DiffusionSamplerConfig(
            num_steps_denoising=DENOISE, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    def _init_rf(self):
        trans = [json.loads(l) for l in open(ASSETS / "transitions.jsonl")]
        self.rf.actions = sorted({t["action_name"] for t in trans})
        self.rf.targets = sorted({t["target"] for t in trans if t["target"]}) + ["<none>"]
        keys = set()
        for t in trans:
            keys |= set(t["effects"].get("int_changes", {}).keys())
        self.rf.hud_keys = sorted(keys) + ["<none>"]
        self.rf.a2i = {a: i for i, a in enumerate(self.rf.actions)}
        self.rf.t2i = {t: i for i, t in enumerate(self.rf.targets)}
        self.rf.h2i = {h: i for i, h in enumerate(self.rf.hud_keys)}

    def render(self, clean_input, action_id, effects):
        lat = encode(self.vae, clean_input, 1024, self.device).unsqueeze(1)
        from rich_wm import build_heatmap
        self.net.heatmap = build_heatmap(self.geom, effects, self.device).unsqueeze(0)
        # rich features from effects
        ic = effects.get("int_changes", {})
        hud_key, hud_delta = "<none>", 0.0
        for k, (a, b) in ic.items():
            hud_key, hud_delta = k, float(b - a)
            if k != "total_score":
                break
        tgt = self.rf.t2i.get(effects.get("target") or "<none>", self.rf.t2i["<none>"])
        self.net.inner_model.rich = dict(
            target=torch.tensor([tgt], device=self.device),
            hud=torch.tensor([self.rf.h2i.get(hud_key, self.rf.h2i["<none>"])], device=self.device),
            hud_delta=torch.tensor([hud_delta], device=self.device, dtype=torch.float32))
        act = torch.tensor([[action_id]], device=self.device)
        torch.manual_seed(0)
        with torch.no_grad():
            return decode(self.vae, self.sampler.sample(lat, act)[0])


# ---- Pygame loop -------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU found. CPU is ~100+ s/frame (impractical).")
    print("loading model + VAE (first run downloads the Qwen VAE)...")
    comp = Compositor()
    rend = Renderer(device)
    mdp = HospitalMDP()

    pygame.init()
    screen = pygame.display.set_mode((VIEW, VIEW + 40))
    pygame.display.set_caption("Hospital Neural Engine")
    font = pygame.font.SysFont("dejavusansmono", 15)
    clock = pygame.time.Clock()

    def render_current():
        clean = comp.frame(mdp.px, mdp.py, mdp.ints)
        return clean

    # initial frame = the composited start state (model renders on first move)
    cur_img = render_current()
    status = "arrows=move  space=treat  R=reset  Esc=quit"
    KEYMAP = {pygame.K_UP: "W", pygame.K_DOWN: "S", pygame.K_LEFT: "A",
              pygame.K_RIGHT: "D", pygame.K_SPACE: "Space"}

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_r:
                    mdp.reset(); cur_img = render_current(); status = "reset"
                elif e.key in KEYMAP:
                    res = mdp.step(KEYMAP[e.key])
                    if res is None:
                        status = "blocked / no action here"
                    else:
                        act_id, effects = res
                        # CLEAN composited input for THIS transition = current state frame
                        clean_in = comp.frame(
                            effects["moved_player"]["from"][0] if effects.get("moved_player") else mdp.px,
                            effects["moved_player"]["from"][1] if effects.get("moved_player") else mdp.py,
                            [mdp.ints[0] - (1 if act_id == 4 else 0), mdp.ints[1], mdp.ints[2],
                             mdp.ints[3] - (1 if act_id == 4 else 0)])
                        t0 = time.time()
                        cur_img = rend.render(clean_in, act_id, effects)
                        status = (f"score {mdp.ints[3]}/8   rendered in {time.time()-t0:.1f}s"
                                  + ("   YOU WIN!" if mdp.done else ""))

        # draw
        surf = pygame.image.frombuffer(
            np.asarray(cur_img.convert("RGB").resize((VIEW, VIEW))).tobytes(),
            (VIEW, VIEW), "RGB")
        screen.fill((17, 24, 33))
        screen.blit(surf, (0, 0))
        screen.blit(font.render(status, True, (231, 238, 245)), (8, VIEW + 12))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
