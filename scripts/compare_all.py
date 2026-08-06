"""Four-way comparison: input | PLAIN | RICH | RICH+LPIPS | ground truth, per val
transition. All deterministic. Prints SSIM/LPIPS for each model.

    python scripts/compare_all.py --game hospital --sigma-data 0.612 \
        --plain runs/spec_hospital_800/denoiser_final.pt \
        --rich  runs/rich_hospital_800/denoiser_final.pt \
        --richlpips runs/rich_lpips_hospital/denoiser_final.pt --out runs/cmp_all
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_msssim import ssim as ssim_fn
import lpips as lpips_lib

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser
from rich_wm import make_rich_denoiser, RichFeatures, build_heatmap, load_geom

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
ID_KEY = {0: "W", 1: "S", 2: "A", 3: "D", 4: "Space"}
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}


def to_t(pil, size, device):
    a = np.asarray(pil.convert("RGB").resize(size)).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def fnt(s, b=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf", s)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps-denoise", type=int, default=30)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--rich", required=True)
    ap.add_argument("--richlpips", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/cmp_all")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    rf = RichFeatures(args.game); geom = load_geom(args.game)

    plain = make_latent_denoiser(device, num_actions=N_ACTIONS,
                                 cond_channels=args.cond_channels, sigma_data=args.sigma_data)
    plain.load_state_dict(torch.load(args.plain, map_location=device)); plain.eval()

    def load_rich(path):
        r = make_rich_denoiser(device, N_ACTIONS, len(rf.targets), len(rf.hud_keys),
                               cond_channels=args.cond_channels, sigma_data=args.sigma_data)
        r.load_state_dict(torch.load(path, map_location=device)); r.eval()
        return r
    rich = load_rich(args.rich)
    richl = load_rich(args.richlpips)

    scfg = dict(num_steps_denoising=args.steps_denoise, sigma_min=2e-3, sigma_max=20, rho=7, order=1)
    s_plain = DiffusionSampler(plain, DiffusionSamplerConfig(**scfg))
    s_rich = DiffusionSampler(rich, DiffusionSamplerConfig(**scfg))
    s_richl = DiffusionSampler(richl, DiffusionSamplerConfig(**scfg))

    trans = [json.loads(l) for l in open(BUNDLE / args.game / "transitions.jsonl")
             if json.loads(l).get("control") in CTRL and json.loads(l)["effects"].get("moved_player")]
    import random
    random.Random(1).shuffle(trans); trans = trans[:args.n]

    P = 230
    metrics = {"plain": dict(ssim=[], lpips=[]), "rich": dict(ssim=[], lpips=[]),
               "richl": dict(ssim=[], lpips=[])}
    rows_img = []
    for t in trans:
        obs_t = Image.open(BUNDLE / args.game / t["obs_t"]).convert("RGB")
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")
        sz = gt.size
        lat = encode(vae, obs_t, args.res, device).unsqueeze(1)
        act = torch.tensor([[CTRL[t["control"]]]], device=device)

        torch.manual_seed(0)
        with torch.no_grad():
            pp = decode(vae, s_plain.sample(lat, act)[0])
        # set rich conditioning (shared by both rich models)
        hm = build_heatmap(geom, t["effects"], device).unsqueeze(0)
        f = rf.encode(t)
        richd = dict(target=torch.tensor([f["target"]], device=device),
                     hud=torch.tensor([f["hud"]], device=device),
                     hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))
        preds = {}
        for name, sm, net in [("rich", s_rich, rich), ("richl", s_richl, richl)]:
            net.heatmap = hm; net.inner_model.rich = richd
            torch.manual_seed(0)
            with torch.no_grad():
                preds[name] = decode(vae, sm.sample(lat, act)[0])
        pr, prl = preds["rich"], preds["richl"]

        for name, pred in [("plain", pp), ("rich", pr), ("richl", prl)]:
            tp, tg = to_t(pred, sz, device), to_t(gt, sz, device)
            metrics[name]["ssim"].append(ssim_fn(tp, tg, data_range=1.0).item())
            with torch.no_grad():
                metrics[name]["lpips"].append(lpips_fn(tp*2-1, tg*2-1).item())

        strip = Image.new("RGB", (P*5 + 24, P + 40), (17, 24, 33))
        d = ImageDraw.Draw(strip)
        mp = t["effects"]["moved_player"]
        d.text((8, 6), f"{args.game}  {ID_KEY[CTRL[t['control']]]}  {tuple(mp['from'])}->{tuple(mp['to'])}",
               (231, 238, 245), fnt(13, True))
        for k, (im, cap, col) in enumerate([
                (obs_t, "input", (140, 160, 180)), (pp, "PLAIN", (198, 40, 40)),
                (pr, "RICH", (60, 110, 165)), (prl, "RICH+LPIPS", (46, 125, 91)),
                (gt, "ground truth", (140, 160, 180))]):
            x = 8 + k*(P+4)
            strip.paste(im.resize((P, P)), (x, 28))
            d.text((x, 28+P+1), cap, col, fnt(11, True))
        rows_img.append(strip)

    sheet = Image.new("RGB", (rows_img[0].width, sum(r.height for r in rows_img)), (17, 24, 33))
    y = 0
    for r in rows_img:
        sheet.paste(r, (0, y)); y += r.height
    sheet.save(out / "compare_all.png")

    def m(a): return round(float(np.mean(a)), 4)
    print("\n=== 4-WAY (deterministic) — hospital ===")
    for name, lbl in [("plain", "PLAIN     "), ("rich", "RICH      "), ("richl", "RICH+LPIPS")]:
        print(f"  {lbl}: SSIM {m(metrics[name]['ssim'])}  LPIPS {m(metrics[name]['lpips'])}")
    print("  (SSIM up=better, LPIPS down=better)")
    print(f"-> {out}/compare_all.png")


if __name__ == "__main__":
    main()
