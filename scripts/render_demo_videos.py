"""Renderiza videos de demostracion desde las trazas congeladas. No simula ni evalua nada.

Dos formatos, ambos 1920x1080 a 20 fps (la misma frecuencia de decision del entorno):
  comparativo  cuatro controladores lado a lado sobre el mismo caso, con reloj compartido
  individual   un controlador a pantalla completa

Las cinco ventanas de exito se dibujan con su banda de tolerancia y la aguja del valor
actual, que se pone roja al salirse. El medidor de omega lleva las marcas del cuanto de
0.477876 rad/s que imprime un solo pulso lateral de 0.05 s. La camara se acerca por debajo
de 6 m: cambia la escala del encuadre, nunca la geometria.

Ejemplos:
  python scripts/render_demo_videos.py --scenario test-vertical-000 --output video/vertical.mp4
  python scripts/render_demo_videos.py --scenario test-offset-008 --mode individual \\
      --methods dqn_assisted42 --output video/fallo_omega.mp4
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SS = 2
W, H = 1920, 1080
FPS = 20
OMEGA_QUANTUM = 0.477876

C = {"ground": (11, 17, 25), "panel": (18, 27, 38), "sunk": (10, 16, 23),
     "ink": (231, 238, 246), "dim": (131, 150, 168), "line": (36, 52, 68),
     "nominal": (47, 212, 189), "violation": (255, 106, 77), "flame": (255, 160, 51),
     "sky_top": (13, 24, 38), "sky_bottom": (22, 34, 47)}
LABELS = {"pid": "PID en cascada", "dqn_assisted42": "DQN asistido",
          "double_assisted42": "Double DQN asistido", "dqn_assisted43": "DQN asistido s43",
          "dqn_scratch": "DQN desde cero", "double_dqn_scratch": "Double DQN desde cero"}
CAUSE = {"success": "aterrizaje valido", "invalid_contact": "contacto fuera de tolerancia",
         "body_contact": "impacto del cuerpo", "missed_platform": "fuera de la plataforma",
         "out_of_bounds": "salio del area", "time_limit": "limite de tiempo"}
CONDITION = {"vertical": "descenso vertical", "offset": "desplazado", "tilted": "inclinado",
             "wind": "con viento", "new_dynamics": "inclinado y rapido", "wind_p40": "viento +40 N",
             "wind_m40": "viento -40 N", "wind_p20": "viento +20 N", "wind_m20": "viento -20 N",
             "wind_control": "control sin viento", "represented_l1": "cerca del suelo",
             "represented_high": "desde 8 a 13 m", "new_displacement": "muy desplazado"}
ACTIONS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/System/Library/Fonts/Supplemental"),
             Path("/Library/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
FACES = {"label": ["bahnschrift.ttf", "HelveticaNeue.ttc", "Arial Narrow.ttf", "DejaVuSans.ttf"],
         "mono": ["consola.ttf", "Menlo.ttc", "Courier New.ttf", "DejaVuSansMono.ttf"]}
_fonts = {}


def font(kind, size):
    key = (kind, size)
    if key not in _fonts:
        chosen = None
        for name in FACES[kind]:
            for directory in FONT_DIRS:
                if (directory / name).exists():
                    try:
                        chosen = ImageFont.truetype(str(directory / name), int(size * SS))
                        break
                    except OSError:
                        pass
            if chosen:
                break
        _fonts[key] = chosen or ImageFont.load_default(int(size * SS))
    return _fonts[key]


def rgb(colour):
    return C[colour] if isinstance(colour, str) else colour


def blend(front, back, alpha):
    front, back = rgb(front), rgb(back)
    return tuple(int(back[i] + (front[i] - back[i]) * alpha) for i in range(3))


def text(draw, xy, body, kind="label", size=16, fill="ink", anchor="la", spacing=0.0):
    face, colour = font(kind, size), rgb(fill)
    x, y = xy[0] * SS, xy[1] * SS
    if not spacing:
        draw.text((x, y), body, font=face, fill=colour, anchor=anchor)
        return
    width = sum(draw.textlength(ch, font=face) + spacing * SS for ch in body) - spacing * SS
    if anchor[0] == "r":
        x -= width
    elif anchor[0] == "m":
        x -= width / 2
    for ch in body:
        draw.text((x, y), ch, font=face, fill=colour, anchor="l" + anchor[1])
        x += draw.textlength(ch, font=face) + spacing * SS


def box(draw, rect, fill=None, outline=None, radius=0, width=1):
    # Con la camara siguiendo al vehiculo un tramo puede quedar fuera del panel.
    if rect[2] < rect[0] or rect[3] < rect[1]:
        return
    coords = [v * SS for v in rect]
    args = dict(fill=rgb(fill) if fill else None, outline=rgb(outline) if outline else None,
                width=max(1, int(width * SS)))
    (draw.rounded_rectangle(coords, radius=radius * SS, **args) if radius
     else draw.rectangle(coords, **args))


def line(draw, points, fill="line", width=1, dash=None):
    pts = [(p[0] * SS, p[1] * SS) for p in points]
    colour, thick = rgb(fill), max(1, int(width * SS))
    if dash is None:
        draw.line(pts, fill=colour, width=thick, joint="curve")
        return
    on, off = dash[0] * SS, dash[1] * SS
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        span = math.hypot(bx - ax, by - ay)
        if span < 1:
            continue
        ux, uy, travelled = (bx - ax) / span, (by - ay) / span, 0.0
        while travelled < span:
            end = min(span, travelled + on)
            draw.line([(ax + ux * travelled, ay + uy * travelled), (ax + ux * end, ay + uy * end)],
                      fill=colour, width=thick)
            travelled = end + off


class Camera:
    """Encuadre adaptativo: vista completa en altura, acercamiento por debajo de 6 m."""

    def __init__(self, rect, cfg, altitude, close_span=8.0, track_x=0.0):
        margin, floor = 26, 32
        usable_w = (rect[2] - rect[0]) - 2 * margin
        usable_h = (rect[3] - rect[1]) - margin - floor
        wide = min(usable_w / (2 * cfg["x_limit"]), usable_h / (cfg["ceiling"] + 1.5))
        # El acercamiento lo fija la altura visible: en paneles altos y estrechos el ancho
        # sobra y limitar por el daria un vehiculo diminuto.
        close = min(usable_h / close_span, usable_w / 5.0)
        t = max(0.0, min(1.0, (6.0 - altitude) / 4.0))
        self.zoom = t * t * (3 - 2 * t)
        self.scale = wide + self.zoom * (close - wide)
        # Al acercarse, la camara sigue parcialmente al vehiculo para que no salga del panel.
        self.ox = (rect[0] + rect[2]) / 2 - self.zoom * 0.7 * track_x * self.scale
        self.oy = rect[3] - floor
        self.rect, self.visible = rect, usable_h / max(1e-6, self.scale)

    def __call__(self, x, y):
        return (self.ox + x * self.scale, self.oy - y * self.scale)


def paste_sky(base, rect):
    x0, y0, x1, y1 = [int(v * SS) for v in rect]
    height = max(1, y1 - y0)
    strip = Image.new("RGB", (1, height))
    for row in range(height):
        f = row / max(1, height - 1)
        strip.putpixel((0, row), blend(C["sky_bottom"], C["sky_top"], f))
    base.paste(strip.resize((max(1, x1 - x0), height)), (x0, y0))


def gate_rows(cfg):
    return [("|x| horizontal", "x", -3.0, 3.0, (-cfg["success_x"], cfg["success_x"]), (), 3),
            ("vx lateral", "vx", -1.6, 1.6, (-cfg["success_vx"], cfg["success_vx"]), (), 3),
            ("vy vertical", "vy", -2.6, 0.6, (-cfg["success_vy"], 0.0), (), 3),
            ("\u03b8 inclinacion", "theta", -0.26, 0.26,
             (-cfg["success_angle"], cfg["success_angle"]), (), 4),
            ("\u03c9 angular", "omega", -0.62, 0.62,
             (-cfg["success_omega"], cfg["success_omega"]), (-OMEGA_QUANTUM, OMEGA_QUANTUM), 4)]


def draw_soft(soft, cam, cfg, state, trail, ghosts, compact, wind=0.0):
    """Todo lo translucido del panel: rejilla, fronteras, corredor, estelas y penachos."""
    rect = cam.rect
    step = 1 if cam.visible < 11 else 2 if cam.visible < 19 else 5
    for metres in range(step, int(cfg["ceiling"]) + 1, step):
        y = cam(0, metres)[1]
        if y < rect[1] + 16 or y > cam.oy - 8:
            continue
        line(soft, [(rect[0] + 10, y), (rect[2] - 10, y)], (*C["line"], 165), 1, dash=(2, 5))
        text(soft, (rect[0] + 12, y - 14), f"{metres} m", "mono", 11 if compact else 13,
             (*C["dim"], 205))

    ceil_y = cam(0, cfg["ceiling"])[1]
    if ceil_y > rect[1] + 10:
        line(soft, [(rect[0] + 8, ceil_y), (rect[2] - 8, ceil_y)], (*C["violation"], 140), 2, dash=(8, 5))
        text(soft, (rect[2] - 12, ceil_y + 6), f"techo {cfg['ceiling']:.0f} m", "mono",
             11 if compact else 13, (*C["violation"], 215), anchor="ra")
    top = max(rect[1] + 6, ceil_y)
    for edge in (cam(-cfg["x_limit"], 0)[0], cam(cfg["x_limit"], 0)[0]):
        if rect[0] < edge < rect[2]:
            line(soft, [(edge, top), (edge, cam.oy)], (*C["violation"], 110), 2, dash=(8, 5))

    corridor = min(cam.oy - rect[1] - 10, cam.scale * 6)
    sxl, sxr = cam(-cfg["success_x"], 0)[0], cam(cfg["success_x"], 0)[0]
    soft.rectangle([sxl * SS, (cam.oy - corridor) * SS, sxr * SS, cam.oy * SS],
                   fill=(*C["nominal"], 24))
    text(soft, ((sxl + sxr) / 2, cam.oy - corridor + 5), f"\u00b1{cfg['success_x']:.1f} m", "mono",
         11 if compact else 13, (*C["nominal"], 185), anchor="ma")

    for ghost in ghosts:
        pts = [cam(gx, gy) for gx, gy in ghost]
        if len(pts) > 1:
            line(soft, pts, (*C["dim"], 48), 1 if compact else 2)
    for index in range(1, len(trail)):
        line(soft, [cam(*trail[index - 1]), cam(*trail[index])],
             (*C["nominal"], int(20 + 150 * index / len(trail))), 2 if compact else 3)

    if wind:
        arrows = min(5, max(2, int(abs(wind) / 10)))
        sign = 1 if wind > 0 else -1
        base_x = rect[0] + 58 if sign > 0 else rect[2] - 26
        for i in range(arrows):
            y = rect[1] + 56 + i * ((cam.oy - rect[1] - 150) / max(1, arrows - 1))
            tip = base_x + 20 * sign
            line(soft, [(base_x, y), (tip, y)], (*C["dim"], 190), 2)
            line(soft, [(tip, y), (tip - 7 * sign, y - 4)], (*C["dim"], 190), 2)
            line(soft, [(tip, y), (tip - 7 * sign, y + 4)], (*C["dim"], 190), 2)
        text(soft, (base_x, rect[1] + 30), f"viento {wind:+.0f} N", "mono",
             11 if compact else 14, (*C["dim"], 220), anchor="la" if sign > 0 else "ra")

    jets = ACTIONS[int(state["action"])]
    if state["gas"] > 0 and any(jets):
        cos, sin = math.cos(state["theta"]), math.sin(state["theta"])
        px, py = cam(state["x"], state["y"])
        k = cam.scale
        def local(lx, ly):
            return (px + (lx * cos - ly * sin) * k, py - (lx * sin + ly * cos) * k)
        mounts = [(0.0, cfg["main_thrust"]), (-0.6, cfg["side_thrust"]), (0.6, cfg["side_thrust"])]
        for active, (mx, thrust) in zip(jets, mounts):
            if not active:
                continue
            full = (thrust / cfg["main_thrust"]) * 1.7
            for layer, (fraction, alpha) in enumerate([(1.0, 55), (0.6, 110), (0.28, 185)]):
                root = local(mx, -cfg["height"] / 2)
                tip = local(mx, -cfg["height"] / 2 - full * fraction)
                wide = (0.2 - 0.045 * layer) * k
                soft.polygon([((root[0] - wide * cos) * SS, (root[1] - wide * sin) * SS),
                              ((root[0] + wide * cos) * SS, (root[1] + wide * sin) * SS),
                              (tip[0] * SS, tip[1] * SS)], fill=(*C["flame"], alpha))


def draw_solid(draw, cam, cfg, state, compact):
    """Suelo, plataforma y vehiculo: opacos, encima de la capa translucida."""
    rect = cam.rect
    lx, rx = cam(-cfg["x_limit"], 0)[0], cam(cfg["x_limit"], 0)[0]
    gl, gr = max(rect[0] + 4, lx), min(rect[2] - 4, rx)
    if gr > gl:
        line(draw, [(gl, cam.oy), (gr, cam.oy)], "dim", 2)
    pxl, pxr = cam(-cfg["platform_half_width"], 0)[0], cam(cfg["platform_half_width"], 0)[0]
    pl, pr = max(rect[0] + 4, pxl), min(rect[2] - 4, pxr)
    if pr > pl:
        box(draw, (pl, cam.oy, pr, cam.oy + 5), fill="nominal")
        hatch = max(7, cam.scale * 0.4)
        x = pl
        while x < pr:
            line(draw, [(x, cam.oy + 5), (min(pr, x + hatch * 0.6), cam.oy + 12)],
                 blend("nominal", "ground", .5), 1)
            x += hatch
        centre = min(max((pxl + pxr) / 2, rect[0] + 90), rect[2] - 90)
        text(draw, (centre, cam.oy + 16), f"plataforma \u00b1{cfg['platform_half_width']:.1f} m",
             "mono", 11 if compact else 13, "dim", anchor="ma")

    cos, sin = math.cos(state["theta"]), math.sin(state["theta"])
    px, py = cam(state["x"], state["y"])
    k = cam.scale
    def local(lx_, ly_):
        return (px + (lx_ * cos - ly_ * sin) * k, py - (lx_ * sin + ly_ * cos) * k)
    stroke = 2 if compact else 3
    for body_x, foot_x in ((-cfg["width"] / 2, -cfg["foot_half_span"]),
                           (cfg["width"] / 2, cfg["foot_half_span"])):
        a, b = local(body_x, -cfg["height"] / 2), local(foot_x, -cfg["foot_drop"])
        line(draw, [a, b], "ink", stroke)
        r = 3 if compact else 4
        draw.ellipse([(b[0] - r) * SS, (b[1] - r) * SS, (b[0] + r) * SS, (b[1] + r) * SS], fill=C["ink"])
    corners = [local(-cfg["width"] / 2, -cfg["height"] / 2), local(cfg["width"] / 2, -cfg["height"] / 2),
               local(cfg["width"] / 2, cfg["height"] / 2), local(-cfg["width"] / 2, cfg["height"] / 2)]
    draw.polygon([(p[0] * SS, p[1] * SS) for p in corners], fill=C["panel"])
    line(draw, corners + [corners[0]], "ink", stroke)
    line(draw, [local(0, -cfg["height"] / 2), local(0, cfg["height"] / 2)], "nominal", stroke - 1)


def draw_gates(draw, rect, cfg, state, compact):
    rows = gate_rows(cfg)
    x0, y0, x1, y1 = rect
    label_w, value_w = (104, 92) if compact else (168, 132)
    row_h = (y1 - y0) / len(rows)
    bar_h, size = (13, 13) if compact else (18, 17)
    band_colour = blend("nominal", "sunk", 0.26)
    for index, (name, key, lo, hi, tol, ticks, digits) in enumerate(rows):
        value = float(state[key])
        inside = tol[0] <= value <= tol[1]
        mid = y0 + index * row_h + row_h / 2
        text(draw, (x0, mid), name, "label", size, "dim", anchor="lm", spacing=0.4)
        bx0, bx1 = x0 + label_w, x1 - value_w
        span = hi - lo
        box(draw, (bx0, mid - bar_h / 2, bx1, mid + bar_h / 2), fill="sunk")
        tl = bx0 + (bx1 - bx0) * (tol[0] - lo) / span
        tr = bx0 + (bx1 - bx0) * (tol[1] - lo) / span
        box(draw, (tl, mid - bar_h / 2, tr, mid + bar_h / 2), fill=band_colour)
        for edge in (tl, tr):
            line(draw, [(edge, mid - bar_h / 2), (edge, mid + bar_h / 2)],
                 blend("nominal", "sunk", .7), 1)
        for tick in ticks:
            tx = bx0 + (bx1 - bx0) * (tick - lo) / span
            line(draw, [(tx, mid - bar_h / 2), (tx, mid + bar_h / 2)], blend("dim", "sunk", .8), 1)
        nx = bx0 + (bx1 - bx0) * (max(lo, min(hi, value)) - lo) / span
        colour = "nominal" if inside else "violation"
        box(draw, (nx - 1.8, mid - bar_h / 2 - 3, nx + 1.8, mid + bar_h / 2 + 3),
            fill="ink" if inside else "violation")
        text(draw, (x1, mid), f"{value:.{digits}f}", "mono", size, colour, anchor="rm")


def load(results, method, scenario):
    traces = pd.read_csv(Path(results) / f"{method}.traces.csv")
    traces = traces[traces.scenario_id == scenario]
    if traces.empty:
        raise SystemExit(f"Sin trazas de {method} en {scenario}")
    episodes = pd.read_csv(Path(results) / f"{method}.episodes.csv")
    row = episodes[episodes.scenario_id == scenario].iloc[0]
    frames = [{"t": r.time, "x": r.x, "y": r.y, "vx": r.vx, "vy": r.vy, "theta": r.theta,
               "omega": r.omega, "gas": r.gas, "action": max(0, int(r.action))}
              for r in traces.itertuples()]
    return {"method": method, "frames": frames, "success": bool(row.success), "outcome": row.outcome,
            "wind": float(row.wind_force_N), "condition": row.condition, "suite": row.suite,
            "duration": float(row.duration_s), "gas": float(row.gas_consumed_Ns)}


def render(results, scenario, methods, cfg, mode, out):
    runs = [load(results, m, scenario) for m in methods]
    compact = mode == "comparativo"
    wind, head = runs[0]["wind"], runs[0]
    longest = max(len(r["frames"]) for r in runs)
    total = longest + FPS
    ghosts = {r["method"]: [[(f["x"], f["y"]) for f in o["frames"]] for o in runs if o is not r]
              for r in runs}
    gates_h = 5 * (26 if compact else 38)

    def frames_iter():
        trails = {r["method"]: [] for r in runs}
        for index in range(total):
            base = Image.new("RGBA", (W * SS, H * SS), C["ground"])
            soft = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
            soft_draw = ImageDraw.Draw(soft)
            cameras = []
            for slot, run in enumerate(runs):
                if compact:
                    gap = 18
                    width = (W - 92 - 3 * gap) / 4
                    rect = (46 + slot * (width + gap), 132, 46 + slot * (width + gap) + width, H - 40)
                else:
                    rect = (46, 132, W - 46, H - 40)
                scene = (rect[0] + 6, rect[1] + (62 if compact else 80),
                         rect[2] - 6, rect[3] - gates_h - 20)
                state = run["frames"][min(index, len(run["frames"]) - 1)]
                trail = trails[run["method"]]
                if not trail or trail[-1] != (state["x"], state["y"]):
                    trail.append((state["x"], state["y"]))
                    if len(trail) > 95:
                        trail.pop(0)
                cam = Camera(scene, cfg, state["y"], 8.0 if compact else 5.5, state["x"])
                paste_sky(base, scene)
                draw_soft(soft_draw, cam, cfg, state, trail, ghosts[run["method"]], compact, wind)
                cameras.append((rect, scene, cam, run, state))
            base.alpha_composite(soft)

            draw = ImageDraw.Draw(base)
            text(draw, (46, 32), CONDITION.get(head["condition"], head["condition"]).upper(),
                 "label", 34, "ink", spacing=1.2)
            text(draw, (46, 80), f"{'caso reservado' if head['suite'] == 'reserved' else 'complemento independiente'}"
                 f"   \u00b7   {scenario}   \u00b7   viento {wind:+.0f} N", "mono", 17, "dim")
            shown = min(index, longest - 1)
            clock = max(r["frames"][min(shown, len(r["frames"]) - 1)]["t"] for r in runs)
            text(draw, (W - 46, 32), f"t = {clock:5.2f} s", "mono", 30, "ink", anchor="ra")
            text(draw, (W - 46, 80), "trazas congeladas \u00b7 sin simulacion nueva", "mono", 15,
                 "dim", anchor="ra")
            line(draw, [(46, 114), (W - 46, 114)], "line", 1)

            for rect, scene, cam, run, state in cameras:
                box(draw, rect, outline="line", radius=10)
                draw_solid(draw, cam, cfg, state, compact)
                text(draw, (rect[0] + 16, rect[1] + 13),
                     LABELS.get(run["method"], run["method"]).upper(), "label",
                     19 if compact else 26, "ink", spacing=0.8)
                ended = index >= len(run["frames"]) - 1
                if ended:
                    colour = "nominal" if run["success"] else "violation"
                    width = 66 if compact else 86
                    height = 22 if compact else 28
                    box(draw, (rect[2] - 16 - width, rect[1] + 11, rect[2] - 16, rect[1] + 11 + height),
                        outline=colour, radius=6)
                    text(draw, (rect[2] - 16 - width / 2, rect[1] + 11 + height / 2),
                         "EXITO" if run["success"] else "FALLO", "label",
                         14 if compact else 17, colour, anchor="mm", spacing=0.6)
                    text(draw, (rect[0] + 16, rect[1] + (38 if compact else 46)),
                         CAUSE.get(run["outcome"], run["outcome"]), "mono",
                         13 if compact else 17, colour)
                draw_gates(draw, (rect[0] + 16, rect[3] - gates_h - 10, rect[2] - 16, rect[3] - 10),
                           cfg, state, compact)
            yield base.convert("RGB").resize((W, H), Image.LANCZOS)

    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
               "-movflags", "+faststart", str(out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = 0
    for image in frames_iter():
        process.stdin.write(image.tobytes())
        count += 1
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg fallo")
    print(f"{out}  {count} cuadros  {count / FPS:.1f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="artifacts/scope_alignment_v1/final")
    p.add_argument("--scenario", required=True)
    p.add_argument("--methods", nargs="+",
                   default=["pid", "dqn_assisted42", "double_assisted42", "dqn_scratch"])
    p.add_argument("--mode", choices=["comparativo", "individual"], default="comparativo")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "individual" and len(args.methods) != 1:
        p.error("El modo individual toma un solo metodo")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.results, args.scenario, args.methods,
           json.loads(Path("configs/env.json").read_text()), args.mode, args.output)


if __name__ == "__main__":
    main()
