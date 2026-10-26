"""The paper's result figures (IEEE TWC), drawn with matplotlib from the results/ files that
COMMANDS sections 6-24 wrote. CPU, seconds; no GPU, no simulation.

    python tools/paper_figs.py                         # every figure: results/ -> figures/
    python tools/paper_figs.py --only fig05 fig09      # some of them
    python tools/paper_figs.py --list                  # what each figure shows and reads
    python tools/paper_figs.py --self-test             # synthetic inputs (no results needed)

Each figure lists its input files (FIGS below). A figure whose required input is missing is
skipped with a note; optional inputs (CLIC, a second variant) only add series or groups.
Every figure is written as figures/<name>.pdf (vector, fonts embedded, for the paper) and
figures/<name>.png (a preview), one IEEE column (3.5 in) or two (7.16 in) wide.

The same encoding in every figure (colours checked for colour-vision deficiency on white;
every series also has its own marker and line style, so the figures read in grayscale):
    Lagrangian index       blue      calibration-aware: solid, filled; the hard-target index
                                     next to it: dashed, hollow markers, hatched bars
    closed form            orange    the advantage index (mean), the online rule (target)
    Whittle index          aqua      calibration-aware / hard as for the Lagrangian index
    planner                yellow    plan_adv (replan_adv where slots are lost)
    PI ratio index         magenta   cross-hatched bars
    PF, PADC               greys     the baselines
    JSCC                   green     one decoder: solid; a decoder per length: dashed, hollow
    digital                greys     HEIC + IR-HARQ: dark, solid; 3 dB-gap code: dashed
    true curves (oracle)   ink       dotted
    phases, user counts    a blue ramp (dark = the first phase, or the most users)
Gains and gaps carry 95% bootstrap intervals over frames, as the tools print them.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------- the look -------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WHITE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN = "#e87ba4", "#008300"
RAMP = ("#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")         # blue steps, dark -> light
COL1, COL2 = 3.5, 7.16                                                       # IEEE widths, in
FONT = 8.0                                     # IEEE: 8-10 pt type in figures, at printed size

ENT = {                                                    # entity -> (label, colour, marker)
    "lagrange_soft": ("Lagrangian index, calibration-aware", BLUE, "o"),
    "lagrange": ("Lagrangian index", BLUE, "o"),
    "true": ("Lagrangian index, true curves", INK, "x"),
    "slope_adv": ("closed form (advantage index)", ORANGE, "s"),
    "online": ("closed form (online rule)", ORANGE, "s"),
    "whittle_soft": ("Whittle index, calibration-aware", AQUA, "^"),
    "whittle": ("Whittle index", AQUA, "^"),
    "planner": ("planner", YELLOW, "D"),
    "pi": ("PI ratio index", MAGENTA, "P"),
    "pf": ("PF", MUTED, "v"),
    "padc": ("PADC", INK2, "X"),
    "jscc": ("JSCC, one decoder", GREEN, "o"),
    "jscc_hd": ("JSCC, a decoder per length", GREEN, "o"),
    "heic": ("HEIC + IR-HARQ", INK2, "s"),
    "gap3": ("3 dB-gap code + IR-HARQ", MUTED, "^"),
    "code": ("piecewise-trained, one decoder", GREEN, "o"),
    "code_hd": ("piecewise-trained, a decoder per length", GREEN, "o"),
    "percbr": ("one model per CBR", INK2, "h"),
    "code_s1": ("constant-SNR-trained", MUTED, "d"),
}
ALT_ALWAYS = ("jscc_hd", "gap3", "code_hd")                # always dashed / hollow
ALT_PAIRED = {"lagrange": "lagrange_soft", "whittle": "whittle_soft"}  # when the partner shows
PLANNER = {"plan_adv": "planner", "replan_adv": "planner"}
MEAN_KEYS = ["slope_adv", "pi", "whittle", "lagrange"]
TGT_KEYS = ["online", "planner", "whittle", "whittle_soft", "lagrange", "lagrange_soft"]


def is_alt(key, keys):
    return key in ALT_ALWAYS or (key in ALT_PAIRED and ALT_PAIRED[key] in keys)


def style(key, keys=()):
    """Line-plot keywords of an entity; keys: the entities in the same figure."""
    label, color, marker = ENT[key]
    alt = is_alt(key, keys)
    kw = dict(color=color, marker=marker, label=label, lw=1.2, ms=4.2, zorder=3,
              ls="--" if alt else "-", mfc=WHITE if alt else color, mec=color, mew=0.9)
    if key == "true":
        kw.update(ls=":", lw=1.1, ms=4.6, mew=1.0)
    return kw


def dot_style(key, keys=()):
    kw = style(key, keys)
    kw.update(ls="none", ms=4.6)
    kw.pop("lw")
    return kw


def bar_style(key, keys=()):
    label, color, _ = ENT[key]
    if is_alt(key, keys):
        return dict(facecolor=WHITE, edgecolor=color, hatch="////", linewidth=0.7, label=label)
    if key == "pi":
        return dict(facecolor=color, edgecolor=WHITE, hatch="xxx", linewidth=0.4, label=label)
    return dict(facecolor=color, edgecolor=WHITE, linewidth=0.4, label=label)


def ordinal(n):
    """n steps of one blue ramp, dark to light (phase 1, or the most users, is the darkest),
    spread evenly through the palette's steps."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    if n <= 1:
        return [BLUE]
    cmap = LinearSegmentedColormap.from_list("ordinal", RAMP)
    return [to_hex(cmap(t)) for t in ([0.0, 0.7] if n == 2 else np.linspace(0.0, 1.0, n))]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
        "legend.fontsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 2.0, "figure.facecolor": WHITE,
        "axes.facecolor": WHITE, "savefig.facecolor": WHITE, "pdf.fonttype": 42,
        "ps.fonttype": 42, "errorbar.capsize": 1.6, "hatch.linewidth": 0.6,
        "axes.titlelocation": "left", "axes.titlepad": 3.0, "mathtext.fontset": "custom",
        "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans:italic", "mathtext.sf": "sans",
    })
    return plt


def panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".strip(), loc="left", color=INK)


def zero(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0.0, color=AXIS, lw=0.8, zorder=1)


def k_axis(ax, Ks, label="users K (B = K / 8 resource units per slot)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.minorticks_off()
    ax.set_xlabel(label)


SHORT = {"lagrange": "Lagrangian index", "lagrange_soft": "Lagrangian index",
         "whittle": "Whittle index", "whittle_soft": "Whittle index",
         "true": "true curves (oracle)"}             # cell labels under a column header


def fig_legend(fig, axes, ncol, extra=(), rows=None, header=None):
    """One legend above the figure for every series in `axes` (first label wins). rows: the
    legend as a table of entity keys, one tuple per row (None: an empty cell), so a hard-target
    index sits next to its calibration-aware partner; header: a title per column (the cells
    then carry short names). A plain legend loses columns until it fits the figure's width."""
    from matplotlib.lines import Line2D
    seen = {}
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if not lab.startswith("_"):
                seen.setdefault(lab, h)
    for h in extra:
        seen.setdefault(h.get_label(), h)
    if not seen:
        return
    handles, labels = list(seen.values()), list(seen)
    if rows:
        rows = [r for r in rows if any(k is not None and ENT[k][0] in seen for k in r)]
        ncol = max(len(r) for r in rows)
        handles, labels, placed = [], [], set()
        for c in range(ncol):                                   # matplotlib fills by column
            if header:
                handles.append(Line2D([], [], ls="none"))
                labels.append(header[c])
            for r in rows:
                k = r[c] if c < len(r) else None
                lab = ENT[k][0] if k is not None else None
                if lab in seen:
                    handles.append(seen[lab])
                    labels.append(SHORT.get(k, lab) if header else lab)
                    placed.add(lab)
                else:
                    handles.append(Line2D([], [], ls="none"))
                    labels.append(" ")
        rest = [lab for lab in seen if lab not in placed]
        handles += [seen[lab] for lab in rest]
        labels += rest
    ncol = min(ncol, len(labels))
    while True:
        leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                         ncol=ncol, frameon=False, columnspacing=1.1, handletextpad=0.5,
                         borderaxespad=0.1)
        for t in leg.get_texts():
            if header and t.get_text() in header:
                t.set_color(INK2)
                t.set_fontstyle("italic")
        wide = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        if rows or ncol <= 1 or wide <= fig.get_figwidth() - 0.04:
            return
        leg.remove()
        ncol -= 1


# --------------------------------------------------------------------------- the inputs -------
class Missing(Exception):
    pass


class Inputs:
    """The files under --results; records which ones a figure read."""

    def __init__(self, root):
        self.root, self.used = root, []

    def path(self, name, required=True):
        p = os.path.join(self.root, name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            self.used.append(name)
            return p
        if required:
            raise Missing(name)
        return None

    def json(self, name, required=True):
        p = self.path(name, required)
        if p is None:
            return None
        with open(p) as f:
            return json.load(f)


def rmab(d):
    """tools/rmab_scale.py --out: ({K: row}, scale): a target run's values are shares, x 100
    gives points."""
    rows = {int(k): v for k, v in d["per_K"].items()}
    return rows, (1.0 if d["args"].get("target") is None else 100.0)


F = r"[+-]?(?:\d+(?:\.\d+)?|nan|inf)"
ROW_MEAN = re.compile(rf"^(\w+)\s+({F})\s+({F})\s+({F})\s+({F})%\s+({F})\s+({F})\s+({F}) "
                      rf"\[({F}), ({F})\]")
ROW_TGT = re.compile(rf"^(\w+)\s+({F})%\s+({F})\s+({F})%\s+({F})\s+({F})\s+({F})% \[({F}), "
                     rf"({F})\](?:\s+({F})% \[({F}), ({F})\])?")


def parse_sim(path):
    """The printed tables of tools/schedule_sim.py, every run of a file: a list of blocks
    {"set": the header's settings, "kind": "mean" | "target", "base", "rows": {policy: ...},
    "cuts": [slots], "delay": {policy: [% at the target by each cut]}}."""
    blocks, cur, mode = [], None, None
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    for ln in lines:
        if re.match(r"^\d+ users, \d+ slots \(", ln):
            cur = {"set": header_settings(ln), "kind": None, "base": None, "rows": {},
                   "cuts": [], "delay": {}}
            blocks.append(cur)
            mode = None
            continue
        if cur is None:
            continue
        if re.match(r"^policy\s+(mean PSNR|satisfied)", ln):
            cur["kind"] = "target" if ln.split()[1] == "satisfied" else "mean"
            b = re.search(r"vs (\w+) \[95% CI\]", ln)
            cur["base"] = b.group(1) if b else None
            mode = "table"
            continue
        if re.match(r"^policy\s+1st", ln):
            cur["cuts"] = [int(x) for x in re.findall(r"by (\d+)", ln)]
            mode = "delay"
            continue
        if mode == "table":
            m = (ROW_TGT if cur["kind"] == "target" else ROW_MEAN).match(ln)
            if m:
                g = m.groups()
                if cur["kind"] == "mean":
                    cur["rows"][g[0]] = {"mean_psnr": float(g[1]), "p5": float(g[2]),
                                         "worst": float(g[3]), "vs": float(g[7]),
                                         "ci": (float(g[8]), float(g[9]))}
                else:
                    r = {"satisfied": float(g[1]) / 100.0, "mean_psnr": float(g[2]),
                         "vs": float(g[6]), "ci": (float(g[7]), float(g[8]))}
                    if g[9] is not None:
                        r["vs_padc"], r["ci_padc"] = float(g[9]), (float(g[10]), float(g[11]))
                    elif cur["base"] == "padc":
                        r["vs_padc"], r["ci_padc"] = r["vs"], r["ci"]
                    cur["rows"][g[0]] = r
        elif mode == "delay" and cur["cuts"]:
            m = re.match(r"^(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)(.*)$", ln)
            if m:
                by = [float(x) for x in re.findall(r"(\d+\.\d+)%", m.group(4))][:len(cur["cuts"])]
                if len(by) == len(cur["cuts"]):
                    cur["delay"][m.group(1)] = by
    return [b for b in blocks if b["kind"]]


def header_settings(h):
    s = {"users": int(re.match(r"^(\d+) users", h).group(1)),
         "slots": int(re.match(r"^\d+ users, (\d+) slots", h).group(1))}
    m = re.search(r"CQI error ([\d.]+) dB", h)
    s["cqi_error"] = float(m.group(1)) if m else 0.0
    m = re.search(r"stale reports \(correlation ([\d.]+); transmitter (\w+)\)", h)
    s["corr"], s["aware"] = (float(m.group(1)), m.group(2) == "aware") if m else (1.0, None)
    m = re.search(r"([\d.]+)% of slots preempted", h)
    s["preempt"] = float(m.group(1)) / 100.0 if m else 0.0
    m = re.search(r"digital: (\w+) link", h)
    s["link"] = m.group(1) if m else None
    m = re.search(r"HARQ (\w+) \(up to", h)
    s["harq"] = m.group(1) if m else None
    m = re.search(r"utility (\S+)\s*$", h)
    s["utility"] = m.group(1) if m else ""
    return s


def pick(blocks, kind, want):
    """The first block of this kind whose settings match `want`."""
    for b in blocks:
        if b["kind"] == kind and all(
                abs(b["set"].get(k, 0.0) - v) < 1e-9 if isinstance(v, float)
                else b["set"].get(k) == v for k, v in want.items()):
            return b
    return None


def sim_rows(blocks, kind, want, field="vs", base="pf"):
    """{entity: (value, (lo, hi))} of the block matching `want`: the gain over `base`."""
    b = pick(blocks, kind, want)
    if b is None or (field == "vs" and b["base"] != base):
        return None
    ci = "ci" if field == "vs" else "ci_padc"
    return {PLANNER.get(p, p): (r[field], r[ci]) for p, r in b["rows"].items() if field in r}


ABS_RULES = (("mean dB", "mean_db"), ("noise variance", "noise"),
             ("capacity (MIESM-like)", "capacity"), ("exponential (EESM)", "eesm"))


def parse_abstractions(path):
    """tools/utility_fit.py --compare-abstractions, every block of a file in order:
    {rule: (MAE with equal weights, MAE with sensitivity weights)}, None where not printed."""
    out, cur = [], None
    with open(path) as f:
        for ln in f:
            if ln.startswith("link abstractions on the held-out profiles"):
                cur = {}
                out.append(cur)
                continue
            if cur is None:
                continue
            s = ln.strip()
            key, rest = None, None
            m = re.match(r"MMSE, kappa \S+\s+(.*)$", s)
            if m:
                key, rest = "mmse", m.group(1)
            else:
                for name, k in ABS_RULES:
                    if s.startswith(name + " "):
                        key, rest = k, s[len(name):]
                        break
            if key is None:
                if s and not s.startswith(("averaging rule", "additive model")):
                    cur = None
                continue
            tok, cells = rest.split(), []
            while tok and len(cells) < 2:
                t = tok.pop(0)
                if t == "-":
                    cells.append(None)
                else:
                    cells.append(float(t))
                    tok = tok[2:]                                  # "/ p95"
            cur[key] = tuple(cells + [None] * (2 - len(cells)))
    return [b for b in out if b]


def decoded_frames(spec, metric="psnr", target=None):
    """{policy: {frame: [per-user values]}} of a decoded replay (tools/mixed_snr.py --design
    file), scored as tools/replay_report.py scores it."""
    from tools.replay_report import score
    per = {}
    for e in spec["entries"]:
        if metric == "psnr":
            v = float(score(e) >= target) if target is not None else float(score(e))
        else:
            v = e.get("actual_" + metric)
            if v is None:
                continue
        per.setdefault(e.get("policy", "?"), {}).setdefault(e["frame"], []).append(float(v))
    return per


def paired(spec, base, metric="psnr", target=None, scale=1.0):
    """{entity: (mean, (lo, hi))}: each policy minus `base` per frame, the bootstrap of
    tools/replay_report.py."""
    from tools.schedule_sim import bootstrap
    per = decoded_frames(spec, metric, target)
    if base not in per:
        return {}
    out = {}
    for p, frames in per.items():
        common = sorted(set(frames) & set(per[base]))
        if p == base or len(common) < 2:
            continue
        d = [np.mean(frames[f]) - np.mean(per[base][f]) for f in common]
        lo, hi = bootstrap(d)
        out[PLANNER.get(p, p)] = (scale * float(np.mean(d)), (scale * lo, scale * hi))
    return out


def missed_share(spec, target):
    """{entity: (%, (lo, hi))}: the users the model predicted at the target whose decode fell
    below it (tools/mixed_snr.py's 'missed'), per frame, bootstrap over frames."""
    from tools.replay_report import score
    from tools.schedule_sim import bootstrap
    per = {}
    for e in spec["entries"]:
        miss = e.get("predicted_psnr", 0.0) >= target and score(e) < target
        per.setdefault(e.get("policy", "?"), {}).setdefault(e["frame"], []).append(float(miss))
    out = {}
    for p, frames in per.items():
        d = [np.mean(v) for v in frames.values()]
        if len(d) > 1:
            lo, hi = bootstrap(d)
            out[PLANNER.get(p, p)] = (100.0 * float(np.mean(d)), (100.0 * lo, 100.0 * hi))
    return out


def constant_curves(d):
    """{prefix length (phases): {SNR: mean PSNR over images}} of a tools/mixed_snr.py run."""
    out = {}
    per = d.get("per_phase") or 1
    for lv in d["levels"]:
        L = int(round(lv["units"] / per))
        ps = np.asarray(lv["psnr"], dtype=np.float64)
        for j, (pr, kind) in enumerate(zip(lv["profiles"], lv["kinds"])):
            if kind == "constant":
                out.setdefault(L, {})[float(pr[0])] = float(np.nanmean(ps[:, j]))
    return out


def _f(v):
    return np.nan if v is None else float(v)


# --------------------------------------------------------------------------- drawing aids -----
def bars(ax, groups, keys, ylabel):
    """Grouped bars with 95% intervals: groups = [(label, {entity: (v, (lo, hi))})]."""
    present = [k for k in keys if any(k in g for _, g in groups)]
    n = max(len(present), 1)
    w = min(0.82 / n, 0.2)
    for i, k in enumerate(present):
        xs, vs, el, eh = [], [], [], []
        for gi, (_, g) in enumerate(groups):
            if k in g:
                v, (lo, hi) = g[k]
                xs.append(gi + (i - (n - 1) / 2) * w)
                vs.append(v)
                el.append(max(v - lo, 0.0))
                eh.append(max(hi - v, 0.0))
        ax.bar(xs, vs, width=w * 0.9, zorder=2, **bar_style(k, present))
        ax.errorbar(xs, vs, yerr=[el, eh], ls="none", ecolor=INK2, elinewidth=0.7, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.tick_params(axis="x", length=0)
    zero(ax)
    ax.set_ylabel(ylabel)


def gap_line(ax, rows, Ks, p, sc, kw):
    Ks = [k for k in Ks if f"{p}_gap" in rows.get(k, {})]
    if not Ks:
        return
    g = np.array([rows[k][f"{p}_gap"] for k in Ks]) * sc
    ci = np.array([rows[k][f"{p}_gap_ci"] for k in Ks]) * sc
    ax.errorbar(Ks, g, yerr=np.vstack([np.maximum(g - ci[:, 0], 0), np.maximum(ci[:, 1] - g, 0)]),
                elinewidth=0.7, **kw)


def _k(p):
    """tools/rmab_scale.py's name of our closed form."""
    return "adv" if p == "slope_adv" else p


# ------------------------------------------------------------------------------ figures -------
RULE_ROWS = (("MIESM (capacity, equal weights)", "capacity", 0), ("capacity", "capacity", 1),
             ("mean dB", "mean_db", 1), ("EESM", "eesm", 1), ("noise variance", "noise", 1),
             ("MMSE (ours)", "mmse", 1))


def fig01(I, plt):
    """Link abstraction (C12): the held-out error of each rule that maps a delivery history
    to one effective SNR on the image's own constant-SNR curve; sensitivity weights except
    MIESM. Four codes on Kodak, two on CLIC."""
    kod = parse_abstractions(I.path("s13_abstractions.txt"))
    p = I.path("s22_clic.txt", required=False)
    clic = parse_abstractions(p) if p else []
    panels = [x for x in (("Kodak", kod, (("code", 0), ("code_hd", 2), ("percbr", 3),
                                          ("code_s1", 1))),
                          ("CLIC (first 60 images)", clic, (("code", 0), ("code_s1", 1)))) if x[1]]
    if not panels:
        raise Missing("s13_abstractions.txt (no abstraction table in it)")
    fig, axes = plt.subplots(1, len(panels), figsize=(COL2 if len(panels) == 2 else COL1, 2.2),
                             sharey=True, squeeze=False)
    y = np.arange(len(RULE_ROWS))[::-1].astype(float)
    for ax, (name, blocks, codes), letter in zip(axes[0], panels, "ab"):
        shown = [(k, blocks[i]) for k, i in codes if i < len(blocks)]
        offs = np.linspace(-0.22, 0.22, len(shown)) if len(shown) > 1 else [0.0]
        for (k, blk), dy in zip(shown, offs):
            x = [_f(blk.get(key, (None, None))[w]) for _, key, w in RULE_ROWS]
            ax.plot(x, y + dy, **dot_style(k))
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in RULE_ROWS])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
        ax.set_xlabel("held-out error of the predicted PSNR, MAE (dB)")
        panel(ax, letter, name)
    fig_legend(fig, axes, 4)
    return fig


def fig02(I, plt):
    """The decoder weighs its phases (C1): mean PSNR with one phase's SNR varied, the
    others at the base SNR; piecewise-trained vs constant-SNR-trained code."""
    runs = [(I.json("p2_full_snrc4.json"), "piecewise-trained code")]
    b = I.json("p2_full_s1.json", required=False)
    if b is not None:
        runs.append((b, "constant-SNR-trained code"))
    fig, axes = plt.subplots(1, len(runs), figsize=(COL2 if len(runs) == 2 else COL1, 2.3),
                             sharey=True, squeeze=False)
    for ax, (d, title), letter in zip(axes[0], runs, "ab"):
        cbr = "1/8" if "1/8" in d["summary"] else list(d["summary"])[-1]
        rows = d["summary"][cbr]["oat"]
        marks = ("o", "s", "^", "D", "v", "P", "X", "h")
        for j, (vals, col) in enumerate(zip(rows, ordinal(len(rows)))):
            ax.plot(d["grid"], vals, color=col, marker=marks[j % len(marks)], ms=3.6, lw=1.2,
                    label=f"phase {j + 1}")
        ax.set_xticks(d["grid"])
        ax.set_xlabel(f"SNR of the varied phase (dB; the others at {d['base_snr']:g} dB)")
        panel(ax, letter, f"{title}, CBR {cbr}")
    axes[0, 0].set_ylabel("mean PSNR (dB)")
    fig_legend(fig, axes[0, :1], 6)
    return fig


def fig03(I, plt):
    """The abstraction as a link-to-system model (C2): the PSNR it predicts for scheduled
    histories against the PSNR the codec decodes, every policy's schedules."""
    sets = (("Kodak", GREEN, ("s22_dec_mean.json", "s22_dec_tgt.json", "s22_dec_r80_mean.json",
                              "s22_dec_r80_tgt.json", "s24_dec_auto.json", "s24_dec_r80.json")),
            ("CLIC", INK2, ("s22_clic_dec_mean.json", "s22_clic_dec_tgt.json",
                            "s24_dec_clic.json")))
    data = []
    for name, col, files in sets:
        pts = []
        for fn in files:
            d = I.json(fn, required=False)
            if d is not None:
                pts += [(e["predicted_psnr"], e["actual_psnr"]) for e in d["entries"]
                        if e.get("actual_psnr") is not None and "predicted_psnr" in e]
        if pts:
            data.append((name, col, np.asarray(pts, dtype=np.float64)))
    if not data:
        raise Missing("s22_dec_mean.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.3]})
    lo = min(float(p.min()) for _, _, p in data) - 0.3
    hi = max(float(p.max()) for _, _, p in data) + 0.3
    a1.plot([lo, hi], [lo, hi], color=AXIS, lw=0.8, zorder=1)
    for name, col, pts in data:
        a1.scatter(pts[:, 0], pts[:, 1], s=1.6, color=col, alpha=0.25, lw=0, rasterized=True,
                   zorder=2, label=name)
    a1.set_xlim(lo, hi)
    a1.set_ylim(lo, hi)
    a1.set_aspect("equal")
    a1.grid(axis="both")
    a1.set_xlabel("predicted PSNR (dB)")
    a1.set_ylabel("decoded PSNR (dB)")
    panel(a1, "a", "every decoded history")
    edges = np.arange(-0.5, 0.5 + 1e-9, 0.02)
    for name, col, pts in data:
        err = pts[:, 0] - pts[:, 1]
        a2.hist(np.clip(err, edges[0], edges[-1]), bins=edges, density=True, histtype="step",
                color=col, lw=1.1, label=f"{name}: bias {err.mean():+.3f} dB, MAE "
                f"{np.abs(err).mean():.3f} dB ({len(err)} histories)")
    zero(a2, horizontal=False)
    a2.set_xlabel("predicted minus decoded PSNR (dB)")
    a2.set_ylabel("density")
    panel(a2, "b", "the model's error")
    fig.legend(*a2.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=1, frameon=False)
    return fig


def fig04(I, plt):
    """Mean PSNR at scale (C13, C3): the gap to the Lagrangian bound per user and summed over
    the frame's users, B = K / 8."""
    rows, _ = rmab(I.json("s22_scale_mean.json"))
    Ks = sorted(rows)
    pols = [p for p in ("pf", "slope_adv", "pi", "whittle", "lagrange")
            if f"{_k(p)}_gap" in rows[Ks[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for ax, total, letter, title in ((axes[0], False, "a", "per user"),
                                     (axes[1], True, "b", "per frame (summed over the K users)")):
        for p in pols:
            g = np.array([rows[k][f"{_k(p)}_gap"] for k in Ks])
            ci = np.array([rows[k][f"{_k(p)}_gap_ci"] for k in Ks])
            m = np.array(Ks, dtype=np.float64) if total else np.ones(len(Ks))
            y, lo, hi = g * m, ci[:, 0] * m, ci[:, 1] * m
            ok = y > 0
            if ok.any():
                yerr = np.vstack([np.clip(y - lo, 0, 0.9 * y), np.clip(hi - y, 0, None)])[:, ok]
                ax.errorbar(np.array(Ks)[ok], y[ok], yerr=yerr, elinewidth=0.7, **style(p, pols))
        ax.set_yscale("log")
        k_axis(ax, Ks)
        ax.set_ylabel("gap to the bound (dB)")
        panel(ax, letter, title)
    fig_legend(fig, axes, 5)
    return fig


def fig05(I, plt):
    """A 30 dB target at scale (C13): the hard-target indices on the calibrated curves, the
    calibration-aware indices on the same curves, and the hard index given the true curves."""
    d = I.json("s24_soft_auto.json", required=False)
    rows, sc = rmab(d if d is not None else I.json("s22_gap_ties.json"))
    tr = I.json("s23_gap_truth.json", required=False)
    trow, tsc = rmab(tr) if tr else ({}, 100.0)
    Ks = sorted(set(rows) | set(trow))
    series = [p for p in ("whittle", "whittle_soft", "lagrange", "lagrange_soft")
              if any(f"{p}_gap" in r for r in rows.values())]
    keys = series + (["true"] if trow else [])
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    zero(ax)
    for p in series:
        gap_line(ax, rows, Ks, p, sc, style(p, keys))
    if trow:
        gap_line(ax, trow, Ks, "lagrange", tsc, style("true", keys))
    k_axis(ax, Ks)
    ax.set_ylabel("gap to the bound per user (points at 30 dB)")
    fig_legend(fig, [ax], 2, rows=[("whittle", "whittle_soft"), ("lagrange", "lagrange_soft"),
                                   ("true", None)], header=("hard target", "calibration-aware"))
    return fig


def _measured_sigma(I):
    """The mean of the per-length sigma 24.1a measured (its printed line), else 0.04 dB."""
    p = I.path("s24_soft_auto.txt", required=False)
    if p:
        with open(p) as f:
            m = re.search(r"sigma ((?:\d+\.\d+ ?)+) dB, bias", f.read())
        if m:
            return float(np.mean([float(x) for x in m.group(1).split()]))
    return 0.04


def fig06(I, plt):
    """The calibration-aware index does not need the spread exactly (C13): its gap at the
    measured sigma and at 0.05 / 0.1 / 0.2 dB; 0 = the hard target; dotted: the hard index
    on the true curves."""
    from matplotlib.lines import Line2D
    auto, sc = rmab(I.json("s24_soft_auto.json"))
    runs = []
    for s, fn in ((0.05, "s24_soft_s05.json"), (0.1, "s24_soft_s10.json"),
                  (0.2, "s24_soft_s20.json")):
        d = I.json(fn, required=False)
        if d is not None:
            runs.append((s, rmab(d)[0]))
    tr = I.json("s23_gap_truth.json", required=False)
    trow, tsc = rmab(tr) if tr else ({}, 100.0)
    sig = _measured_sigma(I)
    Ks = [k for k in (8, 32, 128) if k in auto] or sorted(auto)[:3]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    zero(ax)
    for k, col in zip(Ks, ordinal(len(Ks))[::-1]):
        pts = [(0.0, auto[k]["lagrange_gap"] * sc), (sig, auto[k]["lagrange_soft_gap"] * sc)]
        pts += [(s, r[k]["lagrange_soft_gap"] * sc) for s, r in runs
                if "lagrange_soft_gap" in r.get(k, {})]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=3.8, lw=1.2,
                label=f"K = {k}", zorder=3)
        if "lagrange_gap" in trow.get(k, {}):
            ax.axhline(trow[k]["lagrange_gap"] * tsc, color=col, lw=0.9, ls=":", zorder=2)
    ax.axvline(sig, color=AXIS, lw=0.8, ls="--", zorder=1)
    ax.text(sig, 0.98, " measured", transform=ax.get_xaxis_transform(), ha="left", va="top",
            color=INK2)
    ticks = [0.0, 0.05, 0.1, 0.15, 0.2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel(r"assumed curve error $\sigma$ (dB; 0: the hard target)")
    ax.set_ylabel("gap to the bound per user (points at 30 dB)")
    extra = [Line2D([], [], color=INK2, lw=0.9, ls=":", label="hard index, true curves")] \
        if trow else []
    fig_legend(fig, [ax], 2, extra)
    return fig


VARIANTS = (("hard target, as first run", "s22_gap_ties.json", "lagrange"),
            ("the relaxation's tie rule", "s22_gap_ties.json", "lagrange_relax"),
            ("supply-scaled kink prices", "s22_gap_prices.json", "lagrange"),
            ("LP update every 6 slots", "s22_gap_lpu.json", "lagrange_lpu"),
            ("finer program grids", "s22_gap_grid.json", "lagrange"),
            ("margin 0.25 dB", "s23_gap_m25.json", "lagrange"),
            ("margin 0.5 dB", "s23_gap_m50.json", "lagrange"),
            ("calibration-aware reward", "s24_soft_auto.json", "lagrange_soft"),
            ("true curves (oracle)", "s23_gap_truth.json", "lagrange"))


def fig07(I, plt):
    """What closed the 30 dB gap (owed items 9 and 13): every variant of the Lagrangian index
    tested, its gap to the bound per user at K = 8, 32, 128."""
    found = []
    for label, fn, p in VARIANTS:
        d = I.json(fn, required=False)
        if d is not None:
            rows, sc = rmab(d)
            if any(f"{p}_gap" in r for r in rows.values()):
                found.append((label, rows, sc, p))
    if len(found) < 2:
        raise Missing("s22_gap_ties.json and the other gap runs")
    allK = sorted({k for _, rows, _, _ in found for k in rows})
    Ks = [k for k in (8, 32, 128) if k in allK] or allK[:3]
    fig, ax = plt.subplots(figsize=(COL1, 2.9))
    zero(ax, horizontal=False)
    y = np.arange(len(found))[::-1].astype(float)
    for i, (k, col, mk) in enumerate(zip(Ks, ordinal(len(Ks))[::-1], ("o", "s", "D"))):
        xs, ys, el, eh = [], [], [], []
        for yy, (_, rows, sc, p) in zip(y, found):
            r = rows.get(k, {})
            if f"{p}_gap" in r:
                g, (lo, hi) = r[f"{p}_gap"] * sc, r[f"{p}_gap_ci"]
                xs.append(g)
                ys.append(yy - (i - (len(Ks) - 1) / 2) * 0.24)              # K = 8 on top
                el.append(max(g - lo * sc, 0))
                eh.append(max(hi * sc - g, 0))
        ax.errorbar(xs, ys, xerr=[el, eh], ls="none", marker=mk, ms=3.8, color=col, mec=col,
                    elinewidth=0.7, label=f"K = {k}", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f[0] for f in found])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("gap to the bound per user (points at 30 dB)")
    fig_legend(fig, [ax], 3)
    return fig


def fig08(I, plt):
    """Mean PSNR on the main frames, K 8, T 24 (C3, C13): every policy's gain over PF, fresh
    and under each impairment."""
    main = parse_sim(I.path("s22_main_mean.txt"))
    groups = [("fresh reports", sim_rows(main, "mean", {"cqi_error": 0.0, "corr": 1.0})),
              ("CQI error 2 dB", sim_rows(main, "mean", {"cqi_error": 2.0})),
              ("reports r = 0.8", sim_rows(main, "mean", {"corr": 0.8}))]
    for label, fn, want in (("25% of slots lost", "s21_lag_mean_more.txt", {"preempt": 0.25}),
                            ("T = 16 slots", "s21_lag_t16.txt", {"slots": 16}),
                            ("CLIC images", "s22_clic.txt", {"cqi_error": 0.0, "corr": 1.0})):
        p = I.path(fn, required=False)
        if p:
            groups.append((label, sim_rows(parse_sim(p), "mean", want)))
    groups = [(n, g) for n, g in groups if g]
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    bars(ax, groups, MEAN_KEYS, "mean PSNR gain over PF (dB)")
    fig_legend(fig, [ax], 4)
    return fig


def fig09(I, plt):
    """A 30 dB target on the main frames (C4, C7, C13): the share of users at 30 dB, every
    policy's gain over PADC, fresh and under each impairment."""
    soft = parse_sim(I.path("s24_main_soft.txt"))
    groups = [("fresh reports", sim_rows(soft, "target", {"cqi_error": 0.0, "corr": 1.0},
                                         "vs_padc")),
              ("CQI error 2 dB", sim_rows(soft, "target", {"cqi_error": 2.0}, "vs_padc"))]
    for label, fn, want in (("reports r = 0.8", "s24_main_soft_r80.txt", {"corr": 0.8}),
                            ("25% of slots lost", "s21_lag_tgt_more.txt", {"preempt": 0.25}),
                            ("T = 16 slots", "s21_lag_t16.txt", {"slots": 16}),
                            ("CLIC images", "s22_clic.txt", {"cqi_error": 0.0, "corr": 1.0})):
        p = I.path(fn, required=False)
        if p:
            groups.append((label, sim_rows(parse_sim(p), "target", want, "vs_padc")))
    groups = [(n, {k: v for k, v in g.items() if k != "padc"}) for n, g in groups if g]
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    bars(ax, groups, TGT_KEYS, "users at 30 dB: gain over PADC (points)")
    fig_legend(fig, [ax], 3)
    return fig


def fig10(I, plt):
    """A 30 dB target, decoded by the codec (C4, C13): (a) the gain in users at 30 dB over
    PADC, paired by frame; (b) the users the model put at 30 dB that decoded below it (the
    calibration-aware index hedges against them). 60 frames of each policy's schedules."""
    gains, missed = [], []
    for n, fn in (("Kodak", "s24_dec_auto.json"), ("Kodak, r = 0.8", "s24_dec_r80.json"),
                  ("CLIC", "s24_dec_clic.json")):
        d = I.json(fn, required=False)
        if d is not None:
            tgt = d.get("target") or 30.0
            gains.append((n, paired(d, "padc", target=tgt, scale=100.0)))
            missed.append((n, missed_share(d, tgt)))
    if not any(g for _, g in gains):
        raise Missing("s24_dec_auto.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.5))
    bars(a1, [g for g in gains if g[1]], TGT_KEYS, "gain over PADC (points)")
    panel(a1, "a", "users at 30 dB, decoded")
    bars(a2, missed, ["padc"] + TGT_KEYS, "share of users (%)")
    a2.set_ylim(bottom=0)
    panel(a2, "b", "predicted at 30 dB, decoded below it")
    fig_legend(fig, [a1, a2], 4, rows=[("padc", "online", "whittle", "lagrange"),
                                       (None, "planner", "whittle_soft", "lagrange_soft")])
    return fig


def fig11(I, plt):
    """Mean PSNR, decoded by the codec (C3); perceptual metrics follow PSNR: every policy's
    gain over PF in PSNR, MS-SSIM and LPIPS, paired by frame."""
    data = []
    for n, fn in (("Kodak", "s22_dec_mean.json"), ("Kodak, r = 0.8", "s22_dec_r80_mean.json"),
                  ("CLIC", "s22_clic_dec_mean.json")):
        d = I.json(fn, required=False)
        if d is not None:
            data.append((n, d))
    if not data:
        raise Missing("s22_dec_mean.json")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3))
    for ax, (m, title), letter in zip(axes, (("psnr", "PSNR (dB)"), ("msssim", "MS-SSIM"),
                                             ("lpips", "LPIPS (lower is better)")), "abc"):
        groups = [(n, paired(d, "pf", metric=m)) for n, d in data]
        groups = [g for g in groups if g[1]]
        if groups:
            bars(ax, groups, MEAN_KEYS, "")
        panel(ax, letter, title)
    axes[0].set_ylabel("decoded, minus PF")
    fig_legend(fig, axes, 4)
    return fig


def fig12(I, plt):
    """The latency budget (C14): quality against the frame length, K 8; the arrow shows how
    many slots the baseline needs for the Lagrangian index's quality at T 24."""
    a = I.json("s21_latency_mean.json", required=False)
    b = I.json("s21_latency_tgt.json", required=False)
    if a is None and b is None:
        raise Missing("s21_latency_mean.json")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), squeeze=False)
    for ax, d, letter, ylab, pols, sc, ref in (
            (axes[0, 0], a, "a", "mean PSNR (dB)", ("pf", "slope_adv", "whittle", "lagrange"),
             1.0, "pf"),
            (axes[0, 1], b, "b", "users at 30 dB (%)", ("padc", "online", "whittle", "lagrange"),
             100.0, "padc")):
        if d is None:
            ax.set_visible(False)
            continue
        Ts = d["slots"]
        pols = [p for p in pols if p in d["curves"]]
        for p in pols:
            ax.plot(Ts, sc * np.asarray(d["curves"][p], dtype=np.float64), **style(p, pols))
        m = d.get("match", {}).get("24")
        r = (m or {}).get("rows", {}).get(ref)
        if r and not r.get("flag") and d["args"].get("ref", "lagrange") in pols:
            lv = sc * m["level"]                  # every curve is below it left of T = 24
            ax.annotate("", xy=(r["needed"], lv), xytext=(24, lv),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8, shrinkA=0,
                                        shrinkB=0))
            ax.text(23.5, lv, f"{ENT[ref][0]} needs +{r['needed'] - 24:.1f} slots",
                    ha="right", va="bottom", color=INK2)
        ax.set_xticks(Ts)
        ax.set_xlabel("frame length T (slots)")
        ax.set_ylabel(ylab)
        panel(ax, letter, "mean PSNR" if letter == "a" else "a 30 dB target")
    fig_legend(fig, axes, 6)
    return fig


def fig13(I, plt):
    """Time to the target (C14): the share of all users at 30 dB by each slot of the frame
    (fresh reports, K 8, T 24)."""
    b = next((x for x in parse_sim(I.path("s24_main_soft.txt")) if x["delay"]), None)
    if b is None:
        raise Missing("s24_main_soft.txt (its --delay table)")
    order = ("padc", "online", "plan_adv", "whittle", "whittle_soft", "lagrange", "lagrange_soft")
    pols = [p for p in order if p in b["delay"]]
    keys = [PLANNER.get(p, p) for p in pols]
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for p in pols:
        ax.plot([0] + b["cuts"], [0.0] + b["delay"][p], **style(PLANNER.get(p, p), keys))
    ax.set_xticks([0] + b["cuts"])
    ax.set_xlim(0, b["cuts"][-1] + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"slot of the frame (T = {b['set']['slots']})")
    ax.set_ylabel("users at 30 dB so far (%)")
    fig_legend(fig, [ax], 2, rows=[("padc", "online"), ("planner", None),
                                   ("whittle", "whittle_soft"), ("lagrange", "lagrange_soft")])
    return fig


def _best(blocks, kind, axis, x, pred):
    """The best policy (and transmitter) over the blocks at one point of an axis."""
    key = "mean_psnr" if kind == "mean" else "satisfied"
    best = None
    for b in blocks:
        s = b["set"]
        if b["kind"] != kind or not pred(s):
            continue
        if axis == "corr" and not (abs(s["corr"] - x) < 1e-9 and s["cqi_error"] == 0.0):
            continue
        if axis == "cqi" and not (abs(s["cqi_error"] - x) < 1e-9 and s["corr"] == 1.0):
            continue
        for r in b["rows"].values():
            best = r[key] if best is None else max(best, r[key])
    return best


def fig14(I, plt):
    """JSCC against digital with HARQ (C5): the best policy (and transmitter, naive or
    aware) of each system under CQI errors and outdated reports."""
    single = parse_sim(I.path("s23_c5_single.txt"))
    dig = parse_sim(I.path("s20_harq_mean.txt"))
    dig_t, hd = [], []
    for fn in ("s20_harq_tgt.txt", "s20_harq_tgt_aware.txt", "s20_harq_tgt_cqi.txt"):
        p = I.path(fn, required=False)
        if p:
            dig_t += parse_sim(p)
    for fn in ("s18_stale_jscc.txt", "s13_cqi_jscc.txt"):
        p = I.path(fn, required=False)
        if p:
            hd += [b for b in parse_sim(p) if "utility_hd" in b["set"]["utility"]]

    def ir(link):
        return lambda s: s["harq"] == "ir" and s["link"] == link

    def everything(s):
        return True

    series = (("jscc", single, single, everything), ("jscc_hd", hd, hd, everything),
              ("heic", dig, dig_t, ir("cqi")), ("gap3", dig, dig_t, ir("shannon")))
    keys = [k for k, m, t, _ in series if m or t]
    R, S = (1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.0), (0.0, 1.0, 2.0, 3.0)
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.2), sharex="col")
    for col, (axis, xs, xlabel) in enumerate((("cqi", S, "CQI error sd (dB)"),
                                              ("corr", R, "report correlation r"))):
        for row, kind in enumerate(("mean", "target")):
            ax = axes[row, col]
            sc = 1.0 if kind == "mean" else 100.0
            for key, mb, tb, pred in series:
                blocks = mb if kind == "mean" else tb
                pts = [(x, _best(blocks, kind, axis, x, pred)) for x in xs]
                pts = [(x, sc * y) for x, y in pts if y is not None]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], **style(key, keys))
            if axis == "corr":
                ax.set_xlim(1.04, -0.04)
                ax.set_xticks([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
            else:
                ax.set_xticks(S)
            if row == 1:
                ax.set_xlabel(xlabel)
            ax.set_ylabel("mean PSNR (dB)" if kind == "mean" else "users at 30 dB (%)")
            panel(ax, "abcd"[2 * row + col], ("mean PSNR" if kind == "mean" else "a 30 dB target")
                  + (", CQI errors" if axis == "cqi" else ", outdated reports"))
    fig_legend(fig, axes, 4)
    return fig


def fig15(I, plt):
    """The scheduler makes the code's channel (C8, C11): how much better than the user's mean
    SNR the first three phases' slots are, against report age and the number of users."""
    from tools.codesign import stats
    by_r = ("bank_sa.json", "bank_sa_cqi95.json", "bank_sa_cqi80.json", "bank_sa_cqi60.json",
            "bank_sa_cqi30.json", "bank_sa_cqi0.json")
    by_k = ("bank_lift_k2.json", "bank_lift_k4.json", "bank_sa.json", "bank_sa_k16.json",
            "bank_lift_k24.json")
    I.path("bank_sa.json")
    cache = {}

    def lifts(fn):
        if fn not in cache:
            d = I.json(fn, required=False)
            cache[fn] = None if d is None else (d, [p["lift"] for p in stats(d)["phase"]])
        return cache[fn]

    ra = sorted(((v[0].get("cqi_corr", 1.0), v[1]) for v in map(lifts, by_r) if v), reverse=True)
    ka = sorted((v[0]["users"], v[1]) for v in map(lifts, by_k) if v)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
    for ax, data, letter, xlabel, title in (
            (axes[0], ra, "a", "report correlation r", "against report age (K = 8)"),
            (axes[1], ka, "b", "users K (T = 3K slots)", "against the number of users")):
        for j, col in zip(range(3), ordinal(3)):
            pts = [(x, ph[j]) for x, ph in data if len(ph) > j and ph[j] is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=3.8,
                        lw=1.2, label=f"phase {j + 1}")
        zero(ax)
        ax.set_xlabel(xlabel)
        panel(ax, letter, title)
    axes[0].set_xlim(1.04, -0.04)
    if ka:
        k_axis(axes[1], [k for k, _ in ka], "users K (T = 3K slots)")
    axes[0].set_ylabel("served SNR minus the user's mean SNR (dB)")
    fig_legend(fig, axes, 3)
    return fig


def fig16(I, plt):
    """Encode once with one decoder (C6): what a decoder per length (behind the same encoder)
    and one model per CBR add over the single decoder at each prefix length; mean over the
    constant SNRs, whiskers: the smallest and largest."""
    prefix = constant_curves(I.json("p2_full_snrc4.json"))
    out = {}
    for key, files in (("code_hd", ("p4_full_h48.json", "p4_full_h24.json", "p4_full_h16.json",
                                    "p4_full_h12.json", "p4_full_h5-48.json", "p4_full_h8.json")),
                       ("percbr", tuple(f"p3_full_fix{n}.json" for n in range(1, 7)))):
        cur = {}
        for fn in files:
            d = I.json(fn, required=False)
            if d is not None:
                cur.update(constant_curves(d))
        if cur:
            out[key] = cur
    if not out:
        raise Missing("p4_full_h48.json ... or p3_full_fix1.json ...")
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    zero(ax)
    for key, cur in out.items():
        Ls = sorted(L for L in cur if L in prefix)
        mean, el, eh = [], [], []
        for L in Ls:
            d = [cur[L][s] - prefix[L][s] for s in cur[L] if s in prefix[L]]
            mean.append(float(np.mean(d)))
            el.append(mean[-1] - float(np.min(d)))
            eh.append(float(np.max(d)) - mean[-1])
        ax.errorbar(Ls, mean, yerr=[el, eh], elinewidth=0.7, **style(key, list(out)))
    cbrs = ("1/48", "1/24", "1/16", "1/12", "5/48", "1/8")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"{n}\n{c}" for n, c in zip(range(1, 7), cbrs)])
    ax.set_xlabel("prefix length: phases (CBR)")
    ax.set_ylabel("PSNR over the one-decoder code (dB)")
    fig_legend(fig, [ax], 1)
    return fig


FIGS = (
    ("fig01_abstraction", fig01, "held-out error of each effective-SNR rule, four codes, Kodak "
     "and CLIC", "s13_abstractions.txt [s22_clic.txt]"),
    ("fig02_phase_weights", fig02, "mean PSNR with one phase's SNR varied, two codes",
     "p2_full_snrc4.json [p2_full_s1.json]"),
    ("fig03_predicted_vs_decoded", fig03, "predicted vs decoded PSNR of scheduled histories",
     "s22_dec_*.json s24_dec_*.json [s22_clic_dec_*.json s24_dec_clic.json]"),
    ("fig04_mean_scaling", fig04, "mean PSNR: gap to the bound vs K, per user and per frame",
     "s22_scale_mean.json"),
    ("fig05_target_scaling", fig05, "30 dB: gap to the bound vs K, hard / calibration-aware / "
     "true curves", "s24_soft_auto.json (or s22_gap_ties.json) [s23_gap_truth.json]"),
    ("fig06_soft_sigma", fig06, "30 dB: the calibration-aware index vs the sigma it assumes",
     "s24_soft_auto.json [s24_soft_s05/s10/s20.json s24_soft_auto.txt s23_gap_truth.json]"),
    ("fig07_target_variants", fig07, "30 dB: every variant tested, gap at K 8 / 32 / 128",
     "two or more of s22_gap_ties/prices/lpu/grid.json s23_gap_m25/m50/truth.json "
     "s24_soft_auto.json"),
    ("fig08_main_mean", fig08, "main frames: mean PSNR gain over PF per impairment",
     "s22_main_mean.txt [s21_lag_mean_more.txt s21_lag_t16.txt s22_clic.txt]"),
    ("fig09_main_target", fig09, "main frames: gain in users at 30 dB over PADC per impairment",
     "s24_main_soft.txt [s24_main_soft_r80.txt s21_lag_tgt_more.txt s21_lag_t16.txt "
     "s22_clic.txt]"),
    ("fig10_decoded_target", fig10, "30 dB, decoded: gain over PADC; users predicted at 30 dB "
     "that decoded below", "s24_dec_auto.json [s24_dec_r80.json s24_dec_clic.json]"),
    ("fig11_decoded_mean", fig11, "mean, decoded: gains over PF in PSNR, MS-SSIM and LPIPS",
     "s22_dec_mean.json [s22_dec_r80_mean.json s22_clic_dec_mean.json]"),
    ("fig12_latency", fig12, "quality vs frame length: the latency budget",
     "s21_latency_mean.json and/or s21_latency_tgt.json"),
    ("fig13_time_to_target", fig13, "share of users at 30 dB by each slot",
     "s24_main_soft.txt"),
    ("fig14_digital", fig14, "JSCC vs digital with IR-HARQ: CQI errors and outdated reports",
     "s23_c5_single.txt s20_harq_mean.txt [s20_harq_tgt*.txt s18_stale_jscc.txt "
     "s13_cqi_jscc.txt]"),
    ("fig15_scheduler_channel", fig15, "SNR lift of the first phases vs report age and K",
     "bank_sa.json [bank_sa_cqi*.json bank_sa_k16.json bank_lift_k*.json]"),
    ("fig16_decoders", fig16, "a decoder per length and one model per CBR over the single "
     "decoder, by length", "p2_full_snrc4.json and p4_full_h*.json and/or p3_full_fix*.json"),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results", help="the folder the COMMANDS runs wrote")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="+", default=None, help="figure names or prefixes (fig05 ...)")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=300, help="PNG (and rasterised scatter) resolution")
    ap.add_argument("--list", action="store_true", help="list the figures and their inputs")
    ap.add_argument("--self-test", action="store_true",
                    help="draw every figure from synthetic inputs in the real tools' formats "
                         "(tools/paper_figs_selftest.py; seconds, CPU), stamped as synthetic")
    a = ap.parse_args()
    if a.list:
        for name, _, what, files in FIGS:
            print(f"{name:28s} {what}\n{'':28s} reads {files}")
        return
    tmp = None
    if a.self_test:
        import tempfile
        from tools.paper_figs_selftest import build
        tmp = tempfile.TemporaryDirectory(prefix="paper_figs_selftest_")
        a.results = build(tmp.name)
        print(f"synthetic inputs built in {a.results} (removed at the end)")
    plt = setup()
    os.makedirs(a.out, exist_ok=True)
    done, skipped = [], []
    for name, fn, what, _ in FIGS:
        if a.only and not any(name.startswith(o) for o in a.only):
            continue
        I = Inputs(a.results)
        try:
            fig = fn(I, plt)
        except Missing as e:
            skipped.append(name)
            print(f"  skip  {name}: needs {e}")
            continue
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()                     # labels settle on the first draw; lay out again
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        wide = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if wide > fig.get_figwidth() + 0.05:
            print(f"  note  {name}: {wide:.2f} in wide, more than its {fig.get_figwidth():.2f} "
                  f"in column (a legend or label overflows; it would print smaller)")
        if a.self_test:
            fig.text(0.995, 0.005, "synthetic self-test data", ha="right", va="bottom",
                     fontsize=7, color=MUTED)
        for fmt in a.formats:
            fig.savefig(os.path.join(a.out, f"{name}.{fmt}"), dpi=a.dpi, bbox_inches="tight",
                        pad_inches=0.02)
        plt.close(fig)
        done.append(name)
        print(f"  wrote {name}: {what} (from {', '.join(dict.fromkeys(I.used))})")
    print(f"{len(done)} figures in {a.out}/" + (f", {len(skipped)} skipped" if skipped else ""))
    if tmp is not None:
        tmp.cleanup()
        if skipped:
            raise SystemExit("self-test: a figure was skipped")


if __name__ == "__main__":
    main()
