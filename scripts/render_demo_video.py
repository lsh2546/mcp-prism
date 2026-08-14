
from __future__ import annotations

import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG, PANEL, INK, MUTED, LIME, CYAN, RED = "#07100d", "#101b17", "#f5f7f4", "#a8b0a9", "#b8ff3d", "#54e7ff", "#ff6d6d"


def font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def base(kicker: str, number: str):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for x in range(0, W, 64): d.line((x, 0, x, H), fill="#0d1a16", width=1)
    for y in range(0, H, 64): d.line((0, y, W, y), fill="#0d1a16", width=1)
    d.text((90, 60), "MCP PRISM", font=font(28, True), fill=INK)
    d.text((W - 390, 60), "??NATIVE ARM64", font=font(25, True), fill=LIME)
    d.text((90, 155), number, font=font(24, True), fill=LIME)
    d.text((155, 155), kicker.upper(), font=font(24, True), fill=CYAN)
    return im, d


def multiline(d, xy, text, size, color=INK, bold=False, width=34, spacing=8):
    d.multiline_text(xy, textwrap.fill(text, width), font=font(size, bold), fill=color, spacing=spacing)


def slide_hero(path):
    im, d = base("The MCP schema bottleneck, removed", "00")
    d.text((90, 255), "Fewer schemas.", font=font(112, True), fill=INK)
    d.text((90, 375), "Faster agents.", font=font(112, True), fill=LIME)
    cards = [("8.80횞", "PEAK THROUGHPUT"), ("85% vs 60%", "WORKFLOW SUCCESS"), ("100%", "FIRST-STEP ACCURACY"), ("ARM64", "NEOVERSE-N2")]
    x = 90
    for value, label in cards:
        d.rounded_rectangle((x, 650, x + 410, 870), 10, fill=PANEL, outline="#2d3b35")
        d.text((x + 28, 700), value, font=font(58, True), fill=LIME)
        d.text((x + 28, 795), label, font=font(19, True), fill=MUTED)
        x += 435
    im.save(path)


def slide_problem(path):
    im, d = base("One request should not read sixty-one manuals", "01")
    multiline(d, (90, 245), "Every connected MCP tool schema is serialized into the prompt?봢ven when the request needs only one tool.", 62, bold=True, width=48)
    d.rounded_rectangle((90, 610, 1830, 870), 12, fill=PANEL, outline="#2d3b35")
    d.text((135, 660), "?쏻hat is tomorrow?셲 weather in Seoul???, font=font(42, True), fill=CYAN)
    d.text((135, 755), "BASELINE  61 schemas  쨌  4,891 tokens", font=font(32, True), fill=RED)
    d.text((1030, 755), "PRISM  1 schema  쨌  185 tokens", font=font(32, True), fill=LIME)
    im.save(path)


def slide_system(path):
    im, d = base("Retrieve, canonicalize, reuse", "02")
    steps = [("1", "INT8 RETRIEVAL", "Find relevant tools"), ("2", "WORKFLOW FRONTIER", "Expose the next valid step"), ("3", "STABLE PREFIX", "Canonical schema order"), ("4", "LLAMA.CPP KV", "Reuse prompt cache")]
    x = 90
    for n, title, desc in steps:
        d.rounded_rectangle((x, 310, x + 390, 690), 14, fill=PANEL, outline=LIME if n == "4" else "#2d3b35", width=3)
        d.text((x + 30, 345), n, font=font(34, True), fill=LIME)
        d.text((x + 30, 440), title, font=font(28, True), fill=INK)
        multiline(d, (x + 30, 515), desc, 29, MUTED, width=20)
        if n != "4": d.text((x + 405, 475), "??, font=font(46, True), fill=LIME)
        x += 435
    d.text((90, 800), "OpenAI-compatible gateway 쨌 existing agents change only the base URL", font=font(35, True), fill=CYAN)
    im.save(path)


def slide_arm(path):
    im, d = base("Actual native execution", "03")
    d.rounded_rectangle((90, 245, 1830, 865), 12, fill="#050a08", outline="#30433a")
    console = [
        "$ uname -m", "aarch64", "$ lscpu | grep -E 'Model name|CPU\\(s\\)'", "CPU(s): 4", "Model name: Neoverse-N2",
        "$ llama-server ... -t 4 --jinja", "llama.cpp b9623 쨌 GGML_CPU_KLEIDIAI=ON", "$ python scripts/benchmark_proxy.py --cache-only",
        "A-common-prefix          5.60x", "B-mixed-realistic       8.80x", "C-adversarial-low-share 7.49x", "PASS: every workload >= 3.00x"
    ]
    y = 285
    for line in console:
        color = LIME if line.startswith(("aarch64", "A-", "B-", "C-", "PASS")) else (CYAN if line.startswith("$") else INK)
        d.text((135, y), line, font=font(28, line.startswith("PASS")), fill=color)
        y += 43
    d.text((90, 920), "Public run #35 쨌 github.com/lsh2546/mcp-prism/actions/runs/31783556529", font=font(25, True), fill=MUTED)
    im.save(path)


def slide_quality(path):
    im, d = base("Speed without sacrificing correctness", "04")
    d.text((90, 260), "Complete held-out workflows", font=font(62, True), fill=INK)
    d.rounded_rectangle((90, 410, 820, 760), 12, fill=PANEL, outline="#33423b")
    d.text((145, 460), "BASELINE", font=font(24, True), fill=RED)
    d.text((145, 545), "60%", font=font(112, True), fill=INK)
    d.rounded_rectangle((910, 410, 1640, 760), 12, fill=PANEL, outline=LIME, width=4)
    d.text((965, 460), "MCP PRISM", font=font(24, True), fill=LIME)
    d.text((965, 545), "85%", font=font(112, True), fill=LIME)
    d.text((90, 840), "100% first-step accuracy  쨌  0% wrong first calls  쨌  20 frozen held-out requests", font=font(29, True), fill=CYAN)
    im.save(path)


def slide_finish(path):
    im, d = base("Reusable Cloud AI infrastructure", "05")
    multiline(d, (90, 260), "MCP Prism removes wasted prefill before inference begins.", 76, bold=True, width=38)
    d.text((90, 570), "8.80횞 faster.  85% workflow success.  Native Arm64.", font=font(40, True), fill=LIME)
    d.rounded_rectangle((90, 700, 1470, 830), 10, fill=PANEL, outline="#33423b")
    d.text((130, 742), "github.com/lsh2546/mcp-prism", font=font(42, True), fill=CYAN)
    d.text((90, 920), "Apache-2.0 쨌 source, workflows, raw evidence and reproduction commands are public", font=font(25, True), fill=MUTED)
    im.save(path)


def main():
    out = Path("assets/video")
    out.mkdir(parents=True, exist_ok=True)
    makers = [slide_hero, slide_problem, slide_system, slide_arm, slide_quality, slide_finish]
    durations = [20, 22, 28, 30, 25, 20]
    for i, make in enumerate(makers): make(out / f"slide-{i}.png")
    lines = []
    for i, duration in enumerate(durations):
        lines += [f"file 'slide-{i}.png'", f"duration {duration}"]
    lines.append("file 'slide-5.png'")
    (out / "slides.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "narration.txt").write_text(
        "MCP Prism removes the MCP schema bottleneck on native Arm sixty-four. "
        "It reaches a peak eight point eight times throughput, improves complete held-out workflow success from sixty to eighty-five percent, and achieves one hundred percent first-step accuracy.\n\n"
        "The problem is simple. Agents often send every connected MCP tool definition to the model. A weather request may force the Arm CPU to prefill thousands of tokens describing email, source control, files, databases, and dozens of unrelated tools.\n\n"
        "MCP Prism collects tool definitions once. A pinned INT eight semantic model retrieves candidates. Service, action, schema, and dependency signals select the next executable workflow step. Canonical ordering creates stable prefixes, and llama dot cpp reuses the prompt and K V cache. Existing agents connect through an Open A I compatible endpoint.\n\n"
        "This is an actual native Arm sixty-four run. The host reports aarch sixty-four and four Neoverse N two cores. Llama dot cpp is built with Kleidi A I. With cache enabled, all three published workloads exceed the three-times gate: five point six, eight point eight, and seven point four nine times throughput.\n\n"
        "Quality is evaluated separately with complete outputs and deterministic tool results. On twenty frozen held-out workflows, the all-tools baseline completes sixty percent. MCP Prism completes eighty-five percent. Development first-step accuracy is one hundred percent with zero wrong first calls.\n\n"
        "MCP Prism is reusable Cloud A I infrastructure. Source code, native workflows, raw evidence, model hashes, and reproduction commands are public under Apache two point zero at github dot com slash lsh two five four six slash mcp prism.",
        encoding="utf-8"
    )


if __name__ == "__main__": main()

