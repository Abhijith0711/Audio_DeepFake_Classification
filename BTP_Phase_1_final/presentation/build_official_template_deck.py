#!/usr/bin/env python3
"""
Build Phase I final presentation using IIIT BTP Presentation Template-3.pptx.
Continues Review I deck (first ~6 slides) + 4 literature slides + methodology/results from report.
Target: 27–28 slides, bullet → formula → inference → figures where applicable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[2]
FIG_BTP = ROOT / "results" / "figures" / "btp_report"
FIG_E1 = ROOT / "results" / "figures" / "E1_v4_btp_p30"
FIG_E7 = ROOT / "results" / "figures" / "E7_ft_rcl_btp_p30"

TEMPLATE = Path("/Users/nikkathzabeenshaik/Downloads/BTP Presentation Template-3.pptx")
OUT = Path(__file__).resolve().parent / "BTP_Phase1_Final_Presentation.pptx"
OUT_DOWNLOADS = Path("/Users/nikkathzabeenshaik/Downloads/BTP_Phase1_Final_Presentation_Team29.pptx")


def delete_slide(prs: Presentation, index: int) -> None:
    slide_id_list = prs.slides._sldIdLst
    slide_id = slide_id_list[index]
    r_id = slide_id.rId
    prs.part.drop_rel(r_id)
    slide_id_list.remove(slide_id)


def _clear_body(tf) -> None:
    tf.clear()
    if not tf.paragraphs:
        return
    tf.paragraphs[0].text = ""


def add_bullet_block(
    tf,
    lines: list[str],
    *,
    base_size: int = 17,
    start_blank: bool = False,
) -> None:
    first = True
    for line in lines:
        if start_blank and first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
        if line.startswith("  "):
            p.text = line.strip()
            p.level = 1
            p.font.size = Pt(base_size - 2)
        elif line.startswith("FORMULA:"):
            p.text = line.replace("FORMULA:", "").strip()
            p.level = 0
            p.font.size = Pt(15)
            p.font.italic = True
        elif line.startswith("INFER:"):
            p.text = "→ " + line.replace("INFER:", "").strip()
            p.level = 0
            p.font.size = Pt(15)
            p.font.bold = True
        else:
            p.text = "• " + line if not line.startswith("•") else line
            p.level = 0
            p.font.size = Pt(base_size)


def add_content_slide(
    prs: Presentation,
    title: str,
    bullets: list[str],
    *,
    subtitle: str | None = None,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    if slide.shapes.title:
        slide.shapes.title.text = title
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.word_wrap = True
    _clear_body(tf)
    if subtitle:
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(14)
        p.font.italic = True
        add_bullet_block(tf, bullets, start_blank=True)
    else:
        add_bullet_block(tf, bullets)


def add_image_slide(
    prs: Presentation,
    title: str,
    bullets: list[str],
    image_path: Path,
    *,
    caption: str | None = None,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    if slide.shapes.title:
        slide.shapes.title.text = title
    # Narrow text column (template is wide 13.33")
    left = Inches(0.55)
    top = Inches(1.45)
    width = Inches(5.85)
    height = Inches(5.5)
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    add_bullet_block(tf, bullets, base_size=16)

    if image_path.is_file():
        img_left = Inches(6.55)
        img_top = Inches(1.35)
        img_w = Inches(6.35)
        slide.shapes.add_picture(str(image_path), img_left, img_top, width=img_w)
    else:
        miss = slide.shapes.add_textbox(Inches(6.8), Inches(3.0), Inches(5.5), Inches(1.0))
        miss.text_frame.text = f"[Missing figure: {image_path.name}]"

    if caption:
        cap = slide.shapes.add_textbox(Inches(6.55), Inches(6.55), Inches(6.35), Inches(0.45))
        cp = cap.text_frame.paragraphs[0]
        cp.text = caption
        cp.font.size = Pt(11)
        cp.font.italic = True
        cp.alignment = PP_ALIGN.CENTER


def add_dual_image_slide(
    prs: Presentation,
    title: str,
    bullets: list[str],
    left_img: Path,
    right_img: Path,
    left_cap: str,
    right_cap: str,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    if slide.shapes.title:
        slide.shapes.title.text = title
    box = slide.shapes.add_textbox(Inches(0.55), Inches(1.45), Inches(12.2), Inches(1.35))
    tf = box.text_frame
    tf.word_wrap = True
    add_bullet_block(tf, bullets, base_size=15)

    y = Inches(2.95)
    w = Inches(5.9)
    for x, path, cap in (
        (Inches(0.55), left_img, left_cap),
        (Inches(6.85), right_img, right_cap),
    ):
        if path.is_file():
            slide.shapes.add_picture(str(path), x, y, width=w)
        cbox = slide.shapes.add_textbox(x, Inches(6.35), w, Inches(0.4))
        cbox.text_frame.paragraphs[0].text = cap
        cbox.text_frame.paragraphs[0].font.size = Pt(11)
        cbox.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER


def add_table_slide(
    prs: Presentation,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    footnote: str | None = None,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    if slide.shapes.title:
        slide.shapes.title.text = title
    top = Inches(1.55)
    if footnote:
        fn = slide.shapes.add_textbox(Inches(0.6), Inches(1.35), Inches(12.0), Inches(0.35))
        fn.text_frame.paragraphs[0].text = footnote
        fn.text_frame.paragraphs[0].font.size = Pt(12)
        fn.text_frame.paragraphs[0].font.italic = True
        top = Inches(1.75)

    nrows = len(rows) + 1
    ncols = len(headers)
    table_shape = slide.shapes.add_table(
        nrows,
        ncols,
        Inches(0.75),
        top,
        Inches(11.8),
        Inches(0.38 * nrows),
    )
    table = table_shape.table
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(14)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(13)


def set_title_slide(prs: Presentation) -> None:
    slide = prs.slides[0]
    for shape in slide.shapes:
        if not shape.is_placeholder:
            if hasattr(shape, "text") and "Guided By" in shape.text:
                shape.text = "Guided By,\nDr. Sivaiah Bellamkonda"
            continue
        idx = shape.placeholder_format.idx
        if idx == 0:
            shape.text = (
                "Lightweight, Compression-Robust Audio Deepfake Detection "
                "for Edge Deployment"
            )
        elif idx == 1:
            shape.text = (
                "BY\n"
                "2023BCS0060 Nikkath Zabeen Shaik\n"
                "2023BCS0198 Bhuvan S M\n"
                "2023BCS0109 Abhijith Jayan\n"
                "2023BCS0087 Joshi Kshitij Santosh\n\n"
                "B.Tech Project — Phase I (Final Report Presentation)\n"
                "October 2026"
            )


def main() -> None:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    shutil.copy2(TEMPLATE, OUT)
    prs = Presentation(str(OUT))

    set_title_slide(prs)
    for idx in range(len(prs.slides) - 1, 0, -1):
        delete_slide(prs, idx)

    # --- Continuation of Review I (slides 2–6 content) ---
    add_content_slide(
        prs,
        "Introduction: Audio Deepfakes",
        [
            "Artificially generated or manipulated audio imitating real voices or acoustic scenes",
            "TTS (text-to-speech), voice conversion, vocoders, diffusion, and audio LLMs",
            "Threats: fraud, misinformation, forged evidence, voice-biometric bypass",
            "FORMULA: f_θ(x) → {bona-fide, synthetic/manipulated}  (binary detection)",
            "INFER: Type-agnostic Track 2 — category unknown at test time",
        ],
    )

    add_content_slide(
        prs,
        "Introduction: Accuracy, Robustness, and Efficiency",
        [
            "Reliable deployment needs all three jointly — not benchmark accuracy alone",
            "Clean-lab scores overstate performance on messaging, telephony, and social media",
            "Edge hardware limits model size (~1.5M params) and latency",
            "INFER: Train-time robustness without enlarging the inference graph (FT-RCL)",
        ],
    )

    add_content_slide(
        prs,
        "Motivation: Why Compression Robustness?",
        [
            "Operational audio is resampled, codec-compressed, band-limited, and noise-corrupted",
            "High-frequency synthesis cues may vanish after MP3/AAC or telephony band-limiting",
            "Compression can add artifacts on real speech → false alarms if the model is clean-only",
            "FORMULA: ΔM = M_clean − M_degraded  (Macro-F1, AUROC, or EER)",
            "INFER: Same semantic clip should yield a stable real/fake decision after channel change",
        ],
    )

    add_content_slide(
        prs,
        "Motivation: Consistent Decisions Under Degradation",
        [
            "Original waveform → channel (codec / resample / telephony) → received waveform",
            "Edge detector must classify received audio consistently with the source intent",
            "Robustness is measured by how little M drops under conditioned evaluation",
            "INFER: Phase 1 compares clean-only baseline (M0) vs FT-RCL training (M3)",
        ],
    )

    add_content_slide(
        prs,
        "Motivation: Edge Deployment",
        [
            "Cloud detection: latency, privacy, connectivity, and recurring cost",
            "On-device path: local inference after standard platform compression",
            "FORMULA: max_θ  detection quality  s.t.  params, memory, latency budgets",
            "INFER: MobileNetV3-Small (~1.52M) + 80-bin log-mel @ 24 kHz at inference",
        ],
    )

    # --- Literature (4 slides) ---
    add_content_slide(
        prs,
        "Literature Review (1/4): Benchmarks & Datasets",
        [
            "ASVspoof (LA/PA/DF): speech-centric spoofing; EER is standard metric",
            "AT-ADD Track 2: all-type real vs fake; Macro-F1 over speech/singing/music/sound",
            "Generator-disjoint eval splits reduce memorization of synthesis fingerprints",
            "Challenge winners report 96%+ Macro-F1 with large SSL and fusion",
            "INFER: We adopt AT-ADD Track 2 as primary benchmark for general audio",
        ],
    )

    add_content_slide(
        prs,
        "Literature Review (2/4): Architectures & Scale",
        [
            "CNN/CRNN on log-mel or LFCC: compact but struggle on music and ambient sound",
            "SSL encoders (Wav2Vec2, XLS-R): strong accuracy, heavy inference cost",
            "Audio LLMs + frequency-time RL: SOTA interpretability, not edge-deployable",
            "Multimodal (audio-visual) methods improve robustness with extra modalities",
            "INFER: Gap for ~1.5M-parameter models competitive across all audio types",
        ],
    )

    add_content_slide(
        prs,
        "Literature Review (3/4): Robustness & Compression",
        [
            "Data augmentation (codec/noise) improves robustness but lacks view pairing",
            "Condition-aware and coherence training (e.g., Vicomtech Wav2Vec2) motivate FT-RCL",
            "Audio compression (input) ≠ model compression (INT8) — both matter for edge",
            "Surveys emphasize generalization and adversarial robustness as open problems",
            "INFER: Explicit clean/degraded coherence + curriculum under-studied at edge scale",
        ],
    )

    add_content_slide(
        prs,
        "Literature Review (4/4): Synthesis & Research Gap",
        [
            "Detection ✓ | Robustness △ | Lightweight △ — rarely optimized together",
            "Music/environmental sound remain hardest domains in compact mel-CNN systems",
            "Few student-scale pipelines report clean metrics and ΔM under one protocol",
            "Research question: Can a single lightweight detector generalize across types while tolerating compression-like degradations?",
            "INFER: FT-RCL keeps inference identical to baseline; robustness is training-time only",
        ],
    )

    add_content_slide(
        prs,
        "Objectives (Phase 1)",
        [
            "Reproduce E1 v4 baseline: MobileNetV3-Small, 80-bin log-mel @ 24 kHz, repeat-pad",
            "Implement FT-RCL M1–M3: paired degradations, PC, FTSC, gating, severity curriculum",
            "Evaluate clean Macro-F1, AUROC, EER; per-type breakdown on 30% stratified subset",
            "Frontend & backbone ablation (E1 CNN vs E2 CRNN) on full AT-ADD where reported",
            "Planned next: systematic ΔM, STAR-Edge eval, INT8/latency on target edge device",
        ],
    )

    add_image_slide(
        prs,
        "Methodology: End-to-End Pipeline",
        [
            "Manifest CSV → load 24 kHz → 4 s crop (repeat-pad) → log-mel → classifier",
            "Training (M3): identical crop index for clean x and degraded x̃",
            "Eval: clean clips only for primary tables; conditioned eval for ΔM",
            "INFER: Single mel → MobileNet path at inference for both systems",
        ],
        FIG_BTP / "pipeline_overview.png",
        caption="Pipeline overview (report Figure)",
    )

    add_image_slide(
        prs,
        "Acoustic Frontend (E1 v4 Standard)",
        [
            "24 kHz sampling (captures 8–12 kHz vocoder artifacts vs 16 kHz telephony)",
            "Circular repeat-pad avoids duration shortcuts from trailing silence (ASVspoof study)",
            "80 mel bins; global mean/std normalization from training statistics",
            "FORMULA: x_padded[n] = x[n mod L]  for short clips",
            "INFER: Same frontend for baseline and FT-RCL ensures fair comparison",
        ],
        FIG_BTP / "mel_frontend.png",
    )

    add_image_slide(
        prs,
        "Baseline Detector (Mode M0)",
        [
            "MobileNetV3-Small adapted to 1-channel mel input (~1,519,618 parameters)",
            "Class-weighted cross-entropy on clean batches only",
            "Checkpoint: best development Macro-F1",
            "Full-split reference: 75.70% acc, 0.875 AUROC, 19.92% EER (E1 v4)",
            "INFER: Pure 2D CNN beats CRNN (GRU/LSTM) on full AT-ADD heterogeneous audio",
        ],
        FIG_BTP / "mobilenet_baseline.png",
    )

    add_image_slide(
        prs,
        "FT-RCL: Paired Clean / Degraded Views",
        [
            "Shared backbone weights; dual forward on clean and degraded mels",
            "Subband pooling → L2-normalized 8-D frequency signatures (training only)",
            "Degradations applied in waveform domain before identical mel extraction",
            "INFER: Inference graph unchanged — only one forward pass on received audio",
        ],
        FIG_BTP / "ft_rcl_paired.png",
    )

    add_image_slide(
        prs,
        "Loss Functions (Mode M3)",
        [
            "FORMULA: L_cls = CE(z_c,y) + 0.5·CE(z_d,y)",
            "FORMULA: L_PC = (1/B) Σ (q_c − q_d)²  (fake-class probability match)",
            "FORMULA: L_FTSC = (1/BK) Σ (1 − cos(s_c, s_d))  per mel subband",
            "FORMULA: L_M3 = L_cls + w̄(γ_pc·L_PC + γ_ftsc·L_FTSC);  w_i = α_t·κ_i·η(ρ_i)",
            "INFER: PC aligns decisions; FTSC aligns internal signatures under degradation",
        ],
        FIG_BTP / "pc_ftsc.png",
    )

    add_image_slide(
        prs,
        "Training Modes M0–M3 (Ablation Ladder)",
        [
            "M0: CE on clean mel only (baseline)",
            "M1: CE on degraded view only",
            "M2: L_cls + gated prediction coherence (η ≡ 1)",
            "M3: M2 + FTSC + type/confidence/severity gate + J-score checkpointing",
            "INFER: M3 is the proposed system compared against M0 on identical data",
        ],
        FIG_BTP / "ablation_m0_m3.png",
    )

    add_image_slide(
        prs,
        "Degradations & Severity Curriculum",
        [
            "Families: resample 16 kHz, MP3 bitrates, additive noise (SNR), telephony 300–3400 Hz",
            "Same random crop index ties clean and degraded waveforms",
            "Curriculum: severity cap 1 (epochs 1–5) → 3 (6–12) → 4 (13+)",
            "INFER: Gradual exposure reduces early training collapse on harsh degradations",
        ],
        FIG_BTP / "degradation_severity.png",
    )

    add_image_slide(
        prs,
        "Training vs Inference (Edge Path)",
        [
            "Training: paired views + PC/FTSC + optional STAR-Edge (ABR, STC, CAD)",
            "Inference: waveform → log-mel → MobileNet → softmax fake score",
            "No teacher network, no second forward, no signature head at deploy time",
            "INFER: Robustness bought during training; deploy-time cost equals baseline",
        ],
        FIG_BTP / "train_vs_infer.png",
    )

    add_content_slide(
        prs,
        "Evaluation Protocol & Metrics",
        [
            "Dataset: AT-ADD Track 2; phase runs on 30% stratified subset (seed 42)",
            "Eval subset: n = 13,287 clean clips; types balanced in Macro-F1",
            "Macro-F1 (primary), AUROC (ranking), EER (threshold-free error rate)",
            "FORMULA: J = Macro-F1_dev − 0.5·Δ̂Macro-F1_dev  (FT-RCL checkpoint)",
            "INFER: Baseline uses dev Macro-F1; FT-RCL uses J to favor robustness",
        ],
    )

    add_table_slide(
        prs,
        "Results: Clean Evaluation (30% subset)",
        ["Model", "Macro-F1", "Accuracy", "AUROC", "EER"],
        [
            ["Baseline (E1 / M0)", "0.672", "0.759", "0.852", "0.216"],
            ["FT-RCL (M3)", "0.659", "0.748", "0.843", "0.234"],
            ["Ref. full-split E1 v4", "—", "75.7%", "0.875", "19.92%"],
        ],
        footnote="Clean eval only. Full-split E1 from complete AT-ADD training (report Table).",
    )

    add_table_slide(
        prs,
        "Results: Macro-F1 & EER by Audio Type",
        ["Type", "Baseline F1", "FT-RCL F1", "Baseline EER", "FT-RCL EER"],
        [
            ["Speech", "0.695", "0.665", "0.162", "0.179"],
            ["Singing", "0.794", "0.764", "0.157", "0.208"],
            ["Sound", "0.467", "0.522", "0.426", "0.394"],
            ["Music", "0.393", "0.403", "0.399", "0.467"],
        ],
        footnote="Type-averaged Macro-F1 ≈ 0.587 (baseline) vs 0.589 (FT-RCL) — tied; music remains hardest.",
    )

    add_dual_image_slide(
        prs,
        "Training Dynamics (Development Set)",
        [
            "Baseline peak dev Macro-F1 ≈ 0.930 @ epoch 20",
            "FT-RCL peak dev Macro-F1 ≈ 0.909, best J ≈ 0.902 @ epoch 12",
            "M3 train loss unreliable under FP16 — select checkpoints via dev Macro-F1 / J",
        ],
        FIG_E1 / "dev_macro_f1_auroc.png",
        FIG_E7 / "dev_macro_f1_auroc.png",
        "Baseline (M0)",
        "FT-RCL (M3)",
    )

    add_dual_image_slide(
        prs,
        "Evaluation Curves & Confusion (Clean)",
        [
            "ROC and confusion matrices at threshold 0.5 on held-out eval",
            "Both models struggle most on music (high EER ~0.40)",
            "INFER: Per-type errors drive low type-averaged Macro-F1 vs pooled accuracy",
        ],
        FIG_E1 / "confusion_matrix_eval.png",
        FIG_E7 / "confusion_matrix_eval.png",
        "Baseline confusion",
        "FT-RCL confusion",
    )

    add_content_slide(
        prs,
        "Conclusion, Limitations & Future Work",
        [
            "Established E1 v4 mel-CNN baseline and FT-RCL M3 on shared AT-ADD protocol",
            "Frontend repeat-pad @ 24 kHz and CNN-over-CRNN choice supported by ablations",
            "Limitation: 30% subset; primary tables are clean-only — ΔM sweep in progress",
            "Limitation: music domain still ~40% EER for both systems",
            "Future: full-split training, STAR-Edge held-out eval, INT8/latency edge benchmark",
            "INFER: Lightweight inference preserved; next phase quantifies compression robustness (ΔM)",
        ],
    )

    refs = [
        "[1] Y. Xie et al., “AT-ADD: A benchmark and challenge for robust all-type audio deepfake detection,” arXiv:2608.23437, 2026.",
        "[2] A. Howard et al., “Searching for MobileNetV3,” arXiv:1905.02244, 2019.",
        "[3] T. Wang et al., ASVspoof 2019 challenge evaluation plan, Computer Speech & Language, 2020.",
        "[4] X. Li, P.-Y. Chen, W. Wei, “Where are we in audio deepfake detection?,” ACM TOIT, 2025.",
        "[5] J. M. Martín-Doñas and A. Álvarez, Vicomtech Wav2Vec2 ADD system, ICASSP 2022.",
        "[6] Y. Kim et al., “X-AVDT: Audio-visual cross-attention for robust deepfake detection,” arXiv:2603.08483, 2026.",
        "[7] H.-H. Nguyen-Le et al., “Deepfake Detection Across Image, Video, and Audio,” AI Review, 2026.",
        "[8] Y. Xie et al., “Interpretable All-Type Audio Deepfake Detection with Audio LLMs,” arXiv:2601.02983, 2026.",
        "[9] H. Pham et al., deepfake detection with ImageNet-pretrained CNNs, 2024 (reproduced in report).",
        "[10] B. Zhang et al., “Audio Deepfake Detection: What Has Been Achieved,” Sensors, 2025.",
    ]
    add_content_slide(
        prs,
        "References (IEEE Style)",
        refs,
    )

    add_content_slide(
        prs,
        "Thank You",
        [
            "Questions and discussion",
            "Dr. Sivaiah Bellamkonda — Project Guide",
            "Team 29 — IIIT Kottayam, Department of CSE",
        ],
    )

    prs.save(str(OUT))
    try:
        shutil.copy2(OUT, OUT_DOWNLOADS)
        print(f"Also copied to {OUT_DOWNLOADS}")
    except OSError as e:
        print(f"Could not copy to Downloads: {e}")

    print(f"Created {OUT} with {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
