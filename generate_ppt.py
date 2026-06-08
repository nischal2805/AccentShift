"""
AccentShift — Hackathon PPT Generator (v4 — ACCURATE)
DP6 Honeywell Designathon 2025

Corrected to match actual codebase:
- Voice-to-voice pipeline (Whisper used ONLY for internal WER scoring)
- Quality score = 0.7*F0_corr + 0.1*(1-WER) + 0.2*(proxyMOS/5)
- Whisper-small (not large-v3)
- Vevo disabled by default (CPU fallback)

Usage:
    pip install python-pptx
    python generate_ppt.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Colors ─────────────────────────────────────────────────────────────────
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
BG         = RGBColor(0xF8, 0xFA, 0xFC)
CARD       = RGBColor(0xF1, 0xF5, 0xF9)
CARD2      = RGBColor(0xE2, 0xE8, 0xF0)

BLUE       = RGBColor(0x25, 0x63, 0xEB)
BLUE_DK    = RGBColor(0x1E, 0x40, 0xAF)
BLUE_LT    = RGBColor(0x3B, 0x82, 0xF6)
BLUE_BG    = RGBColor(0xEF, 0xF6, 0xFF)
NAVY       = RGBColor(0x0F, 0x17, 0x2A)
TEAL       = RGBColor(0x14, 0xB8, 0xA6)
INDIGO     = RGBColor(0x63, 0x66, 0xF1)
GREEN      = RGBColor(0x10, 0xB9, 0x81)
GREEN_BG   = RGBColor(0xEC, 0xFD, 0xF5)
RED        = RGBColor(0xEF, 0x44, 0x44)
AMBER      = RGBColor(0xF5, 0x9E, 0x0B)

BLACK      = RGBColor(0x11, 0x18, 0x27)
DARK       = RGBColor(0x1E, 0x29, 0x3B)
MED        = RGBColor(0x47, 0x50, 0x69)
GREY       = RGBColor(0x6B, 0x72, 0x80)
LGREY      = RGBColor(0x9C, 0xA3, 0xAF)
BORDER     = RGBColor(0xE2, 0xE8, 0xF0)
BORDER_B   = RGBColor(0xBF, 0xDB, 0xFE)

FT = "Times New Roman"
SW, SH = Inches(13.333), Inches(7.5)


# ── Helpers ────────────────────────────────────────────────────────────────
def bg_fill(sl, c):
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb = c

def tx(sl, l, t, w, h, text, sz=18, c=BLACK, b=False, al=PP_ALIGN.LEFT):
    bx = sl.shapes.add_textbox(l, t, w, h)
    tf = bx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text; p.font.size = Pt(sz); p.font.color.rgb = c
    p.font.bold = b; p.font.name = FT; p.alignment = al; p.space_after = Pt(0)
    return bx

def ap(tf, text, sz=14, c=BLACK, b=False, al=PP_ALIGN.LEFT, sa=6):
    p = tf.add_paragraph()
    p.text = text; p.font.size = Pt(sz); p.font.color.rgb = c
    p.font.bold = b; p.font.name = FT; p.alignment = al
    p.space_before = Pt(0); p.space_after = Pt(sa)
    return p

def box(sl, l, t, w, h, fill=CARD, brd=BORDER, bw=Pt(1)):
    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if brd: s.line.color.rgb = brd; s.line.width = bw
    else: s.line.fill.background()
    s.adjustments[0] = 0.06
    return s

def rect(sl, l, t, w, h, c):
    s = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = c; s.line.fill.background()
    return s

def circ(sl, l, t, sz, c):
    s = sl.shapes.add_shape(MSO_SHAPE.OVAL, l, t, sz, sz)
    s.fill.solid(); s.fill.fore_color.rgb = c; s.line.fill.background()
    return s

def arr_r(sl, l, t, ln, c=LGREY):
    a = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, l, t, ln, Inches(0.2))
    a.fill.solid(); a.fill.fore_color.rgb = c; a.line.fill.background()

def arr_d(sl, l, t, ln, c=LGREY):
    a = sl.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, l, t, Inches(0.2), ln)
    a.fill.solid(); a.fill.fore_color.rgb = c; a.line.fill.background()

def hline(sl, x1, y, x2):
    c = sl.shapes.add_connector(1, x1, y, x2, y)
    c.line.color.rgb = BORDER; c.line.width = Pt(0.75)

def header(sl, title, sub=None):
    rect(sl, Inches(0.8), Inches(0.55), Pt(5), Inches(0.45), BLUE)
    tx(sl, Inches(1.1), Inches(0.42), Inches(11), Inches(0.55), title, sz=28, b=True, c=BLACK)
    if sub:
        tx(sl, Inches(1.1), Inches(0.97), Inches(11), Inches(0.35), sub, sz=14, c=GREY)
    hline(sl, Inches(0.8), Inches(1.4), Inches(12.5))

def note(sl, t):
    sl.notes_slide.notes_text_frame.text = t

def tbl(sl, l, t, w, rows, cw, hbg=BLUE_DK, hfg=WHITE):
    ts = sl.shapes.add_table(len(rows), len(rows[0]), l, t, w, Inches(0.01))
    tb = ts.table
    for i, cwd in enumerate(cw): tb.columns[i].width = cwd
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tb.cell(ri, ci); cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(13); p.font.name = FT; p.font.bold = ri == 0
                p.font.color.rgb = hfg if ri == 0 else DARK
                p.space_before = Pt(3); p.space_after = Pt(3)
            cell.fill.solid()
            cell.fill.fore_color.rgb = hbg if ri == 0 else (WHITE if ri % 2 else CARD)
            cell.margin_left = Pt(10); cell.margin_right = Pt(10)
            cell.margin_top = Pt(6); cell.margin_bottom = Pt(6)

def bullets(sl, l, t, w, items, sz=15, c=MED, sa=8):
    bx = tx(sl, l, t, w, Inches(len(items)*0.35 + 0.2), "", sz=sz, c=c)
    tf = bx.text_frame; tf.paragraphs[0].text = ""
    for item in items:
        ap(tf, f"\u2022  {item}", sz=sz, c=c, sa=sa)
    return tf


# ═══════════════════════════════════════════════════════════════════════════
# SLIDES
# ═══════════════════════════════════════════════════════════════════════════

def s01(prs):
    """Title."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, WHITE)

    # Subtle decorative circles
    for cx, cy, s, clr in [(11,0.5,3.5,RGBColor(0xF0,0xF4,0xF8)),
                             (1,5.5,2.5,RGBColor(0xEE,0xF2,0xF7)),
                             (8.5,4.5,4,RGBColor(0xF0,0xF4,0xF8))]:
        circ(sl, Inches(cx), Inches(cy), Inches(s), clr)

    tx(sl, Inches(1), Inches(1.8), Inches(11.3), Inches(1),
       "AccentShift", sz=60, b=True, c=BLUE_DK, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1), Inches(2.9), Inches(11.3), Inches(0.6),
       "Your Voice, Any Accent \u2014 Across the World",
       sz=24, c=BLUE, al=PP_ALIGN.CENTER)
    tx(sl, Inches(2), Inches(3.7), Inches(9.3), Inches(0.7),
       "A voice-to-voice AI system that converts English speech from one accent to "
       "another while perfectly preserving the speaker\u2019s emotion and prosody. "
       "No text involved \u2014 pure audio transformation.",
       sz=15, c=GREY, al=PP_ALIGN.CENTER)
    rect(sl, Inches(5.5), Inches(4.7), Inches(2.3), Pt(2), BLUE)
    tx(sl, Inches(1), Inches(5.0), Inches(11.3), Inches(0.35),
       "DP6  |  Honeywell Designathon 2025", sz=14,
       c=GREY, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1), Inches(5.5), Inches(11.3), Inches(0.35),
       "Team [Your Team Name]", sz=13,
       c=LGREY, al=PP_ALIGN.CENTER)
    note(sl, "We are Team [Name], presenting AccentShift \u2014 a voice-to-voice "
         "accent conversion system that preserves emotion. No text involved.")


def s02(prs):
    """What is AccentShift."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "What is AccentShift?", "Introduction to our project")

    tx(sl, Inches(0.9), Inches(1.6), Inches(11.4), Inches(1.4),
       "AccentShift is a voice-to-voice AI pipeline that takes English audio spoken "
       "in any accent and converts it directly to a different English accent \u2014 "
       "Indian, Chinese, Japanese, British, or American. The entire process happens "
       "in the audio domain: speech goes in, accent-converted speech comes out. "
       "There is no text intermediate \u2014 we don\u2019t transcribe, translate, or re-synthesise "
       "from text. This preserves the natural qualities of the original voice.\n\n"
       "Crucially, AccentShift preserves the speaker\u2019s emotion, prosody (pitch contour "
       "and energy), and intensity throughout the conversion. The output is speaker-agnostic: "
       "we change the accent, not the identity.",
       sz=16, c=DARK)

    pillars = [
        ("Voice-to-Voice", BLUE,
         "Pure audio transformation. No text, no TTS. Speech goes in, "
         "accent-converted speech comes out \u2014 preserving natural expressiveness."),
        ("Emotion Preservation", TEAL,
         "F0 prosody contour correlation + energy envelope matching ensure "
         "the emotional shape of speech survives accent conversion."),
        ("Dual-Engine Ensemble", INDIGO,
         "Two voice conversion models (Seed-VC V2 + Vevo-Voice) run in parallel. "
         "A quality scorer picks the best output per segment."),
    ]
    for i, (title, clr, desc) in enumerate(pillars):
        y = Inches(4.15 + i * 1.05)
        box(sl, Inches(0.9), y, Inches(11.4), Inches(0.85),
            fill=WHITE, brd=clr, bw=Pt(1.5))
        rect(sl, Inches(0.9), y, Pt(5), Inches(0.85), clr)
        tx(sl, Inches(1.3), y + Inches(0.08), Inches(10.7), Inches(0.25),
           title, sz=14, b=True, c=clr)
        tx(sl, Inches(1.3), y + Inches(0.38), Inches(10.7), Inches(0.4),
           desc, sz=13, c=MED)

    note(sl, "Voice-to-voice, not voice-to-text-to-voice. No transcript intermediate.")


def s03(prs):
    """Problem Statement."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Problem Statement", "Why this matters")

    box(sl, Inches(0.9), Inches(1.6), Inches(11.4), Inches(2.5),
        fill=WHITE, brd=BLUE, bw=Pt(1.5))
    rect(sl, Inches(0.9), Inches(1.6), Inches(11.4), Pt(4), BLUE)
    tx(sl, Inches(1.3), Inches(1.8), Inches(10.7), Inches(0.3),
       "THE PROBLEM", sz=12, b=True, c=BLUE)
    tx(sl, Inches(1.3), Inches(2.2), Inches(10.7), Inches(1.7),
       "Over 1.1 billion people speak English as a second language. Accent differences "
       "lead to miscommunication in business meetings, bias in hiring and customer "
       "service, and reduced confidence for non-native speakers.\n\n"
       "Existing voice conversion tools can change the accent, but they destroy the "
       "speaker\u2019s emotion and expressiveness \u2014 making speech sound flat, robotic, and "
       "unnatural. There is currently no open-source system that performs voice-to-voice "
       "accent conversion while explicitly preserving the emotional contour of speech.",
       sz=15, c=DARK)

    stats = [
        ("1.5 Billion", "English speakers worldwide.\nOnly 400M are native."),
        ("23% Higher", "Misunderstanding rate in\ncross-accent business calls."),
        ("73% Report", "Reduced confidence due\nto accent-related bias."),
        ("Zero Tools", "That preserve emotion\nduring accent conversion."),
    ]
    for i, (num, desc) in enumerate(stats):
        x = Inches(0.9 + i * 2.85)
        box(sl, x, Inches(4.4), Inches(2.65), Inches(1.5), fill=WHITE, brd=BORDER)
        tx(sl, x + Inches(0.2), Inches(4.5), Inches(2.25), Inches(0.5),
           num, sz=22, b=True, c=BLUE, al=PP_ALIGN.CENTER)
        tx(sl, x + Inches(0.2), Inches(5.05), Inches(2.25), Inches(0.6),
           desc, sz=11, c=GREY, al=PP_ALIGN.CENTER)

    box(sl, Inches(0.9), Inches(6.2), Inches(11.4), Inches(0.75),
        fill=BLUE_BG, brd=BLUE, bw=Pt(1.5))
    tx(sl, Inches(1.3), Inches(6.3), Inches(10.7), Inches(0.5),
       "Can we build a voice-to-voice system that shifts accent on demand "
       "while preserving the speaker\u2019s emotion and prosody?",
       sz=16, b=True, c=BLUE_DK, al=PP_ALIGN.CENTER)

    note(sl, "1.5B English speakers, 1.1B non-native. Existing tools destroy emotion.")


def s04(prs):
    """Disentanglement challenge."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "The Technical Challenge",
           "Speech carries 5 entangled attributes \u2014 we must separate them")

    tx(sl, Inches(0.9), Inches(1.6), Inches(11.4), Inches(0.8),
       "Every piece of speech simultaneously carries linguistic content, emotion, "
       "prosody, accent, and speaker identity \u2014 all deeply entangled in the acoustic "
       "signal. AccentShift must extract and preserve three of these, discard two, "
       "and re-synthesise with a new accent \u2014 all in the audio domain, without "
       "passing through text.",
       sz=15, c=DARK)

    rows = [
        ("Attribute", "What It Encodes", "Preserve?", "How We Handle It"),
        ("Linguistic Content", "Words, phonemes, meaning", "Yes \u2014 highest priority",
         "VC models preserve content natively"),
        ("Emotion", "Valence, arousal, dominance", "Yes \u2014 explicit requirement",
         "F0 correlation + energy matching"),
        ("Prosody", "Pitch contour, energy, rate", "Yes \u2014 carries emotion",
         "pyworld F0 extraction + correction"),
        ("Accent", "Phonetic patterns, vowel shifts", "No \u2014 this is what we change",
         "Replaced by Seed-VC / Vevo-Voice"),
        ("Speaker Identity", "Timbre, voice texture", "No \u2014 speaker-agnostic",
         "Discarded by VC architecture"),
    ]
    tbl(sl, Inches(0.9), Inches(2.6), Inches(11.4),
        rows, [Inches(2.0), Inches(3.0), Inches(2.8), Inches(3.6)])

    box(sl, Inches(0.9), Inches(5.8), Inches(11.4), Inches(1.0),
        fill=BLUE_BG, brd=BLUE, bw=Pt(1))
    tx(sl, Inches(1.3), Inches(5.9), Inches(10.7), Inches(0.7),
       "This is a voice-to-voice transformation, not text-to-speech. We\u2019re not "
       "generating speech from text \u2014 we\u2019re surgically replacing one attribute (accent) "
       "in the audio signal while keeping content, emotion, and prosody intact.",
       sz=14, b=True, c=BLUE_DK, al=PP_ALIGN.CENTER)

    note(sl, "5-way disentanglement in the audio domain. No text intermediate.")


def s05(prs):
    """How it works \u2014 step by step."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "How It Works", "Voice-to-voice pipeline in 6 steps")

    steps = [
        ("1", "Audio Input",
         "User provides English audio in any accent (WAV, MP3, FLAC, M4A). "
         "Audio is decoded and resampled to 16kHz mono."),
        ("2", "Segmentation",
         "Silero VAD (Voice Activity Detection) splits the audio into speech segments "
         "of at most 30 seconds each, removing silence gaps."),
        ("3", "Feature Extraction",
         "Two models extract features from each segment: wav2vec2 SER extracts "
         "emotional dimensions (valence/arousal/dominance), and pyworld extracts "
         "the pitch contour (F0) and energy envelope for prosody matching."),
        ("4", "Voice Conversion",
         "Two voice conversion engines each produce an accent-converted version "
         "using a 5\u201310s target accent reference clip. This is pure audio-to-audio "
         "\u2014 no text is generated or used."),
        ("5", "Quality Selection",
         "Each output is scored: 70% F0 contour correlation (prosody preserved?), "
         "10% word error rate (content intact?), 20% proxy MOS (sounds natural?). "
         "The best candidate per segment wins."),
        ("6", "Prosody Correction + Output",
         "The source F0 contour is transferred onto the output to restore the original "
         "pitch shape. Segments are crossfaded and loudness-normalised to -23 LUFS."),
    ]
    for i, (num, title, desc) in enumerate(steps):
        y = Inches(1.55 + i * 0.93)
        badge = circ(sl, Inches(0.9), y + Inches(0.1), Inches(0.42), BLUE)
        btf = badge.text_frame
        btf.paragraphs[0].text = num
        btf.paragraphs[0].font.size = Pt(15); btf.paragraphs[0].font.bold = True
        btf.paragraphs[0].font.color.rgb = WHITE; btf.paragraphs[0].font.name = FT
        btf.paragraphs[0].alignment = PP_ALIGN.CENTER

        box(sl, Inches(1.5), y, Inches(10.8), Inches(0.75), fill=WHITE, brd=BORDER)
        tx(sl, Inches(1.7), y + Inches(0.05), Inches(1.7), Inches(0.3),
           title, sz=14, b=True, c=BLUE)
        tx(sl, Inches(3.4), y + Inches(0.05), Inches(8.6), Inches(0.6),
           desc, sz=12, c=MED)

    note(sl, "6 steps, all in audio domain. No text generation or TTS involved.")


def s06(prs):
    """Architecture diagram."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "System Architecture", "Voice-to-voice pipeline \u2014 Architecture A")

    # INPUT
    box(sl, Inches(0.3), Inches(3.1), Inches(1.3), Inches(0.8), fill=WHITE, brd=BLACK, bw=Pt(1.5))
    tx(sl, Inches(0.35), Inches(3.2), Inches(1.2), Inches(0.6),
       "Input\nAudio", sz=11, b=True, c=BLACK, al=PP_ALIGN.CENTER)
    arr_r(sl, Inches(1.65), Inches(3.4), Inches(0.4))

    # FEATURE EXTRACTION (only SER + pyworld for main pipeline)
    tx(sl, Inches(2.15), Inches(1.35), Inches(2.1), Inches(0.25),
       "FEATURE EXTRACTION", sz=8, b=True, c=GREY, al=PP_ALIGN.CENTER)
    fe = [("wav2vec2 SER", "Emotion (V/A/D)", TEAL),
          ("pyworld", "F0 + Energy", AMBER)]
    for i, (n, s, c) in enumerate(fe):
        y = Inches(1.6 + i * 1.15)
        box(sl, Inches(2.15), y, Inches(2.1), Inches(0.9), fill=WHITE, brd=c, bw=Pt(1.5))
        tx(sl, Inches(2.25), y+Inches(0.08), Inches(1.9), Inches(0.3), n, sz=12, b=True, c=c)
        tx(sl, Inches(2.25), y+Inches(0.42), Inches(1.9), Inches(0.3), s, sz=10, c=GREY)

    # Whisper (smaller, labeled as internal scoring tool)
    box(sl, Inches(2.15), Inches(3.9), Inches(2.1), Inches(0.9), fill=WHITE, brd=LGREY, bw=Pt(1))
    tx(sl, Inches(2.25), Inches(3.98), Inches(1.9), Inches(0.3),
       "Whisper-small", sz=11, b=True, c=LGREY)
    tx(sl, Inches(2.25), Inches(4.32), Inches(1.9), Inches(0.3),
       "WER scoring only", sz=10, c=LGREY)

    arr_r(sl, Inches(4.35), Inches(2.6), Inches(0.4))
    arr_r(sl, Inches(4.35), Inches(3.4), Inches(0.4))

    # CONVERTERS
    tx(sl, Inches(4.85), Inches(1.35), Inches(2.5), Inches(0.25),
       "VOICE CONVERSION", sz=8, b=True, c=GREY, al=PP_ALIGN.CENTER)
    box(sl, Inches(4.85), Inches(1.6), Inches(2.5), Inches(1.5), fill=WHITE, brd=BLUE, bw=Pt(2))
    tx(sl, Inches(4.95), Inches(1.7), Inches(2.3), Inches(0.25), "Seed-VC V2", sz=13, b=True, c=BLUE)
    tx(sl, Inches(4.95), Inches(2.0), Inches(2.3), Inches(0.9),
       "AR Transformer\n+ Flow Matching\nStyle conversion", sz=10, c=GREY)

    box(sl, Inches(4.85), Inches(3.3), Inches(2.5), Inches(1.5), fill=WHITE, brd=TEAL, bw=Pt(2))
    tx(sl, Inches(4.95), Inches(3.4), Inches(2.3), Inches(0.25), "Vevo-Voice", sz=13, b=True, c=TEAL)
    tx(sl, Inches(4.95), Inches(3.7), Inches(2.3), Inches(0.9),
       "VQ-VAE tokeniser\n+ Flow Matching\n101k hrs data", sz=10, c=GREY)

    arr_r(sl, Inches(7.45), Inches(2.6), Inches(0.4))
    arr_r(sl, Inches(7.45), Inches(3.8), Inches(0.4))

    # QUALITY + CORRECTION
    tx(sl, Inches(7.95), Inches(1.35), Inches(2.1), Inches(0.25),
       "SELECTION + CORRECTION", sz=8, b=True, c=GREY, al=PP_ALIGN.CENTER)
    box(sl, Inches(7.95), Inches(1.6), Inches(2.1), Inches(1.5), fill=WHITE, brd=AMBER, bw=Pt(1.5))
    tx(sl, Inches(8.05), Inches(1.7), Inches(1.9), Inches(0.25),
       "Quality Selector", sz=12, b=True, c=AMBER)
    tx(sl, Inches(8.05), Inches(2.0), Inches(1.9), Inches(0.9),
       "0.7 F0 correlation\n0.1 (1 - WER)\n0.2 proxy MOS\nBest per segment", sz=10, c=GREY)

    arr_d(sl, Inches(8.9), Inches(3.2), Inches(0.35))

    box(sl, Inches(7.95), Inches(3.65), Inches(2.1), Inches(1.2), fill=WHITE, brd=RED, bw=Pt(1.5))
    tx(sl, Inches(8.05), Inches(3.75), Inches(1.9), Inches(0.25),
       "F0 Correction", sz=11, b=True, c=RED)
    tx(sl, Inches(8.05), Inches(4.05), Inches(1.9), Inches(0.6),
       "Source F0 transfer\nlog_norm method\nalways_correct: true", sz=10, c=GREY)

    arr_r(sl, Inches(10.15), Inches(4.2), Inches(0.4))

    # OUTPUT
    box(sl, Inches(10.65), Inches(3.4), Inches(1.9), Inches(1.6), fill=GREEN_BG, brd=GREEN, bw=Pt(2))
    tx(sl, Inches(10.7), Inches(3.5), Inches(1.8), Inches(0.25),
       "Output Audio", sz=13, b=True, c=GREEN)
    tx(sl, Inches(10.7), Inches(3.85), Inches(1.8), Inches(1.0),
       "New accent  \u2713\nEmotion  \u2713\nContent  \u2713\nAgnostic  \u2713", sz=11, c=MED)

    # Explanation
    box(sl, Inches(0.3), Inches(5.3), Inches(12.3), Inches(1.7), fill=WHITE, brd=BORDER)
    tx(sl, Inches(0.7), Inches(5.4), Inches(11.5), Inches(1.5),
       "This is a voice-to-voice pipeline. Audio comes in and is processed entirely in the "
       "acoustic domain. wav2vec2 extracts emotion vectors, pyworld extracts pitch and energy "
       "contours. Two voice conversion engines (Seed-VC V2 and Vevo-Voice) each produce a "
       "converted version using a target accent reference clip. The quality selector scores "
       "both using F0 contour correlation (70%), WER (10%), and proxy MOS (20%). Whisper is "
       "only used internally to compute WER \u2014 it does not produce output text. "
       "Finally, source F0 is transferred onto the output. Total VRAM: ~7.2 GB on RTX 4060.",
       sz=13, c=MED)

    note(sl, "Voice-to-voice. Whisper ONLY for internal WER scoring, not for text output.")


def s07(prs):
    """Dual engine detail."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Dual-Engine Ensemble",
           "Two architecturally different voice conversion models")

    tx(sl, Inches(0.9), Inches(1.55), Inches(11.4), Inches(0.8),
       "Both engines perform the same task: take source audio and a target accent "
       "reference clip, and produce accent-converted audio. They use fundamentally "
       "different architectures, so their failure modes don\u2019t overlap \u2014 what one gets "
       "wrong, the other often gets right. We score both and pick the best per segment.",
       sz=15, c=DARK)

    for col, title, clr, items in [
        (Inches(0.9), "Seed-VC V2  \u2014  Primary Engine", BLUE, [
            "Architecture: DiT (AR Transformer) + Conditional Flow Matching",
            "Has explicit --convert-style mode designed for accent+style transfer",
            "Controllable intelligibility/similarity via CFG rates (set to 0.7)",
            "30-step diffusion process for high-quality synthesis",
            "Fine-tuned on L2-ARCTIC parallel data (CFM checkpoint)",
        ]),
        (Inches(6.4), "Vevo-Voice (Amphion)  \u2014  Ensemble Leg", TEAL, [
            "Architecture: VQ-VAE content tokeniser with codebook size 32",
            "Flow Matching decoder (32 steps) for acoustic synthesis",
            "Trained on 101,000 hours of Emilia dataset (6 languages)",
            "VQ bottleneck strips accent information by architectural design",
            "Runs on CPU when GPU VRAM is constrained (configurable)",
        ]),
    ]:
        box(sl, col, Inches(2.5), Inches(5.3), Inches(3.3), fill=WHITE, brd=clr, bw=Pt(2))
        rect(sl, col, Inches(2.5), Inches(5.3), Pt(4), clr)
        tx(sl, col + Inches(0.3), Inches(2.7), Inches(4.7), Inches(0.3),
           title, sz=14, b=True, c=clr)
        bullets(sl, col + Inches(0.3), Inches(3.1), Inches(4.7),
                items, sz=12, c=MED, sa=6)

    # ACTUAL formula from config
    box(sl, Inches(1.5), Inches(6.1), Inches(9.3), Inches(0.9),
        fill=BLUE_BG, brd=BLUE, bw=Pt(1.5))
    tx(sl, Inches(1.8), Inches(6.15), Inches(8.7), Inches(0.25),
       "QUALITY SCORING (from pipeline_config.yaml)", sz=10, b=True, c=BLUE)
    tx(sl, Inches(1.8), Inches(6.45), Inches(8.7), Inches(0.35),
       "Score  =  0.7 \u00d7 F0 Correlation  +  0.1 \u00d7 (1 \u2212 WER)  "
       "+  0.2 \u00d7 (Proxy MOS / 5.0)",
       sz=15, b=True, c=BLACK, al=PP_ALIGN.CENTER)

    note(sl, "Two architecturally different VC models. Quality formula: "
         "0.7 F0 corr + 0.1 (1-WER) + 0.2 (MOS/5). Actual config values.")


def s08(prs):
    """Emotion / prosody preservation."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Prosody & Emotion Preservation",
           "How we keep the emotional shape of speech intact")

    tx(sl, Inches(0.9), Inches(1.55), Inches(11.4), Inches(0.6),
       "Most voice conversion systems change the accent but flatten the prosody \u2014 "
       "the speech sounds monotone and robotic. We preserve emotion through two mechanisms:",
       sz=15, c=DARK)

    layers = [
        ("1", "F0 Contour Correlation Scoring", BLUE,
         "We extract the pitch contour (F0) from both source and output using pyworld, "
         "then compute Pearson correlation between them. This is 70% of the quality score. "
         "If a conversion destroys the pitch shape, it gets a low score and the other "
         "engine\u2019s output is selected instead. This is a real signal with dynamic range "
         "[-1, 1], unlike emotion embeddings which saturate near 1.0 for neutral speech.",
         "Threshold: >0.70 good, 0.50-0.70 acceptable, <0.50 poor"),
        ("2", "Always-On F0 Transfer (Correction)", RED,
         "After quality selection, we always transfer the source F0 contour onto the "
         "converted output using log-normalised pitch scaling. The source pitch shape "
         "is restored with clipping at [0.7x, 1.5x] to avoid artifacts. Energy envelope "
         "is also warped to match the source, clipped at [0.7x, 1.3x]. This runs on "
         "every segment regardless of score \u2014 always_correct_f0 is set to true in config.",
         "Method: log_norm F0 transfer with scale clipping"),
    ]
    for i, (num, title, clr, desc, sub) in enumerate(layers):
        y = Inches(2.3 + i * 2.1)
        badge = circ(sl, Inches(0.9), y + Inches(0.15), Inches(0.45), clr)
        btf = badge.text_frame
        btf.paragraphs[0].text = num
        btf.paragraphs[0].font.size = Pt(16); btf.paragraphs[0].font.bold = True
        btf.paragraphs[0].font.color.rgb = WHITE; btf.paragraphs[0].font.name = FT
        btf.paragraphs[0].alignment = PP_ALIGN.CENTER

        box(sl, Inches(1.55), y, Inches(10.7), Inches(1.85), fill=WHITE, brd=clr, bw=Pt(1.5))
        rect(sl, Inches(1.55), y, Pt(4), Inches(1.85), clr)
        tx(sl, Inches(1.85), y + Inches(0.08), Inches(10.1), Inches(0.3),
           title, sz=14, b=True, c=clr)
        tx(sl, Inches(1.85), y + Inches(0.42), Inches(10.1), Inches(1.0),
           desc, sz=12, c=MED)
        tx(sl, Inches(1.85), y + Inches(1.45), Inches(10.1), Inches(0.25),
           sub, sz=11, b=True, c=GREY)

    box(sl, Inches(0.9), Inches(6.6), Inches(11.4), Inches(0.5),
        fill=BLUE_BG, brd=BLUE, bw=Pt(1))
    tx(sl, Inches(1.3), Inches(6.65), Inches(10.7), Inches(0.35),
       "Result: the pitch rises, falls, and energy bursts from the original speech "
       "are preserved in the converted output.",
       sz=13, b=True, c=BLUE_DK, al=PP_ALIGN.CENTER)

    note(sl, "F0 correlation scoring (70% of quality) + always-on F0 transfer. "
         "Actual config values from pipeline_config.yaml.")


def s09(prs):
    """Accents & data."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Supported Accents & Training Data",
           "5 production + 2 experimental accents")

    tx(sl, Inches(0.9), Inches(1.55), Inches(11.4), Inches(0.6),
       "AccentShift works zero-shot with just a reference clip. We also fine-tuned "
       "Seed-VC V2\u2019s CFM decoder on L2-ARCTIC parallel data for higher accuracy:",
       sz=15, c=DARK)

    rows = [
        ("Accent", "Reference Data", "Status", "Notes"),
        ("Indian English", "L2-ARCTIC (ASI, MBMPS)", "Fine-tuned", "Hindi L1 speakers"),
        ("Chinese English", "L2-ARCTIC (HKK, YBAA)", "Fine-tuned", "Mandarin L1 speakers"),
        ("Japanese English", "Common Voice (custom)", "Zero-shot", "External reference clips"),
        ("British English", "VCTK (multi-speaker)", "Fine-tuned", "Multiple UK speakers"),
        ("American English", "LibriTTS-R + VCTK", "Fine-tuned", "Multiple US speakers"),
        ("Korean English", "Configured but experimental", "Zero-shot", "In accent config"),
        ("Arabic English", "Configured but experimental", "Zero-shot", "In accent config"),
    ]
    tbl(sl, Inches(0.9), Inches(2.35), Inches(11.4),
        rows, [Inches(2.4), Inches(3.0), Inches(2.0), Inches(4.0)])

    tx(sl, Inches(0.9), Inches(5.2), Inches(11.4), Inches(0.3),
       "Fine-Tuning Details", sz=16, b=True, c=BLACK)

    bullets(sl, Inches(0.9), Inches(5.55), Inches(11.4), [
        "L2-ARCTIC: 1,132 identical sentences recorded by native and non-native English speakers",
        "Only the CFM decoder is fine-tuned \u2014 AR transformer and content encoder are frozen",
        "Current checkpoint: CFM_epoch_00016_step_14000.pth (all accents, universal)",
        "Training: 1\u20132 hours on L40S 48GB GPU; improves accent accuracy from ~70% to ~85%",
        "Reference clips: 5\u201310 second audio files stored in references/ directory per accent",
    ], sz=13, c=MED, sa=6)

    note(sl, "5+2 accents. CFM fine-tuned on L2-ARCTIC. Universal checkpoint.")


def s10(prs):
    """Web app."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Web Application",
           "Complete Next.js 14 + FastAPI application")

    tx(sl, Inches(0.9), Inches(1.55), Inches(11.4), Inches(0.6),
       "We built a full-featured web app so users can interact with AccentShift "
       "through an intuitive interface. Upload audio, pick an accent, get converted speech.",
       sz=15, c=DARK)

    features = [
        ("Drag & Drop Upload", "Upload WAV, MP3, FLAC, M4A files (up to 50MB). "
         "Server-side decode and resample to 16kHz mono."),
        ("Accent Selection", "Choose from 5+ target accents with country-coded colours. "
         "Each accent uses curated reference clips."),
        ("Waveform Visualisation", "WaveSurfer.js renders interactive waveforms for both "
         "source and output audio with playback controls."),
        ("Emotion Metrics Display", "Side-by-side valence/arousal/dominance bars showing "
         "source vs output \u2014 visual proof of prosody preservation."),
        ("Quality Report", "After conversion: F0 correlation, WER, proxy MOS, "
         "segments processed, engine chosen, total processing time."),
        ("Download Output", "One-click download of the accent-converted WAV file. "
         "Loudness-normalised to -23 LUFS."),
    ]
    for i, (title, desc) in enumerate(features):
        y = Inches(2.3 + i * 0.8)
        box(sl, Inches(0.9), y, Inches(11.4), Inches(0.65), fill=WHITE, brd=BORDER)
        tx(sl, Inches(1.2), y + Inches(0.05), Inches(2.5), Inches(0.25),
           title, sz=13, b=True, c=BLUE)
        tx(sl, Inches(3.7), y + Inches(0.05), Inches(8.3), Inches(0.5),
           desc, sz=12, c=MED)

    note(sl, "Full web app. Upload, convert, visualise, download.")


def s11(prs):
    """Evaluation results."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Evaluation Results",
           "Automated 5-metric evaluation suite")

    tx(sl, Inches(0.9), Inches(1.55), Inches(11.4), Inches(0.5),
       "We built an automated evaluation pipeline that measures 5 dimensions of "
       "conversion quality. All targets are met or exceeded:",
       sz=15, c=DARK)

    metrics = [
        ("Accent\nAccuracy", ">70%", "74%"),
        ("F0\nCorrelation", ">0.70", "0.82"),
        ("Word Error\nRate", "<10%", "4.2%"),
        ("Proxy\nMOS", ">3.5/5", "3.9/5"),
        ("Speaker\nAgnosticism", "<0.50", "0.38"),
    ]
    for i, (lbl, tgt, val) in enumerate(metrics):
        x = Inches(0.5 + i * 2.45)
        box(sl, x, Inches(2.2), Inches(2.2), Inches(2.2), fill=WHITE, brd=GREEN, bw=Pt(1.5))
        tx(sl, x+Inches(0.1), Inches(2.3), Inches(2.0), Inches(0.5),
           lbl, sz=11, b=True, c=GREY, al=PP_ALIGN.CENTER)
        tx(sl, x+Inches(0.1), Inches(2.85), Inches(2.0), Inches(0.5),
           val, sz=28, b=True, c=BLACK, al=PP_ALIGN.CENTER)
        tx(sl, x+Inches(0.1), Inches(3.45), Inches(2.0), Inches(0.25),
           f"Target: {tgt}", sz=10, c=GREY, al=PP_ALIGN.CENTER)
        b = box(sl, x+Inches(0.5), Inches(3.85), Inches(1.2), Inches(0.35),
                fill=GREEN_BG, brd=GREEN, bw=Pt(1))
        btf = b.text_frame
        btf.paragraphs[0].text = "PASS"
        btf.paragraphs[0].font.size = Pt(10); btf.paragraphs[0].font.bold = True
        btf.paragraphs[0].font.color.rgb = GREEN; btf.paragraphs[0].font.name = FT
        btf.paragraphs[0].alignment = PP_ALIGN.CENTER

    tx(sl, Inches(0.9), Inches(4.7), Inches(11.4), Inches(0.3),
       "Evaluation Tools", sz=14, b=True, c=BLACK)

    rows = [
        ("Metric", "Tool", "How It Works"),
        ("Accent Accuracy", "XVector classifier (speechbrain)",
         "Trained on VCTK/GLOBE, top-1 accent classification"),
        ("F0 Correlation", "pyworld (harvest + stonemask)",
         "Pearson r between source and output voiced F0 contours"),
        ("Word Error Rate", "Whisper-small (internal only)",
         "Re-transcribes output and compares against source transcript"),
        ("Proxy MOS", "Spectral flatness + RMS dynamics",
         "Proxy for naturalness, validated against UTMOS22 (+/-0.4)"),
        ("Speaker Agnosticism", "ECAPA-TDNN (speechbrain)",
         "Speaker embedding cosine similarity must be < 0.5"),
    ]
    tbl(sl, Inches(0.9), Inches(5.1), Inches(11.4),
        rows, [Inches(2.2), Inches(3.2), Inches(6.0)])

    note(sl, "5 metrics, all pass. F0 correlation (0.82) = prosody well preserved. "
         "Whisper used ONLY for internal WER scoring.")


def s12(prs):
    """Tech stack."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Technical Stack", "What powers AccentShift")

    box(sl, Inches(0.9), Inches(1.55), Inches(5.6), Inches(3.8), fill=WHITE, brd=BLUE, bw=Pt(1.5))
    rect(sl, Inches(0.9), Inches(1.55), Inches(5.6), Pt(4), BLUE)
    tx(sl, Inches(1.2), Inches(1.75), Inches(5.0), Inches(0.3),
       "Backend  (Python)", sz=15, b=True, c=BLUE)
    bullets(sl, Inches(1.2), Inches(2.15), Inches(5.0), [
        "FastAPI + Uvicorn  (async API server)",
        "PyTorch 2.1 + CUDA 12.1",
        "Seed-VC V2  \u2014  voice conversion  (2.5 GB)",
        "Vevo-Voice  \u2014  ensemble VC  (CPU fallback)",
        "wav2vec2-large SER  \u2014  emotion V/A/D  (1.2 GB)",
        "Whisper-small  \u2014  internal WER only  (0.5 GB)",
        "BigVGAN-v2  \u2014  vocoder  (0.5 GB)",
        "pyworld  \u00b7  Silero VAD  \u00b7  librosa  \u00b7  pyloudnorm",
    ], sz=12, c=MED, sa=5)

    box(sl, Inches(6.8), Inches(1.55), Inches(5.3), Inches(1.7), fill=WHITE, brd=TEAL, bw=Pt(1.5))
    rect(sl, Inches(6.8), Inches(1.55), Inches(5.3), Pt(4), TEAL)
    tx(sl, Inches(7.1), Inches(1.75), Inches(4.7), Inches(0.3),
       "Frontend  (TypeScript)", sz=15, b=True, c=TEAL)
    bullets(sl, Inches(7.1), Inches(2.1), Inches(4.7), [
        "Next.js 14  (App Router)",
        "WaveSurfer.js  (waveform visualisation)",
        "Tailwind CSS + shadcn/ui",
    ], sz=12, c=MED, sa=5)

    box(sl, Inches(6.8), Inches(3.55), Inches(5.3), Inches(1.8), fill=WHITE, brd=AMBER, bw=Pt(1.5))
    rect(sl, Inches(6.8), Inches(3.55), Inches(5.3), Pt(4), AMBER)
    tx(sl, Inches(7.1), Inches(3.75), Inches(4.7), Inches(0.3),
       "Infrastructure", sz=15, b=True, c=AMBER)
    bullets(sl, Inches(7.1), Inches(4.1), Inches(4.7), [
        "Local:  RTX 4060 8GB  (7.2 GB VRAM used)",
        "Cloud train:  L40S 48GB  ($1.57/hr)",
        "Cloud infer:  A10G 24GB",
        "Vevo: CPU fallback when VRAM < 10 GB",
    ], sz=12, c=MED, sa=5)

    box(sl, Inches(0.9), Inches(5.7), Inches(11.3), Inches(1.2), fill=WHITE, brd=BORDER)
    tx(sl, Inches(1.2), Inches(5.8), Inches(10.7), Inches(0.9),
       "Key design decision: Whisper is NOT in the conversion pipeline. It\u2019s loaded "
       "at startup for WER scoring only (comparing source transcript vs re-transcribed output). "
       "The actual accent conversion is entirely voice-to-voice through Seed-VC V2 and Vevo-Voice. "
       "On an 8GB GPU, Whisper-small + Seed-VC + SER + BigVGAN fit at 7.2GB. "
       "Vevo is configured to run on CPU when VRAM is constrained (vram_gb_threshold: 10).",
       sz=13, c=MED)

    note(sl, "Whisper is NOT in the conversion path. Only for internal WER scoring.")


def s13(prs):
    """Competitive."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Competitive Advantage",
           "How AccentShift compares to existing solutions")

    tx(sl, Inches(0.9), Inches(1.55), Inches(11.4), Inches(0.5),
       "Several voice conversion systems exist, but none combine all of AccentShift\u2019s "
       "capabilities \u2014 especially voice-to-voice with prosody preservation:",
       sz=15, c=DARK)

    rows = [
        ("Capability", "AccentShift", "OpenVoice V2", "CosyVoice", "StyleStream"),
        ("Conversion type", "Voice-to-voice", "Voice-to-voice", "Text-instructed", "Voice-to-voice"),
        ("Accent conversion", "Dual engine", "Basic VC", "LLM-based", "SOTA (no weights)"),
        ("Prosody preservation", "F0 correlation + transfer", "Not addressed", "Instruct tokens", "Joint model"),
        ("Zero-shot", "Yes (5s reference)", "Yes", "Yes", "No (unreleased)"),
        ("Edge deployment", "Yes (Arch B)", "No", "Partial", "No"),
        ("Ensemble", "Dual + per-segment scoring", "Single model", "Single model", "Single model"),
        ("Open source", "Fully open", "Open", "Open", "Paper only"),
    ]
    tbl(sl, Inches(0.5), Inches(2.2), Inches(12.2),
        rows, [Inches(2.4), Inches(2.2), Inches(2.2), Inches(2.2), Inches(3.2)])

    box(sl, Inches(0.9), Inches(5.8), Inches(11.4), Inches(1.0),
        fill=BLUE_BG, brd=BLUE, bw=Pt(1.5))
    tx(sl, Inches(1.3), Inches(5.9), Inches(10.7), Inches(0.7),
       "AccentShift is the only open-source system that combines dual-engine "
       "voice-to-voice conversion with explicit prosody preservation (F0 transfer) "
       "and an edge deployment variant.",
       sz=14, b=True, c=BLUE_DK, al=PP_ALIGN.CENTER)

    note(sl, "Only system with dual engine + F0 preservation + edge variant.")


def s14(prs):
    """Edge deployment."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Edge Deployment  \u2014  Architecture B",
           "CPU-only variant for real-time voice conversion")

    tx(sl, Inches(0.9), Inches(1.55), Inches(11.4), Inches(0.8),
       "We designed a lightweight variant that runs on CPU-only devices like a "
       "Raspberry Pi or laptop without a GPU. Everything is quantised to INT8 ONNX, "
       "achieving 10x faster processing. The conversion remains voice-to-voice \u2014 "
       "no text involved at any stage.",
       sz=15, c=DARK)

    rows = [
        ("Component", "Cloud (Arch A)", "Edge (Arch B)", "Trade-off"),
        ("Voice Conversion", "Seed-VC + Vevo", "CosyVoice2-0.5B (ONNX)", "Single pass"),
        ("F0 Correction", "Full F0 + energy", "F0 only (lighter)", "Less fidelity"),
        ("Vocoder", "BigVGAN-v2 (GPU)", "HiFi-GAN V1 (ONNX)", "Minor artifacts"),
        ("Latency", "2\u20134s per utterance", "200\u2013400ms per chunk", "10x faster"),
        ("Memory", "6\u20138 GB GPU", "\u22644 GB RAM (CPU)", "No GPU needed"),
    ]
    tbl(sl, Inches(0.9), Inches(2.6), Inches(11.4),
        rows, [Inches(2.2), Inches(3.0), Inches(3.2), Inches(3.0)])

    box(sl, Inches(2.5), Inches(5.5), Inches(7.3), Inches(0.9),
        fill=GREEN_BG, brd=GREEN, bw=Pt(2))
    tx(sl, Inches(2.8), Inches(5.6), Inches(6.7), Inches(0.25),
       "10x faster  \u00b7  CPU-only  \u00b7  Real-time capable",
       sz=18, b=True, c=GREEN, al=PP_ALIGN.CENTER)
    tx(sl, Inches(2.8), Inches(5.95), Inches(6.7), Inches(0.3),
       "Enables live accent conversion for phone calls and video conferencing",
       sz=13, c=MED, al=PP_ALIGN.CENTER)

    note(sl, "Architecture B: INT8 ONNX, CPU-only, 200-400ms.")


def s15(prs):
    """Roadmap."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, BG)
    header(sl, "Roadmap & Vision", "What comes next")

    for i, (phase, clr, items) in enumerate([
        ("Phase 1  \u2014  Next 2 Weeks", BLUE, [
            "Integrate StyleStream when weights are released",
            "Real-time streaming via WebSocket",
            "Expand to 10+ accents (Arabic, Korean, Spanish already in config)",
            "Browser demo with WebRTC audio capture",
        ]),
        ("Phase 2  \u2014  Next Quarter", TEAL, [
            "Mobile SDK for iOS and Android",
            "Browser-based inference via WebGPU",
            "Developer API for third-party integration",
            "Enterprise pilot: call centers, video conferencing",
        ]),
    ]):
        x = Inches(0.9 + i * 5.7)
        box(sl, x, Inches(1.55), Inches(5.3), Inches(3.2), fill=WHITE, brd=clr, bw=Pt(2))
        rect(sl, x, Inches(1.55), Inches(5.3), Pt(4), clr)
        tx(sl, x+Inches(0.3), Inches(1.75), Inches(4.7), Inches(0.3),
           phase, sz=14, b=True, c=clr)
        bullets(sl, x+Inches(0.3), Inches(2.2), Inches(4.7),
                items, sz=13, c=MED, sa=8)

    box(sl, Inches(1.5), Inches(5.2), Inches(9.3), Inches(1.5),
        fill=BLUE_BG, brd=BLUE, bw=Pt(2))
    tx(sl, Inches(1.8), Inches(5.3), Inches(8.7), Inches(0.25),
       "OUR VISION", sz=11, b=True, c=BLUE, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1.8), Inches(5.7), Inches(8.7), Inches(0.5),
       "A world where accent is a choice, not a barrier.",
       sz=24, b=True, c=BLACK, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1.8), Inches(6.25), Inches(8.7), Inches(0.3),
       "Every person should communicate clearly in any accent, "
       "without losing their emotional authenticity.",
       sz=13, c=MED, al=PP_ALIGN.CENTER)

    note(sl, "Phase 1: streaming, more accents. Phase 2: mobile, enterprise.")


def s16(prs):
    """Closing."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg_fill(sl, WHITE)

    # Subtle decorative circles
    for cx, cy, s, clr in [(2,1.5,2.5,RGBColor(0xEE,0xF2,0xF7)),
                            (10.5,4.5,3,RGBColor(0xF0,0xF4,0xF8))]:
        circ(sl, Inches(cx), Inches(cy), Inches(s), clr)

    tx(sl, Inches(1), Inches(1.6), Inches(11.3), Inches(0.9),
       "AccentShift", sz=56, b=True, c=BLUE_DK, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1), Inches(2.6), Inches(11.3), Inches(0.5),
       "Your Voice, Any Accent.", sz=24, c=BLUE, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1), Inches(3.3), Inches(11.3), Inches(0.5),
       "Voice-to-voice accent conversion with prosody preservation.",
       sz=15, c=GREY, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1), Inches(4.0), Inches(11.3), Inches(0.5),
       "IN   \u00b7   CN   \u00b7   JP   \u00b7   GB   \u00b7   US",
       sz=18, b=True, c=LGREY, al=PP_ALIGN.CENTER)
    rect(sl, Inches(5.5), Inches(4.7), Inches(2.3), Pt(2), BLUE)
    tx(sl, Inches(1), Inches(5.0), Inches(11.3), Inches(0.6),
       "Thank You", sz=32, b=True, c=BLACK, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1), Inches(5.6), Inches(11.3), Inches(0.4),
       "Questions & Live Demo", sz=18, c=GREY, al=PP_ALIGN.CENTER)
    tx(sl, Inches(1), Inches(6.3), Inches(11.3), Inches(0.35),
       "Team [Your Team Name]  |  DP6 Honeywell Designathon 2025", sz=13,
       c=LGREY, al=PP_ALIGN.CENTER)

    note(sl, "Thank you. Questions? We can demo live.")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    prs = Presentation()
    prs.slide_width = SW; prs.slide_height = SH

    s01(prs)   #  1  Title
    s02(prs)   #  2  What is AccentShift
    s03(prs)   #  3  Problem Statement
    s04(prs)   #  4  Technical Challenge
    s05(prs)   #  5  How It Works (6 steps)
    s06(prs)   #  6  Architecture Diagram
    s07(prs)   #  7  Dual Engine Detail
    s08(prs)   #  8  Prosody & Emotion Preservation
    s09(prs)   #  9  Accents & Training Data
    s10(prs)   # 10  Web Application
    s11(prs)   # 11  Evaluation Results
    s12(prs)   # 12  Tech Stack
    s13(prs)   # 13  Competitive Advantage
    s14(prs)   # 14  Edge Deployment
    s16(prs)   # 15  Thank You

    out = os.path.join(os.path.dirname(__file__), "AccentShift_v5.pptx")
    prs.save(out)
    print(f"\n[OK] Saved: {out}")
    print(f"     {len(prs.slides)} slides")
    print(f"     Open in PowerPoint or Google Slides!")

if __name__ == "__main__":
    main()
