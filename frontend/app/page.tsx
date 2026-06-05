"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";

// ── Types ──────────────────────────────────────────────────────────────────
interface AccentOption { key: string; label: string; flag: string; city: string; }
interface ConversionMetrics {
  emotion_similarity: number; wer: number; mos_estimate: number;
  valence_source: number; valence_output: number;
  arousal_source: number; arousal_output: number;
  dominance_source: number; dominance_output: number;
  n_segments: number; processing_time_ms: number; chosen_backends: string[];
}
interface ConversionResponse { audio_b64: string; metrics: ConversionMetrics; }

// ── Accent data with flags ─────────────────────────────────────────────────
const ACCENTS: AccentOption[] = [
  { key: "indian_english",   label: "Indian English",           flag: "🇮🇳", city: "Mumbai" },
  { key: "chinese_english",  label: "Chinese English",          flag: "🇨🇳", city: "Shanghai" },
  { key: "japanese_english", label: "Japanese English",         flag: "🇯🇵", city: "Tokyo" },
  { key: "british_english",  label: "British English",          flag: "🇬🇧", city: "London" },
  { key: "american_english", label: "American English",         flag: "🇺🇸", city: "New York" },
];

const ACCENT_COLORS: Record<string, string> = {
  indian_english:   "#ff9500",
  chinese_english:  "#ff3b6b",
  japanese_english: "#bf5af2",
  british_english:  "#00d4ff",
  american_english: "#30e67a",
};

// ── Helpers ────────────────────────────────────────────────────────────────
function b64ToBlob(b64: string): Blob {
  const binary = atob(b64);
  const arr = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i);
  return new Blob([arr], { type: "audio/wav" });
}

function grade(value: number, good: number, bad: number, higher = true): "good" | "warn" | "bad" {
  const better = higher ? value >= good : value <= good;
  const worse  = higher ? value <= bad  : value >= bad;
  if (better) return "good";
  if (worse)  return "bad";
  return "warn";
}

function fmtMs(ms: number) { return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`; }
function fmtTime(s: number) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }
function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }

// ── Sub-components ─────────────────────────────────────────────────────────

function SoundWave({ active, color = "var(--accent-sky)" }: { active: boolean; color?: string }) {
  return (
    <div className={`wave-container ${active ? "wave-active" : "wave-idle"}`}>
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={i} className="wave-bar" style={{ background: `linear-gradient(180deg, ${color}, transparent)` }} />
      ))}
    </div>
  );
}

function WavePlayer({ blob, label, color, id }: { blob: Blob | null; label: string; color: string; id: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!containerRef.current || !blob) return;
    wsRef.current?.destroy();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(56,189,248,0.15)",
      progressColor: color,
      cursorColor: color,
      barWidth: 2, barGap: 1, barRadius: 2,
      height: 56,
      url: URL.createObjectURL(blob),
    });
    ws.on("ready", () => setDuration(ws.getDuration()));
    ws.on("timeupdate", (t) => setCurrentTime(t));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    wsRef.current = ws;
    return () => { ws.destroy(); wsRef.current = null; };
  }, [blob, color]);

  if (!blob) return (
    <div style={{ height: 56, display: "flex", alignItems: "center", justifyContent: "center",
      background: "rgba(255,255,255,0.02)", borderRadius: 8, border: "1px dashed rgba(56,189,248,0.1)" }}>
      <span style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>No audio</span>
    </div>
  );

  return (
    <div>
      <div ref={containerRef} id={id} style={{ borderRadius: 8, overflow: "hidden" }} />
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem" }}>
        <button onClick={() => wsRef.current?.playPause()}
          style={{ width: 34, height: 34, borderRadius: "50%", background: color, border: "none",
            cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0, boxShadow: `0 0 16px ${color}55`, transition: "all 0.2s" }}
          aria-label={playing ? "Pause" : "Play"}>
          {playing
            ? <svg width="12" height="12" viewBox="0 0 12 12" fill="white"><rect x="1" y="0" width="4" height="12" rx="1"/><rect x="7" y="0" width="4" height="12" rx="1"/></svg>
            : <svg width="12" height="12" viewBox="0 0 12 12" fill="white"><polygon points="1,0 12,6 1,12"/></svg>
          }
        </button>
        <div style={{ flex: 1, height: 2, background: "rgba(255,255,255,0.06)", borderRadius: 99 }}>
          <div style={{ height: "100%", width: `${duration ? (currentTime / duration) * 100 : 0}%`,
            background: color, borderRadius: 99, transition: "width 0.1s linear" }} />
        </div>
        <span style={{ fontSize: "0.7rem", color: "var(--text-dim)", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </span>
      </div>
    </div>
  );
}

function EmotionBar({ label, src, out, color }: { label: string; src: number; out: number; color: string }) {
  const pSrc = ((src + 1) / 2) * 100;
  const pOut = ((out + 1) / 2) * 100;
  return (
    <div style={{ marginBottom: "0.875rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.375rem" }}>
        <span style={{ fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.1em",
          textTransform: "uppercase", color: "var(--text-dim)" }}>{label}</span>
        <span style={{ fontSize: "0.7rem", color, fontWeight: 600 }}>
          {src.toFixed(2)} → {out.toFixed(2)}
        </span>
      </div>
      <div className="emo-track">
        <div className="emo-fill" style={{ width: `${pSrc}%`, background: "rgba(255,255,255,0.08)" }} />
        <div className="emo-dot" style={{ left: `${pOut}%`, background: color, color }} />
      </div>
    </div>
  );
}

// ── Flight Path Background with Airplanes ─────────────────────────────────
function FlightPathsBG() {
  // Airplane silhouette (top-down view, pointing right →)
  const PlaneShape = ({ color, scale = 1 }: { color: string; scale?: number }) => (
    <g transform={`scale(${scale})`}>
      {/* Fuselage */}
      <ellipse cx="0" cy="0" rx="10" ry="2.2" fill={color} />
      {/* Wings */}
      <polygon points="0,-1.5 -3,-1.5 -9,6 -3,6" fill={color} opacity="0.85" />
      <polygon points="0,1.5 -3,1.5 -9,-6 -3,-6" fill={color} opacity="0.85" />
      {/* Tail fins */}
      <polygon points="-7,-1 -10,-1 -12,-4 -9,-2" fill={color} opacity="0.7" />
      <polygon points="-7,1 -10,1 -12,4 -9,2" fill={color} opacity="0.7" />
      {/* Cockpit glint */}
      <ellipse cx="7" cy="0" rx="2.5" ry="1.2" fill="white" opacity="0.45" />
      {/* Engine trail glow */}
      <ellipse cx="-13" cy="0" rx="4" ry="1" fill={color} opacity="0.18" />
    </g>
  );

  // Trail element — a fading streak behind the plane
  const PlaneTrail = ({ color }: { color: string }) => (
    <>
      <line x1="-14" y1="0" x2="-30" y2="0" stroke={color} strokeWidth="1.5" strokeOpacity="0.35"
        strokeLinecap="round" />
      <line x1="-14" y1="0" x2="-50" y2="0" stroke={color} strokeWidth="0.6" strokeOpacity="0.15"
        strokeLinecap="round" />
    </>
  );

  const planes = [
    {
      id: "p1",
      path: "M -50,700 Q 200,200 720,420 Q 1100,620 1480,180",
      color: "#00e5ff",
      dur: "14s",
      delay: "0s",
      scale: 1.1,
    },
    {
      id: "p2",
      path: "M 1490,150 Q 1100,550 700,280 Q 300,50 -50,480",
      color: "#ff0080",
      dur: "19s",
      delay: "-7s",
      scale: 0.9,
    },
    {
      id: "p3",
      path: "M -50,400 Q 350,80 760,320 Q 1150,550 1490,250",
      color: "#bf5af2",
      dur: "16s",
      delay: "-11s",
      scale: 1.0,
    },
    {
      id: "p4",
      path: "M 1490,600 Q 1000,200 500,450 Q 200,600 -50,300",
      color: "#ffd60a",
      dur: "22s",
      delay: "-5s",
      scale: 0.75,
    },
  ];

  return (
    <svg className="flight-paths" viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice">
      <defs>
        <filter id="glow-cyan">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="glow-pink">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <linearGradient id="trailCyan" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#00e5ff" stopOpacity="0" />
          <stop offset="60%" stopColor="#00e5ff" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#00e5ff" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="trailPink" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#ff0080" stopOpacity="0" />
          <stop offset="60%" stopColor="#ff0080" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#ff0080" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="trailPurple" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#bf5af2" stopOpacity="0" />
          <stop offset="60%" stopColor="#bf5af2" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#bf5af2" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Background curved flight route lines */}
      <path d="M -50,700 Q 200,200 720,420 Q 1100,620 1480,180"
        fill="none" stroke="#00e5ff" strokeWidth="0.6" strokeDasharray="6 10" opacity="0.12"
        style={{ animation: "dash-march 5s linear infinite" }} />
      <path d="M 1490,150 Q 1100,550 700,280 Q 300,50 -50,480"
        fill="none" stroke="#ff0080" strokeWidth="0.6" strokeDasharray="6 10" opacity="0.10"
        style={{ animation: "dash-march 7s linear infinite reverse" }} />
      <path d="M -50,400 Q 350,80 760,320 Q 1150,550 1490,250"
        fill="none" stroke="#bf5af2" strokeWidth="0.5" strokeDasharray="5 12" opacity="0.10"
        style={{ animation: "dash-march 9s linear infinite" }} />
      <path d="M 1490,600 Q 1000,200 500,450 Q 200,600 -50,300"
        fill="none" stroke="#ffd60a" strokeWidth="0.5" strokeDasharray="5 12" opacity="0.08"
        style={{ animation: "dash-march 11s linear infinite reverse" }} />

      {/* ✈ Animated airplanes along each path */}
      {planes.map((p) => (
        <g key={p.id}>
          <animateMotion
            dur={p.dur}
            repeatCount="indefinite"
            begin={p.delay}
            rotate="auto"
          >
            <mpath href={`#route-${p.id}`} />
          </animateMotion>
          {/* Glow layer */}
          <g filter="url(#glow-cyan)" opacity="0.6">
            <PlaneShape color={p.color} scale={p.scale} />
          </g>
          {/* Solid layer */}
          <PlaneShape color={p.color} scale={p.scale} />
        </g>
      ))}

      {/* Hidden path defs for animateMotion mpath references */}
      {planes.map((p) => (
        <path key={`route-${p.id}`} id={`route-${p.id}`} d={p.path} fill="none" stroke="none" />
      ))}
    </svg>
  );
}

// ── Hero Globe Rings ───────────────────────────────────────────────────────
function GlobeRings() {
  return (
    <div style={{ position: "relative", width: 220, height: 220, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div className="globe-ring globe-ring-3" style={{ position: "absolute" }} />
      <div className="globe-ring globe-ring-2" style={{ position: "absolute" }} />
      <div className="globe-ring globe-ring-1" style={{ position: "absolute" }} />
      {/* Core mic */}
      <div style={{ width: 76, height: 76, borderRadius: "50%",
            background: "linear-gradient(135deg, #00b4d8, #7c3aed 50%, #c026d3)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 50px rgba(0,229,255,0.6), 0 0 100px rgba(192,38,211,0.4), 0 0 150px rgba(124,58,237,0.2)",
            animation: "floatMic 4s ease-in-out infinite",
            position: "relative", zIndex: 1,
          }}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="22"/>
          <line x1="8"  y1="22" x2="16" y2="22"/>
        </svg>
      </div>
      {/* Pulse rings */}
      {[1, 2, 3].map(i => (
        <div key={i} style={{
          position: "absolute", width: 76, height: 76, borderRadius: "50%",
          border: `1px solid ${i === 1 ? 'rgba(0,229,255,0.6)' : i === 2 ? 'rgba(192,38,211,0.45)' : 'rgba(255,0,128,0.3)'}`,
          animation: `ring-expand 2.5s ease-out infinite`,
          animationDelay: `${(i - 1) * 0.8}s`,
        }} />
      ))}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Home() {
  const [selectedAccent, setSelectedAccent] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceBlob, setSourceBlob] = useState<Blob | null>(null);
  const [outputBlob, setOutputBlob] = useState<Blob | null>(null);
  const [metrics, setMetrics] = useState<ConversionMetrics | null>(null);
  const [convStatus, setConvStatus] = useState<"idle" | "converting" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    setSourceFile(file);
    setSourceBlob(file);
    setOutputBlob(null);
    setMetrics(null);
    setConvStatus("idle");
    setErrorMsg("");
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleConvert = async () => {
    if (!sourceFile || !selectedAccent) return;
    setConvStatus("converting"); setErrorMsg("");
    setOutputBlob(null); setMetrics(null);
    const fd = new FormData();
    fd.append("audio", sourceFile);
    fd.append("target_accent", selectedAccent);
    try {
      const res = await fetch("/api/convert", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error ?? "Conversion failed");
      }
      const data: ConversionResponse = await res.json();
      setOutputBlob(b64ToBlob(data.audio_b64));
      setMetrics(data.metrics);
      setConvStatus("done");
    } catch (e: unknown) {
      setConvStatus("error");
      setErrorMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const downloadOutput = () => {
    if (!outputBlob) return;
    const url = URL.createObjectURL(outputBlob);
    const a = document.createElement("a");
    a.href = url; a.download = `accentshift_${selectedAccent}_${Date.now()}.wav`; a.click();
    URL.revokeObjectURL(url);
  };

  const accent = ACCENTS.find(a => a.key === selectedAccent);
  const accentColor = accent ? ACCENT_COLORS[accent.key] : "var(--accent-sky)";
  const canConvert = !!sourceFile && !!selectedAccent && convStatus !== "converting";

  return (
    <>
      {/* Fixed background layers */}
      <div className="space-bg">
        <div className="stars" />
        <div className="nebula nebula-1" />
        <div className="nebula nebula-2" />
        <div className="nebula nebula-3" />
        <div className="nebula nebula-4" />
      </div>
      <div className="grid-overlay" />
      <FlightPathsBG />

      <div style={{ position: "relative", zIndex: 1, minHeight: "100vh" }}>

        {/* ── HEADER ──────────────────────────────────────────────── */}
        <header style={{ padding: "1.5rem 2rem", display: "flex", alignItems: "center",
          justifyContent: "space-between", borderBottom: "1px solid rgba(56,189,248,0.07)",
          backdropFilter: "blur(12px)", background: "rgba(3,7,18,0.6)", position: "sticky", top: 0, zIndex: 50 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.875rem" }}>
            <div style={{ width: 40, height: 40, borderRadius: 10,
              background: "linear-gradient(135deg, #00b4d8, #7c3aed 50%, #c026d3)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 25px rgba(0,229,255,0.6), 0 0 50px rgba(192,38,211,0.3)", animation: "glowPulse 3s ease-in-out infinite" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="22"/>
                <line x1="8"  y1="22" x2="16" y2="22"/>
              </svg>
            </div>
            <div>
              <h1 className="font-display" style={{ fontSize: "1.125rem", fontWeight: 800, letterSpacing: "-0.02em" }}>
                Accent<span className="gradient-text">Shift</span>
              </h1>
              <p style={{ fontSize: "0.65rem", color: "var(--text-dim)", letterSpacing: "0.08em", marginTop: 1 }}>
                EMOTION-PRESERVING ACCENT CONVERSION
              </p>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
            {/* HUD status */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem",
              background: "rgba(0,229,255,0.08)", border: "1px solid rgba(0,229,255,0.3)",
              padding: "0.35rem 0.875rem", borderRadius: 99 }}>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#30e67a",
                boxShadow: "0 0 10px #30e67a, 0 0 20px #30e67a60", animation: "glowPulse 2s ease-in-out infinite" }} />
              <span style={{ fontSize: "0.7rem", color: "#00e5ff", fontWeight: 700, letterSpacing: "0.1em" }}>
                SYSTEM READY
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.375rem" }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-dim)" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
              </svg>
              <span style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>DP6 · Honeywell</span>
            </div>
          </div>
        </header>

        {/* ── HERO ────────────────────────────────────────────────── */}
        <section style={{ display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", padding: "4rem 2rem 3rem", textAlign: "center", gap: "1.5rem" }}>
          <GlobeRings />
          <div style={{ maxWidth: 680 }}>
            <p className="section-label" style={{ marginBottom: "0.75rem", color: "#00e5ff", letterSpacing: "0.2em" }}>✈ POWERED BY SEED-VC V2 + VEVO</p>
            <h2 className="font-display" style={{ fontSize: "clamp(2rem, 5vw, 3.25rem)", fontWeight: 800,
              lineHeight: 1.1, letterSpacing: "-0.03em", marginBottom: "1rem" }}>
              Your Voice,{" "}
              <span style={{ background: "linear-gradient(135deg, #00e5ff 0%, #bf5af2 45%, #ff0080 80%, #ffd60a 100%)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>Any Accent</span>
              <br />Across the World
            </h2>
            <p style={{ fontSize: "1rem", color: "var(--text-dim)", lineHeight: 1.7, maxWidth: 520, margin: "0 auto" }}>
              AI-powered accent conversion that preserves your emotion, prosody, and voice
              intensity — like your words flying across the globe, landing perfectly in a new accent.
            </p>
          </div>
          {/* Destination tags */}
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "center", marginTop: "0.5rem" }}>
            {ACCENTS.map(a => (
              <span key={a.key} style={{ display: "flex", alignItems: "center", gap: "0.375rem",
                padding: "0.3rem 0.75rem", borderRadius: 99,
                background: selectedAccent === a.key ? `${ACCENT_COLORS[a.key]}22` : "rgba(255,255,255,0.03)",
                border: `1px solid ${selectedAccent === a.key ? ACCENT_COLORS[a.key] + "66" : "rgba(255,255,255,0.06)"}`,
                fontSize: "0.75rem", color: selectedAccent === a.key ? ACCENT_COLORS[a.key] : "var(--text-dim)",
                transition: "all 0.2s" }}>
                <span>{a.flag}</span> {a.city}
              </span>
            ))}
          </div>
        </section>

        {/* ── MAIN WORKSPACE ──────────────────────────────────────── */}
        <main style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem 3rem" }}>
          <div className="two-col" style={{ display: "flex", gap: "1.5rem" }}>

            {/* ── LEFT PANEL ── Upload + Accent + Convert ─────────── */}
            <div className="glass-card" style={{ flex: 1, padding: "2rem", minWidth: 0 }}>
              {/* HUD corner brackets */}
              <div className="hud-corner hud-tl" />
              <div className="hud-corner hud-tr" />
              <div className="hud-corner hud-bl" />
              <div className="hud-corner hud-br" />

              <p className="section-label" style={{ marginBottom: "1.25rem" }}>SOURCE AUDIO</p>

              {/* Drop zone */}
              <div className={`drop-zone ${dragging ? "drag-over" : ""}`}
                style={{ padding: "1.75rem 1rem", textAlign: "center", marginBottom: "1.5rem" }}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                role="button" tabIndex={0} aria-label="Upload audio"
                onKeyDown={e => e.key === "Enter" && fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" accept=".wav,.mp3,.flac,.m4a,audio/*"
                  style={{ display: "none" }} onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
                  id="audio-file-input" />
                {sourceFile ? (
                  <div className="fade-in">
                    <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>🎵</div>
                    <p style={{ fontWeight: 700, fontFamily: "var(--font-display)", marginBottom: "0.25rem",
                      color: "var(--text-bright)", fontSize: "0.9rem" }}>{sourceFile.name}</p>
                    <p style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>
                      {(sourceFile.size / 1024 / 1024).toFixed(2)} MB · click to change
                    </p>
                  </div>
                ) : (
                  <div>
                    <div style={{ width: 58, height: 58, borderRadius: "50%",
                      background: "rgba(0,229,255,0.1)", border: "1.5px dashed rgba(0,229,255,0.4)",
                      display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1rem",
                      boxShadow: "0 0 20px rgba(0,229,255,0.15)" }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00e5ff" strokeWidth="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                      </svg>
                    </div>
                    <p style={{ fontWeight: 600, color: "var(--text)", marginBottom: "0.375rem" }}>
                      Drop your audio file here
                    </p>
                    <p style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>
                      WAV · MP3 · FLAC · M4A — max 50 MB
                    </p>
                  </div>
                )}
              </div>

              {/* Source waveform */}
              <div style={{ marginBottom: "1.75rem" }}>
                <p style={{ fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.1em",
                  textTransform: "uppercase", color: "var(--text-dim)", marginBottom: "0.625rem" }}>
                  SOURCE WAVEFORM
                </p>
                <WavePlayer blob={sourceBlob} label="" color="#38bdf8" id="source-waveform" />
              </div>

              {/* Accent selector */}
              <div style={{ marginBottom: "1.75rem" }}>
                <p style={{ fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.1em",
                  textTransform: "uppercase", color: "#00e5ff", marginBottom: "0.75rem", opacity: 0.8 }}>
                  ✈ TARGET DESTINATION
                </p>
                <div className="accent-chips" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {ACCENTS.map(a => (
                    <button key={a.key} id={`accent-${a.key}`}
                      className={`accent-chip ${selectedAccent === a.key ? "selected" : ""}`}
                      style={selectedAccent === a.key ? { borderColor: ACCENT_COLORS[a.key],
                        color: ACCENT_COLORS[a.key], background: `${ACCENT_COLORS[a.key]}18`,
                        boxShadow: `0 0 16px ${ACCENT_COLORS[a.key]}30` } : {}}
                      onClick={() => setSelectedAccent(a.key)}>
                      <span style={{ fontSize: "1rem" }}>{a.flag}</span>
                      <span>{a.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Convert button */}
              <button id="convert-btn" className="btn-primary" disabled={!canConvert}
                style={{ width: "100%", fontSize: "1rem", padding: "0.875rem",
                  ...(canConvert ? {
                    background: `linear-gradient(135deg, ${accentColor}, #6366f1 50%, #8b5cf6)`,
                    boxShadow: `0 0 30px ${accentColor}44, 0 8px 25px rgba(0,0,0,0.3)`,
                  } : {}) }}
                onClick={handleConvert}>
                {convStatus === "converting" ? (
                  <><span className="spinner" /> Converting — processing audio…</>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                    Launch Conversion
                    {accent && <span style={{ fontSize: "0.9rem" }}>{accent.flag}</span>}
                  </>
                )}
              </button>

              {convStatus === "converting" && (
                <div className="shimmer" style={{ marginTop: "0.75rem", borderRadius: 8, height: 3 }} />
              )}
              {convStatus === "converting" && (
                <p style={{ textAlign: "center", fontSize: "0.75rem", color: "var(--text-dim)", marginTop: "0.625rem" }}>
                  ✈️ Estimated flight time: 15–90 seconds
                </p>
              )}

              {convStatus === "error" && (
                <div className="fade-in" style={{ marginTop: "0.875rem", padding: "0.75rem 1rem",
                  background: "rgba(251,113,133,0.08)", border: "1px solid rgba(251,113,133,0.25)",
                  borderRadius: 8, color: "#fb7185", fontSize: "0.82rem" }}>
                  ⚠ {errorMsg}
                </div>
              )}
            </div>

            {/* ── RIGHT PANEL ── Output + Metrics ─────────────────── */}
            <div className="glass-card" style={{ flex: 1, padding: "2rem", minWidth: 0 }}>
              <div className="hud-corner hud-tl" />
              <div className="hud-corner hud-tr" />
              <div className="hud-corner hud-bl" />
              <div className="hud-corner hud-br" />

              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.25rem" }}>
                <p className="section-label">OUTPUT AUDIO</p>
                {accent && convStatus === "done" && (
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem",
                padding: "0.3rem 0.75rem", borderRadius: 99,
                background: `${accentColor}20`, border: `1px solid ${accentColor}55`,
                boxShadow: `0 0 12px ${accentColor}35` }}>
                    <span>{accent.flag}</span>
                    <span style={{ fontSize: "0.72rem", fontWeight: 700, color: accentColor }}>{accent.label.toUpperCase()}</span>
                  </div>
                )}
              </div>

              {/* Output waveform or processing */}
              <div style={{ marginBottom: "1.5rem" }}>
                {convStatus === "converting" ? (
                  <div style={{ height: 88, borderRadius: 12, background: "rgba(56,189,248,0.03)",
                    border: "1px solid rgba(56,189,248,0.1)", display: "flex", flexDirection: "column",
                    alignItems: "center", justifyContent: "center", gap: "0.75rem" }}>
                    <SoundWave active={true} color={accentColor} />
                    <span style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>Synthesising…</span>
                  </div>
                ) : (
                  <div className={convStatus === "done" ? "fade-in" : ""}>
                    <WavePlayer blob={outputBlob} label="" color={accentColor} id="output-waveform" />
                  </div>
                )}
              </div>

              {/* Download */}
              {outputBlob && (
                <button id="download-btn" className="btn-secondary scale-in" onClick={downloadOutput}
                  style={{ width: "100%", marginBottom: "1.75rem", borderColor: `${accentColor}55`, color: accentColor }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download Converted WAV
                </button>
              )}

              {/* Emotion preservation */}
              {metrics ? (
                <div className="fade-in">
                  <p style={{ fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.1em",
                    textTransform: "uppercase", color: "var(--text-dim)", marginBottom: "0.875rem" }}>
                    EMOTION PRESERVATION ANALYSIS
                  </p>
                  <EmotionBar label="Valence"   src={metrics.valence_source}   out={metrics.valence_output}   color="#38bdf8" />
                  <EmotionBar label="Arousal"   src={metrics.arousal_source}   out={metrics.arousal_output}   color="#a78bfa" />
                  <EmotionBar label="Dominance" src={metrics.dominance_source} out={metrics.dominance_output} color="#fb7185" />
                  <p style={{ fontSize: "0.68rem", color: "var(--text-dim)", marginTop: "0.5rem" }}>
                    Grey fill = source · Coloured dot = output · Closer = better emotion preservation
                  </p>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                  gap: "1rem", padding: "2rem 0", opacity: 0.4 }}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent-sky)" strokeWidth="1" strokeLinecap="round">
                    <circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 4 10 15 15 0 0 1-4 10 15 15 0 0 1-4-10 15 15 0 0 1 4-10z"/>
                  </svg>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-dim)", textAlign: "center" }}>
                    Upload audio and select a destination<br />to begin the journey
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* ── METRICS ROW ─────────────────────────────────────────── */}
          {metrics && (
            <div className="slide-up" style={{ marginTop: "1.5rem" }}>
              {/* Divider */}
              <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.25rem" }}>
                <div style={{ flex: 1, height: 1, background: "linear-gradient(90deg, transparent, rgba(0,229,255,0.5), transparent)" }} />
                <p className="section-label" style={{ color: "#ffd60a", letterSpacing: "0.25em" }}>FLIGHT REPORT</p>
                <div style={{ flex: 1, height: 1, background: "linear-gradient(90deg, transparent, rgba(0,229,255,0.5), transparent)" }} />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "1rem" }}>
                <div className={`metric-card ${grade(metrics.emotion_similarity, 0.85, 0.6)}`} style={{ animationDelay: "0.05s" }}>
                  <span className="metric-label">Emotion Sim.</span>
                  <span className="metric-value">{pct(metrics.emotion_similarity)}</span>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-dim)" }}>
                    {metrics.emotion_similarity >= 0.85 ? "✓ Preserved" : metrics.emotion_similarity >= 0.6 ? "⚠ Partial" : "✗ Drifted"}
                  </span>
                </div>
                <div className={`metric-card ${grade(metrics.wer, 0.1, 0.3, false)}`} style={{ animationDelay: "0.10s" }}>
                  <span className="metric-label">Word Error Rate</span>
                  <span className="metric-value">{pct(metrics.wer)}</span>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-dim)" }}>
                    {metrics.wer <= 0.1 ? "✓ Accurate" : metrics.wer <= 0.3 ? "⚠ Fair" : "✗ High"}
                  </span>
                </div>
                <div className={`metric-card ${grade(metrics.mos_estimate, 3.8, 3.0)}`} style={{ animationDelay: "0.15s" }}>
                  <span className="metric-label">UTMOS Score</span>
                  <span className="metric-value">{metrics.mos_estimate.toFixed(2)}<span style={{ fontSize: "0.95rem", fontWeight: 400 }}>/5</span></span>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-dim)" }}>
                    {metrics.mos_estimate >= 3.8 ? "✓ Natural" : metrics.mos_estimate >= 3.0 ? "⚠ Acceptable" : "✗ Artefacts"}
                  </span>
                </div>
                <div className="metric-card" style={{ animationDelay: "0.20s" }}>
                  <span className="metric-label">Segments</span>
                  <span className="metric-value">{metrics.n_segments}</span>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-dim)" }}>VAD chunks</span>
                </div>
                <div className="metric-card" style={{ animationDelay: "0.25s" }}>
                  <span className="metric-label">Flight Time</span>
                  <span className="metric-value">{fmtMs(metrics.processing_time_ms)}</span>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-dim)" }}>end-to-end</span>
                </div>
                <div className="metric-card" style={{ animationDelay: "0.30s" }}>
                  <span className="metric-label">Engine</span>
                  <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--accent-sky)",
                    marginTop: "0.25rem", textTransform: "capitalize" }}>
                    {[...new Set(metrics.chosen_backends)].join(" + ")}
                  </span>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-dim)" }}>winner / segment</span>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* ── FOOTER ────────────────────────────────────────────────── */}
        <footer style={{ borderTop: "1px solid rgba(56,189,248,0.06)", padding: "1.5rem 2rem",
          display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem",
          background: "rgba(3,7,18,0.5)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span className="gradient-text" style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: "0.875rem" }}>AccentShift</span>
            <span style={{ color: "var(--text-dim)", fontSize: "0.75rem" }}>· DP6 Honeywell Designathon 2025</span>
          </div>
          <span style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
            Seed-VC V2 · Vevo (Amphion) · Whisper large-v3 · wav2vec2 SER · pyworld
          </span>
        </footer>
      </div>
    </>
  );
}
