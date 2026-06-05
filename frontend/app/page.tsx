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

// ── Accents ────────────────────────────────────────────────────────────────
const ACCENTS: AccentOption[] = [
  { key: "indian_english",   label: "Indian English",   flag: "🇮🇳", city: "Mumbai"   },
  { key: "chinese_english",  label: "Chinese English",  flag: "🇨🇳", city: "Shanghai" },
  { key: "japanese_english", label: "Japanese English", flag: "🇯🇵", city: "Tokyo"    },
  { key: "british_english",  label: "British English",  flag: "🇬🇧", city: "London"   },
  { key: "american_english", label: "American English", flag: "🇺🇸", city: "New York" },
];

// Warm, country-coded accent colors — no purple
const ACCENT_COLORS: Record<string, string> = {
  indian_english:   "#f97316",  // saffron orange
  chinese_english:  "#dc2626",  // China red
  japanese_english: "#e11d48",  // Japan crimson
  british_english:  "#0d9488",  // British teal
  american_english: "#16a34a",  // US green
};

// ── Helpers ────────────────────────────────────────────────────────────────
function b64ToBlob(b64: string): Blob {
  const binary = atob(b64);
  const arr = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i);
  return new Blob([arr], { type: "audio/wav" });
}
function grade(v: number, good: number, bad: number, higher = true): "good"|"warn"|"bad" {
  const better = higher ? v >= good : v <= good;
  const worse  = higher ? v <= bad  : v >= bad;
  return better ? "good" : worse ? "bad" : "warn";
}
function fmtMs(ms: number) { return ms >= 1000 ? `${(ms/1000).toFixed(1)}s` : `${ms}ms`; }
function fmtTime(s: number) { return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}`; }
function pct(v: number) { return `${(v*100).toFixed(1)}%`; }

// ── Sound Wave ─────────────────────────────────────────────────────────────
function SoundWave({ active, color="#ea580c" }: { active: boolean; color?: string }) {
  return (
    <div className={`wave-container ${active ? "wave-active" : "wave-idle"}`}>
      {Array.from({length:12}).map((_,i) => (
        <div key={i} className="wave-bar"
          style={{ background: `linear-gradient(180deg, ${color}, ${color}66)` }} />
      ))}
    </div>
  );
}

// ── Wave Player ────────────────────────────────────────────────────────────
function WavePlayer({ blob, color, id }: { blob: Blob|null; color: string; id: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef        = useRef<WaveSurfer|null>(null);
  const [playing, setPlaying]       = useState(false);
  const [duration, setDuration]     = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!containerRef.current || !blob) return;
    wsRef.current?.destroy();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(234,88,12,0.15)",
      progressColor: color,
      cursorColor: color,
      barWidth: 2, barGap: 1, barRadius: 2,
      height: 56,
      url: URL.createObjectURL(blob),
    });
    ws.on("ready",      () => setDuration(ws.getDuration()));
    ws.on("timeupdate", (t) => setCurrentTime(t));
    ws.on("play",       () => setPlaying(true));
    ws.on("pause",      () => setPlaying(false));
    ws.on("finish",     () => setPlaying(false));
    wsRef.current = ws;
    return () => { ws.destroy(); wsRef.current = null; };
  }, [blob, color]);

  if (!blob) return (
    <div style={{ height: 56, display:"flex", alignItems:"center", justifyContent:"center",
      background:"rgba(234,88,12,0.04)", borderRadius:8,
      border:"1.5px dashed rgba(234,88,12,0.20)" }}>
      <span style={{ fontSize:"0.75rem", color:"#a8a29e" }}>No audio</span>
    </div>
  );

  return (
    <div>
      <div ref={containerRef} id={id} style={{ borderRadius:8, overflow:"hidden" }} />
      <div style={{ display:"flex", alignItems:"center", gap:"0.75rem", marginTop:"0.5rem" }}>
        <button onClick={() => wsRef.current?.playPause()}
          style={{ width:34, height:34, borderRadius:"50%", background:color, border:"none",
            cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center",
            flexShrink:0, boxShadow:`0 4px 12px ${color}55`, transition:"all 0.2s" }}
          aria-label={playing ? "Pause" : "Play"}>
          {playing
            ? <svg width="12" height="12" viewBox="0 0 12 12" fill="white"><rect x="1" y="0" width="4" height="12" rx="1"/><rect x="7" y="0" width="4" height="12" rx="1"/></svg>
            : <svg width="12" height="12" viewBox="0 0 12 12" fill="white"><polygon points="1,0 12,6 1,12"/></svg>}
        </button>
        <div style={{ flex:1, height:2, background:"rgba(0,0,0,0.08)", borderRadius:99 }}>
          <div style={{ height:"100%", width:`${duration ? (currentTime/duration)*100 : 0}%`,
            background:color, borderRadius:99, transition:"width 0.1s linear" }} />
        </div>
        <span style={{ fontSize:"0.7rem", color:"#a8a29e", fontVariantNumeric:"tabular-nums", flexShrink:0 }}>
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </span>
      </div>
    </div>
  );
}

// ── Emotion Bar ────────────────────────────────────────────────────────────
function EmotionBar({ label, src, out, color }: { label:string; src:number; out:number; color:string }) {
  const pSrc = ((src+1)/2)*100;
  const pOut = ((out+1)/2)*100;
  return (
    <div style={{ marginBottom:"0.875rem" }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"0.375rem" }}>
        <span style={{ fontSize:"0.7rem", fontWeight:700, letterSpacing:"0.10em",
          textTransform:"uppercase", color:"#a8a29e" }}>{label}</span>
        <span style={{ fontSize:"0.7rem", color, fontWeight:700 }}>
          {src.toFixed(2)} → {out.toFixed(2)}
        </span>
      </div>
      <div className="emo-track">
        <div className="emo-fill" style={{ width:`${pSrc}%`, background:"rgba(0,0,0,0.10)" }} />
        <div className="emo-dot" style={{ left:`${pOut}%`, background:color, color }} />
      </div>
    </div>
  );
}

// ── Flight Paths + Airplane SVGs ───────────────────────────────────────────
function FlightPathsBG() {
  // Top-down airplane silhouette, pointing right →
  const Plane = ({ color, s=1 }: { color:string; s?:number }) => (
    <g transform={`scale(${s})`} opacity="0.9">
      <ellipse cx="0" cy="0" rx="10" ry="2.2" fill={color} />
      <polygon points="0,-1.5 -3,-1.5 -9,6.5 -2.5,6.5" fill={color} opacity="0.82" />
      <polygon points="0,1.5 -3,1.5 -9,-6.5 -2.5,-6.5" fill={color} opacity="0.82" />
      <polygon points="-7,-1 -10,-1 -12.5,-4.5 -9,-2" fill={color} opacity="0.68" />
      <polygon points="-7,1 -10,1 -12.5,4.5 -9,2" fill={color} opacity="0.68" />
      <ellipse cx="7" cy="0" rx="2.5" ry="1.2" fill="white" opacity="0.55" />
    </g>
  );

  const planes = [
    { id:"p1", d:"M -50,700 Q 200,200 720,420 Q 1100,620 1480,180", color:"#ea580c", dur:"14s", begin:"0s",    s:1.1 },
    { id:"p2", d:"M 1490,150 Q 1100,550 700,280 Q 300,50 -50,480",  color:"#dc2626", dur:"19s", begin:"-7s",   s:0.9 },
    { id:"p3", d:"M -50,400 Q 350,80 760,320 Q 1150,550 1490,250",  color:"#d97706", dur:"16s", begin:"-11s",  s:1.0 },
    { id:"p4", d:"M 1490,600 Q 1000,200 500,450 Q 200,600 -50,300", color:"#0d9488", dur:"22s", begin:"-5s",   s:0.78},
  ];

  return (
    <svg className="flight-paths" viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice">
      <defs>
        <filter id="pglow"><feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>

      {/* Subtle route lines */}
      <path d="M -50,700 Q 200,200 720,420 Q 1100,620 1480,180"
        fill="none" stroke="#ea580c" strokeWidth="0.9" strokeDasharray="7 10" opacity="0.14"
        style={{ animation:"dash-march 5s linear infinite" }} />
      <path d="M 1490,150 Q 1100,550 700,280 Q 300,50 -50,480"
        fill="none" stroke="#dc2626" strokeWidth="0.9" strokeDasharray="7 10" opacity="0.11"
        style={{ animation:"dash-march 7s linear infinite reverse" }} />
      <path d="M -50,400 Q 350,80 760,320 Q 1150,550 1490,250"
        fill="none" stroke="#d97706" strokeWidth="0.8" strokeDasharray="5 12" opacity="0.11"
        style={{ animation:"dash-march 9s linear infinite" }} />
      <path d="M 1490,600 Q 1000,200 500,450 Q 200,600 -50,300"
        fill="none" stroke="#0d9488" strokeWidth="0.8" strokeDasharray="5 12" opacity="0.09"
        style={{ animation:"dash-march 11s linear infinite reverse" }} />

      {/* Hidden path defs */}
      {planes.map(p => <path key={`r-${p.id}`} id={`rt-${p.id}`} d={p.d} fill="none" stroke="none"/>)}

      {/* Animated airplanes */}
      {planes.map(p => (
        <g key={p.id}>
          <animateMotion dur={p.dur} repeatCount="indefinite" begin={p.begin} rotate="auto">
            <mpath href={`#rt-${p.id}`} />
          </animateMotion>
          <g filter="url(#pglow)" opacity="0.55"><Plane color={p.color} s={p.s} /></g>
          <Plane color={p.color} s={p.s} />
        </g>
      ))}
    </svg>
  );
}

// ── Globe Rings ────────────────────────────────────────────────────────────
function GlobeRings() {
  return (
    <div style={{ position:"relative", width:240, height:240,
      display:"flex", alignItems:"center", justifyContent:"center" }}>
      <div className="globe-ring globe-ring-3" style={{ position:"absolute" }} />
      <div className="globe-ring globe-ring-2" style={{ position:"absolute" }} />
      <div className="globe-ring globe-ring-1" style={{ position:"absolute" }} />

      {/* Core mic — orange-to-red gradient, distinctive */}
      <div style={{
        width:76, height:76, borderRadius:"50%",
        background:"linear-gradient(135deg, #f97316, #ea580c 45%, #dc2626 80%)",
        display:"flex", alignItems:"center", justifyContent:"center",
        boxShadow:"0 8px 32px rgba(234,88,12,0.45), 0 4px 16px rgba(220,38,38,0.25)",
        animation:"floatMic 4s ease-in-out infinite",
        position:"relative", zIndex:1,
      }}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
          stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="22"/>
          <line x1="8"  y1="22" x2="16" y2="22"/>
        </svg>
      </div>

      {/* Warm pulse rings */}
      {[1,2,3].map(i => (
        <div key={i} style={{
          position:"absolute", width:76, height:76, borderRadius:"50%",
          border:`1px solid ${i===1?"rgba(234,88,12,0.55)":i===2?"rgba(220,38,38,0.38)":"rgba(13,148,136,0.28)"}`,
          animation:"ring-expand 2.5s ease-out infinite",
          animationDelay:`${(i-1)*0.8}s`,
        }} />
      ))}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────
export default function Home() {
  const [selectedAccent, setSelectedAccent] = useState("");
  const [sourceFile, setSourceFile]   = useState<File|null>(null);
  const [sourceBlob, setSourceBlob]   = useState<Blob|null>(null);
  const [outputBlob, setOutputBlob]   = useState<Blob|null>(null);
  const [metrics, setMetrics]         = useState<ConversionMetrics|null>(null);
  const [convStatus, setConvStatus]   = useState<"idle"|"converting"|"done"|"error">("idle");
  const [errorMsg, setErrorMsg]       = useState("");
  const [dragging, setDragging]       = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file:File) => {
    setSourceFile(file); setSourceBlob(file);
    setOutputBlob(null); setMetrics(null);
    setConvStatus("idle"); setErrorMsg("");
  }, []);

  const onDrop = useCallback((e:React.DragEvent) => {
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
      const res = await fetch("/api/convert", { method:"POST", body:fd });
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
  const accentColor = accent ? ACCENT_COLORS[accent.key] : "#ea580c";
  const canConvert  = !!sourceFile && !!selectedAccent && convStatus !== "converting";

  return (
    <>
      {/* Background */}
      <div className="space-bg">
        <div className="stars" />
        <div className="nebula nebula-1" />
        <div className="nebula nebula-2" />
        <div className="nebula nebula-3" />
        <div className="nebula nebula-4" />
      </div>
      <div className="grid-overlay" />
      <FlightPathsBG />

      <div style={{ position:"relative", zIndex:1, minHeight:"100vh" }}>

        {/* ── HEADER ─────────────────────────────────────────────── */}
        <header style={{
          padding:"1rem 2rem",
          display:"flex", alignItems:"center", justifyContent:"space-between",
          borderBottom:"1px solid rgba(234,88,12,0.10)",
          background:"rgba(255,251,245,0.85)",
          backdropFilter:"blur(16px) saturate(180%)",
          position:"sticky", top:0, zIndex:50,
          boxShadow:"0 1px 12px rgba(234,88,12,0.07)",
        }}>
          <div style={{ display:"flex", alignItems:"center", gap:"0.875rem" }}>
            {/* Logo icon */}
            <div style={{ width:42, height:42, borderRadius:12,
              background:"linear-gradient(135deg, #f97316, #ea580c 50%, #dc2626)",
              display:"flex", alignItems:"center", justifyContent:"center",
              boxShadow:"0 4px 16px rgba(234,88,12,0.40)",
              animation:"glowPulse 3s ease-in-out infinite" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="22"/>
                <line x1="8"  y1="22" x2="16" y2="22"/>
              </svg>
            </div>
            <div>
              <h1 className="font-display" style={{ fontSize:"1.125rem", fontWeight:800,
                letterSpacing:"-0.02em", color:"#1c0a00" }}>
                Accent<span className="gradient-text">Shift</span>
              </h1>
              <p style={{ fontSize:"0.62rem", color:"#a8a29e", letterSpacing:"0.10em", marginTop:1 }}>
                EMOTION-PRESERVING ACCENT CONVERSION
              </p>
            </div>
          </div>

          <div style={{ display:"flex", alignItems:"center", gap:"1.25rem" }}>
            <div style={{ display:"flex", alignItems:"center", gap:"0.5rem",
              background:"rgba(13,148,136,0.08)", border:"1.5px solid rgba(13,148,136,0.25)",
              padding:"0.35rem 0.875rem", borderRadius:99 }}>
              <div style={{ width:7, height:7, borderRadius:"50%", background:"#0d9488",
                boxShadow:"0 0 8px rgba(13,148,136,0.65)",
                animation:"glowPulse 2s ease-in-out infinite" }} />
              <span style={{ fontSize:"0.68rem", color:"#0d9488", fontWeight:700, letterSpacing:"0.10em" }}>
                SYSTEM READY
              </span>
            </div>
            <div style={{ display:"flex", alignItems:"center", gap:"0.375rem" }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a8a29e" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10"/>
                <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
              </svg>
              <span style={{ fontSize:"0.7rem", color:"#a8a29e" }}>DP6 · Honeywell</span>
            </div>
          </div>
        </header>

        {/* ── HERO ───────────────────────────────────────────────── */}
        <section style={{ display:"flex", flexDirection:"column", alignItems:"center",
          justifyContent:"center", padding:"3.5rem 2rem 2.5rem",
          textAlign:"center", gap:"1.5rem" }}>
          <GlobeRings />
          <div style={{ maxWidth:660 }}>
            <p style={{ fontSize:"0.65rem", fontWeight:800, letterSpacing:"0.20em",
              color:"#ea580c", marginBottom:"0.75rem" }}>
              ✈ POWERED BY SEED-VC V2 + VEVO
            </p>
            <h2 className="font-display" style={{ fontSize:"clamp(2rem,5vw,3.25rem)",
              fontWeight:800, lineHeight:1.1, letterSpacing:"-0.03em",
              marginBottom:"1rem", color:"#1c0a00" }}>
              Your Voice,{" "}
              <span style={{
                background: "linear-gradient(135deg, #ea580c 0%, #dc2626 38%, #d97706 70%, #0d9488 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}>Any Accent</span>
              <br />Across the World
            </h2>
            <p style={{ fontSize:"1rem", color:"#78716c", lineHeight:1.75,
              maxWidth:500, margin:"0 auto" }}>
              AI-powered accent conversion that preserves your emotion, prosody, and
              voice intensity — like your words flying across the globe, landing perfectly in a new accent.
            </p>
          </div>

          {/* City destination pills */}
          <div style={{ display:"flex", gap:"0.5rem", flexWrap:"wrap", justifyContent:"center" }}>
            {ACCENTS.map(a => (
              <span key={a.key} style={{
                display:"flex", alignItems:"center", gap:"0.375rem",
                padding:"0.3rem 0.875rem", borderRadius:99,
                background: selectedAccent===a.key ? `${ACCENT_COLORS[a.key]}12` : "rgba(255,255,255,0.85)",
                border:`1.5px solid ${selectedAccent===a.key ? ACCENT_COLORS[a.key]+"55" : "rgba(234,88,12,0.12)"}`,
                fontSize:"0.78rem", fontWeight:600,
                color: selectedAccent===a.key ? ACCENT_COLORS[a.key] : "#78716c",
                boxShadow:"0 1px 4px rgba(120,50,0,0.06)",
                transition:"all 0.2s",
              }}>
                <span>{a.flag}</span> {a.city}
              </span>
            ))}
          </div>
        </section>

        {/* ── WORKSPACE ──────────────────────────────────────────── */}
        <main style={{ maxWidth:1200, margin:"0 auto", padding:"0 1.5rem 3rem" }}>
          <div className="two-col" style={{ display:"flex", gap:"1.5rem" }}>

            {/* ── LEFT: Upload + Accent + Convert ─────────────────── */}
            <div className="glass-card" style={{ flex:1, padding:"2rem", minWidth:0 }}>
              <div className="hud-corner hud-tl"/><div className="hud-corner hud-tr"/>
              <div className="hud-corner hud-bl"/><div className="hud-corner hud-br"/>

              <p className="section-label" style={{ marginBottom:"1.25rem" }}>SOURCE AUDIO</p>

              {/* Drop zone */}
              <div className={`drop-zone ${dragging?"drag-over":""}`}
                style={{ padding:"1.75rem 1rem", textAlign:"center", marginBottom:"1.5rem" }}
                onDragOver={e=>{e.preventDefault();setDragging(true);}}
                onDragLeave={()=>setDragging(false)}
                onDrop={onDrop}
                onClick={()=>fileInputRef.current?.click()}
                role="button" tabIndex={0} aria-label="Upload audio"
                onKeyDown={e=>e.key==="Enter"&&fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" accept=".wav,.mp3,.flac,.m4a,audio/*"
                  style={{ display:"none" }} id="audio-file-input"
                  onChange={e=>e.target.files?.[0]&&handleFile(e.target.files[0])} />
                {sourceFile ? (
                  <div className="fade-in">
                    <div style={{ fontSize:"2.5rem", marginBottom:"0.5rem" }}>🎵</div>
                    <p style={{ fontWeight:700, fontFamily:"var(--font-display)", marginBottom:"0.25rem",
                      color:"#1c0a00", fontSize:"0.9rem" }}>{sourceFile.name}</p>
                    <p style={{ fontSize:"0.75rem", color:"#a8a29e" }}>
                      {(sourceFile.size/1024/1024).toFixed(2)} MB · click to change
                    </p>
                  </div>
                ) : (
                  <div>
                    <div style={{ width:58, height:58, borderRadius:"50%",
                      background:"linear-gradient(135deg, rgba(234,88,12,0.10), rgba(220,38,38,0.07))",
                      border:"1.5px dashed rgba(234,88,12,0.32)",
                      display:"flex", alignItems:"center", justifyContent:"center",
                      margin:"0 auto 1rem", boxShadow:"0 2px 12px rgba(234,88,12,0.10)" }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                        stroke="#ea580c" strokeWidth="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                      </svg>
                    </div>
                    <p style={{ fontWeight:700, color:"#1c0a00", marginBottom:"0.375rem" }}>
                      Drop your audio file here
                    </p>
                    <p style={{ fontSize:"0.75rem", color:"#a8a29e" }}>
                      WAV · MP3 · FLAC · M4A — max 50 MB
                    </p>
                  </div>
                )}
              </div>

              {/* Source waveform */}
              <div style={{ marginBottom:"1.75rem" }}>
                <p style={{ fontSize:"0.65rem", fontWeight:700, letterSpacing:"0.12em",
                  textTransform:"uppercase", color:"#a8a29e", marginBottom:"0.625rem" }}>
                  SOURCE WAVEFORM
                </p>
                <WavePlayer blob={sourceBlob} color="#ea580c" id="source-waveform" />
              </div>

              {/* Accent selector */}
              <div style={{ marginBottom:"1.75rem" }}>
                <p style={{ fontSize:"0.65rem", fontWeight:700, letterSpacing:"0.12em",
                  textTransform:"uppercase", color:"#a8a29e", marginBottom:"0.75rem" }}>
                  ✈ TARGET DESTINATION
                </p>
                <div className="accent-chips" style={{ display:"flex", gap:"0.5rem", flexWrap:"wrap" }}>
                  {ACCENTS.map(a => (
                    <button key={a.key} id={`accent-${a.key}`}
                      className={`accent-chip ${selectedAccent===a.key?"selected":""}`}
                      style={selectedAccent===a.key ? {
                        borderColor: ACCENT_COLORS[a.key],
                        color:       ACCENT_COLORS[a.key],
                        background:  `${ACCENT_COLORS[a.key]}10`,
                        boxShadow:   `0 2px 12px ${ACCENT_COLORS[a.key]}30`,
                      } : {}}
                      onClick={()=>setSelectedAccent(a.key)}>
                      <span style={{ fontSize:"1rem" }}>{a.flag}</span>
                      <span>{a.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Convert */}
              <button id="convert-btn" className="btn-primary" disabled={!canConvert}
                style={{ width:"100%", fontSize:"1rem", padding:"0.9rem",
                  ...(canConvert ? {
                    background: `linear-gradient(135deg, ${accentColor}, #ea580c 50%, #d97706)`,
                    boxShadow:  `0 6px 24px ${accentColor}45`,
                  } : {
                    background: "linear-gradient(135deg, #ea580c, #dc2626 55%, #d97706)",
                  }) }}
                onClick={handleConvert}>
                {convStatus==="converting"
                  ? <><span className="spinner"/> Converting — processing audio…</>
                  : <>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14M12 5l7 7-7 7"/>
                      </svg>
                      Launch Conversion
                      {accent && <span style={{ fontSize:"0.95rem" }}>{accent.flag}</span>}
                    </>
                }
              </button>

              {convStatus==="converting" && (
                <div className="shimmer" style={{ marginTop:"0.75rem", borderRadius:8, height:3 }} />
              )}
              {convStatus==="converting" && (
                <p style={{ textAlign:"center", fontSize:"0.75rem", color:"#a8a29e", marginTop:"0.625rem" }}>
                  ✈️ Estimated flight time: 15 – 90 seconds
                </p>
              )}
              {convStatus==="error" && (
                <div className="fade-in" style={{ marginTop:"0.875rem", padding:"0.75rem 1rem",
                  background:"rgba(220,38,38,0.06)", border:"1.5px solid rgba(220,38,38,0.22)",
                  borderRadius:8, color:"#dc2626", fontSize:"0.82rem" }}>
                  ⚠ {errorMsg}
                </div>
              )}
            </div>

            {/* ── RIGHT: Output + Metrics ──────────────────────────── */}
            <div className="glass-card" style={{ flex:1, padding:"2rem", minWidth:0 }}>
              <div className="hud-corner hud-tl"/><div className="hud-corner hud-tr"/>
              <div className="hud-corner hud-bl"/><div className="hud-corner hud-br"/>

              <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between",
                marginBottom:"1.25rem" }}>
                <p className="section-label">OUTPUT AUDIO</p>
                {accent && convStatus==="done" && (
                  <div style={{ display:"flex", alignItems:"center", gap:"0.5rem",
                    padding:"0.3rem 0.75rem", borderRadius:99,
                    background:`${accentColor}10`, border:`1.5px solid ${accentColor}45`,
                    boxShadow:`0 2px 8px ${accentColor}22` }}>
                    <span>{accent.flag}</span>
                    <span style={{ fontSize:"0.7rem", fontWeight:800, color:accentColor }}>
                      {accent.label.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>

              <div style={{ marginBottom:"1.5rem" }}>
                {convStatus==="converting" ? (
                  <div style={{ height:88, borderRadius:12,
                    background:"rgba(234,88,12,0.04)",
                    border:"1.5px dashed rgba(234,88,12,0.22)",
                    display:"flex", flexDirection:"column", alignItems:"center",
                    justifyContent:"center", gap:"0.75rem" }}>
                    <SoundWave active={true} color={accentColor} />
                    <span style={{ fontSize:"0.75rem", color:"#a8a29e" }}>Synthesising…</span>
                  </div>
                ) : (
                  <div className={convStatus==="done"?"fade-in":""}>
                    <WavePlayer blob={outputBlob} color={accentColor} id="output-waveform" />
                  </div>
                )}
              </div>

              {outputBlob && (
                <button id="download-btn" className="btn-secondary scale-in"
                  onClick={downloadOutput}
                  style={{ width:"100%", marginBottom:"1.75rem",
                    borderColor:`${accentColor}45`, color:accentColor,
                    background:`${accentColor}08` }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download Converted WAV
                </button>
              )}

              {metrics ? (
                <div className="fade-in">
                  <p style={{ fontSize:"0.65rem", fontWeight:700, letterSpacing:"0.12em",
                    textTransform:"uppercase", color:"#a8a29e", marginBottom:"0.875rem" }}>
                    EMOTION PRESERVATION ANALYSIS
                  </p>
                  <EmotionBar label="Valence"   src={metrics.valence_source}   out={metrics.valence_output}   color="#ea580c" />
                  <EmotionBar label="Arousal"   src={metrics.arousal_source}   out={metrics.arousal_output}   color="#dc2626" />
                  <EmotionBar label="Dominance" src={metrics.dominance_source} out={metrics.dominance_output} color="#0d9488" />
                  <p style={{ fontSize:"0.68rem", color:"#a8a29e", marginTop:"0.5rem" }}>
                    Grey fill = source · Coloured dot = output · Closer = better
                  </p>
                </div>
              ) : (
                <div style={{ display:"flex", flexDirection:"column", alignItems:"center",
                  justifyContent:"center", gap:"1rem", padding:"2rem 0", opacity:0.45 }}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
                    stroke="#ea580c" strokeWidth="1" strokeLinecap="round">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M2 12h20M12 2a15 15 0 0 1 4 10 15 15 0 0 1-4 10 15 15 0 0 1-4-10 15 15 0 0 1 4-10z"/>
                  </svg>
                  <p style={{ fontSize:"0.8rem", color:"#a8a29e", textAlign:"center" }}>
                    Upload audio and select a destination<br />to begin the journey
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* ── FLIGHT REPORT ────────────────────────────────────── */}
          {metrics && (
            <div className="slide-up" style={{ marginTop:"1.5rem" }}>
              <div style={{ display:"flex", alignItems:"center", gap:"1rem", marginBottom:"1.25rem" }}>
                <div style={{ flex:1, height:1,
                  background:"linear-gradient(90deg, transparent, rgba(234,88,12,0.30), transparent)" }}/>
                <p style={{ fontSize:"0.65rem", fontWeight:800, letterSpacing:"0.22em",
                  textTransform:"uppercase", color:"#d97706" }}>✈ FLIGHT REPORT</p>
                <div style={{ flex:1, height:1,
                  background:"linear-gradient(90deg, transparent, rgba(234,88,12,0.30), transparent)" }}/>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(130px, 1fr))", gap:"1rem" }}>
                {[
                  { label:"Emotion Sim.",   value:pct(metrics.emotion_similarity),       cls:grade(metrics.emotion_similarity,0.85,0.6),        sub:metrics.emotion_similarity>=0.85?"✓ Preserved":metrics.emotion_similarity>=0.6?"⚠ Partial":"✗ Drifted",   delay:"0.05s"},
                  { label:"Word Error Rate",value:pct(metrics.wer),                      cls:grade(metrics.wer,0.1,0.3,false),                   sub:metrics.wer<=0.1?"✓ Accurate":metrics.wer<=0.3?"⚠ Fair":"✗ High",                                         delay:"0.10s"},
                  { label:"UTMOS Score",    value:`${metrics.mos_estimate.toFixed(2)}/5`,cls:grade(metrics.mos_estimate,3.8,3.0),                sub:metrics.mos_estimate>=3.8?"✓ Natural":metrics.mos_estimate>=3.0?"⚠ OK":"✗ Artefacts",                     delay:"0.15s"},
                  { label:"Segments",       value:String(metrics.n_segments),            cls:"" as const,                                        sub:"VAD chunks",                                                                                              delay:"0.20s"},
                  { label:"Flight Time",    value:fmtMs(metrics.processing_time_ms),     cls:"" as const,                                        sub:"end-to-end",                                                                                             delay:"0.25s"},
                ].map(m => (
                  <div key={m.label} className={`metric-card ${m.cls}`} style={{ animationDelay:m.delay }}>
                    <span className="metric-label">{m.label}</span>
                    <span className="metric-value">{m.value}</span>
                    <span style={{ fontSize:"0.68rem", color:"#a8a29e" }}>{m.sub}</span>
                  </div>
                ))}
                <div className="metric-card" style={{ animationDelay:"0.30s" }}>
                  <span className="metric-label">Engine</span>
                  <span style={{ fontSize:"0.8rem", fontWeight:800, color:"#ea580c",
                    marginTop:"0.25rem", textTransform:"capitalize" }}>
                    {[...new Set(metrics.chosen_backends)].join(" + ")}
                  </span>
                  <span style={{ fontSize:"0.68rem", color:"#a8a29e" }}>winner / segment</span>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* ── FOOTER ─────────────────────────────────────────────── */}
        <footer style={{ borderTop:"1px solid rgba(234,88,12,0.10)",
          padding:"1.5rem 2rem",
          display:"flex", justifyContent:"space-between", alignItems:"center",
          flexWrap:"wrap", gap:"0.5rem",
          background:"rgba(255,251,245,0.75)", backdropFilter:"blur(12px)" }}>
          <div style={{ display:"flex", alignItems:"center", gap:"0.5rem" }}>
            <span className="gradient-text"
              style={{ fontFamily:"var(--font-display)", fontWeight:800, fontSize:"0.9rem" }}>
              AccentShift
            </span>
            <span style={{ color:"#a8a29e", fontSize:"0.75rem" }}>· DP6 Honeywell Designathon 2025</span>
          </div>
          <span style={{ fontSize:"0.72rem", color:"#a8a29e" }}>
            Seed-VC V2 · Vevo (Amphion) · Whisper large-v3 · wav2vec2 SER · pyworld
          </span>
        </footer>
      </div>
    </>
  );
}
