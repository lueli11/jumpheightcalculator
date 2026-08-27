"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { apiBase } from "../../api-base";

// MediaPipe-Skelett: nur die Glieder, die man zur Beurteilung braucht.
const BONES: [number, number][] = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
];
const HIP_L = 23, HIP_R = 24;

type Track = {
  t: number[]; xy: number[][][]; width: number; height: number;
  contact: number[]; n0: number; n1: number; fps: number;
};
type Result = Record<string, number | string | boolean | string[] | null>;

export default function Review() {
  const { id } = useParams<{ id: string }>();
  const [status, setStatus] = useState("laedt");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [track, setTrack] = useState<Track | null>(null);
  const [frame, setFrame] = useState(0);
  const [n0, setN0] = useState<number | null>(null);
  const [n1, setN1] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const signal = useRef<HTMLCanvasElement>(null);

  // Polling, bis die Analyse fertig ist. Sie läuft im Worker, nicht im Request.
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const r = await fetch(`${apiBase()}/api/analyses/${id}`).then((x) => x.json());
      if (stop) return;
      setStatus(r.status);
      if (r.status === "ready") {
        setResult(r.result);
        const tr: Track = await fetch(`${apiBase()}/api/analyses/${id}/track`).then((x) => x.json());
        if (stop) return;
        setTrack(tr); setN0(tr.n0); setN1(tr.n1); setFrame(tr.n0);
      } else if (r.status === "failed") {
        setError(r.error);
        // Auch bei einer Verweigerung den Track holen: die teure Extraktion ist
        // gelaufen, und der Review-Screen soll die Ereignisframes von Hand
        // korrigierbar machen statt in einer Sackgasse zu enden.
        const tr: Track | null = await fetch(`${apiBase()}/api/analyses/${id}/track`)
          .then((x) => (x.ok ? x.json() : null))
          .catch(() => null);
        if (stop || !tr) return;
        setTrack(tr); setN0(tr.n0); setN1(tr.n1); setFrame(tr.n0);
      } else {
        setTimeout(tick, 2000);
      }
    };
    tick();
    return () => { stop = true; };
  }, [id]);

  // Frame-Stepper über die PTS-Tabelle, nicht über eine angenommene Framerate:
  // die Zeitstempel sind variabel, currentTime = n/fps wäre schlicht falsch (R1).
  useEffect(() => {
    if (track && video.current) video.current.currentTime = track.t[frame] ?? 0;
  }, [frame, track]);

  const draw = useCallback(() => {
    if (!track) return;
    const a0 = n0 ?? track.n0;
    const a1 = n1 ?? track.n1;

    const c = overlay.current;
    if (c && result) {
      c.width = track.width; c.height = track.height;
      const g = c.getContext("2d");
      if (g) {
        g.clearRect(0, 0, c.width, c.height);

        // Gefittete Parabel über das Flugfenster - genau die Kurve, aus der die
        // Zahl kommt. Deckt sie sich nicht mit der Hüfte, stimmt die Zahl nicht.
        const r = result as unknown as Record<string, number>;
        g.strokeStyle = "#3ed0de"; g.lineWidth = 2; g.setLineDash([6, 4]);
        g.beginPath();
        for (let i = a0; i <= a1 && i < track.t.length; i++) {
          const u = track.t[i] - r.fit_t_ref;
          const y = r.fit_c0 + r.fit_c1 * u + r.fit_c2 * u * u;
          const p = track.xy[i];
          const x = (p[HIP_L][0] + p[HIP_R][0]) / 2;
          if (i === a0) g.moveTo(x, y); else g.lineTo(x, y);
        }
        g.stroke(); g.setLineDash([]);

        // Skelett des aktuellen Frames
        const p = track.xy[frame];
        if (p) {
          g.strokeStyle = "#e4eaf0cc"; g.lineWidth = 2;
          for (const [a, b] of BONES) {
            if (p[a][0] < 0 || p[b][0] < 0) continue;
            g.beginPath(); g.moveTo(p[a][0], p[a][1]); g.lineTo(p[b][0], p[b][1]); g.stroke();
          }
          g.fillStyle = "#3ed0de";
          g.beginPath();
          g.arc((p[HIP_L][0] + p[HIP_R][0]) / 2, (p[HIP_L][1] + p[HIP_R][1]) / 2,
                5, 0, Math.PI * 2);
          g.fill();
        }
      }
    }

    const s = signal.current;
    if (s) {
      // Kontaktsignal auf derselben Achse wie das Flugfenster darüber.
      const W = s.width = Math.max(600, s.clientWidth * 2);
      const H = s.height = 120;
      const g = s.getContext("2d");
      if (!g) return;
      g.clearRect(0, 0, W, H);
      const lo = Math.max(0, a0 - 40);
      const hi = Math.min(track.contact.length - 1, a1 + 40);
      const px = (i: number) => ((i - lo) / Math.max(1, hi - lo)) * W;

      g.fillStyle = "#3ed0de1f";
      g.fillRect(px(a0), 0, px(a1) - px(a0), H);

      g.strokeStyle = "#3ed0de"; g.lineWidth = 2; g.beginPath();
      for (let i = lo; i <= hi; i++) {
        const y = H - 10 - track.contact[i] * (H - 20);
        if (i === lo) g.moveTo(px(i), y); else g.lineTo(px(i), y);
      }
      g.stroke();

      g.strokeStyle = "#ff6e6e"; g.lineWidth = 3;
      for (const i of [a0, a1]) {
        g.beginPath(); g.moveTo(px(i), 0); g.lineTo(px(i), H); g.stroke();
      }
      g.strokeStyle = "#e4eaf099"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(px(frame), 0); g.lineTo(px(frame), H); g.stroke();
    }
  }, [track, result, frame, n0, n1]);

  useEffect(() => { draw(); }, [draw]);

  async function correct(dTakeoff: number, dLanding: number) {
    if (!track || n0 === null || n1 === null) return;
    const body = {
      takeoff_frame: Math.max(0, n0 + dTakeoff),
      landing_frame: Math.min(track.t.length - 1, n1 + dLanding),
    };
    if (body.landing_frame - body.takeoff_frame < 10) return;

    // Marker sofort setzen, nicht erst nach der Antwort. Der rote Strich und die
    // Flugzeit haengen allein an den beiden Frames - beides ist ohne Server
    // bekannt. Nur die Hoehe braucht einen neuen Fit; die kommt nach.
    setN0(body.takeoff_frame); setN1(body.landing_frame);
    setFrame(dTakeoff !== 0 ? body.takeoff_frame : body.landing_frame);
    setBusy(true);
    try {
      const r = await fetch(`${apiBase()}/api/analyses/${id}/correct`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await r.json();
      if (r.ok) {
        setResult(payload.result); setError(null);
      } else {
        // Die Marker bleiben stehen, wo sie sind: sonst koennte man sich nicht
        // schrittweise an eine gueltige Stellung herantasten.
        setResult(null); setError(payload.detail);
      }
    } finally {
      setBusy(false);
    }
  }

  if (error && !track) {
    return (
      <>
        <h1>Keine Messung möglich</h1>
        <div className="panel err" data-testid="refusal">{error}</div>
        <a href="/">zurück</a>
      </>
    );
  }
  if (!track || (status !== "ready" && !error)) {
    return (
      <>
        <h1>Analyse läuft</h1>
        <p className="sub" data-testid="status">Status: {status}</p>
      </>
    );
  }

  const cm = (v: unknown) => (typeof v === "number" ? (v * 100).toFixed(1) : "-");
  const num = (v: unknown, d: number) => (typeof v === "number" ? v.toFixed(d) : "-");

  // Flugzeit sofort aus der PTS-Tabelle: sie haengt nur an den beiden gesetzten
  // Frames, nicht am Fit. Waehrend der Server neu rechnet, steht hier also schon
  // der richtige Wert, statt dass die Anzeige stehenbleibt.
  const a0 = n0 ?? track.n0;
  const a1 = n1 ?? track.n1;
  const liveTf = (track.t[Math.min(a1, track.t.length - 1)] ?? 0) - (track.t[a0] ?? 0);

  return (
    <>
      <h1>{result ? "Ergebnis" : "Keine Messung möglich"}</h1>

      <div className="review">
        <div className="col-left">
          {/* Seitenverhaeltnis aus den Trackdaten, nicht aus dem geladenen Video:
              laedt die Videospur nicht (fehlender Codec, langsames Netz),
              kollabiert die Buehne sonst und streckt das Overlay. */}
          <div className="stage"
               style={{ aspectRatio: `${track.width} / ${track.height}` }}>
            <video ref={video} src={`${apiBase()}/api/analyses/${id}/video`}
                   preload="auto" playsInline muted onLoadedMetadata={draw}
                   data-testid="video" />
            <canvas ref={overlay} />
          </div>

          <div className="row" style={{ marginTop: 10 }}>
            <button onClick={() => setFrame((f) => Math.max(0, f - 1))}>&larr;</button>
            <span className="mono" style={{ fontSize: 14 }}>
              Frame {frame} &middot; {num(track.t[frame], 3)} s
            </span>
            <button onClick={() => setFrame((f) => Math.min(track.t.length - 1, f + 1))}>
              &rarr;
            </button>
            <button onClick={() => setFrame(a0)}>Absprung</button>
            <button onClick={() => setFrame(a1)}>Landung</button>
          </div>
        </div>

        <div>
          {result ? (
            <p className="sub" style={{ marginBottom: 12 }}>
              <span className="big mono" data-testid="height">{cm(result.h_e2)}</span>
              <span className="unit"> cm Sprunghöhe</span>
              <br />
              <span className="mono" style={{ fontSize: 15 }}>
                {(liveTf * 1000).toFixed(0)} ms Flugzeit
                {busy ? " · rechnet…" : ""}
              </span>
            </p>
          ) : (
            <>
              <div className="panel err" data-testid="refusal">{error}</div>
              <p className="sub" style={{ fontSize: 14 }}>
                Die Kurve ist das Kontaktsignal: <b>hoch heisst Fuss am Boden</b>,
                tief heisst Flug. Die roten Striche sind Absprung und Landung.
                Stimmen sie nicht mit den Flanken überein, verschiebe sie unten.
                Aktuell {(liveTf * 1000).toFixed(0)} ms Flugzeit.
              </p>
            </>
          )}

          <canvas ref={signal}
                  style={{ width: "100%", height: 110, marginTop: 4 }} />

          <div className="row" style={{ marginTop: 10 }}>
            <span className="mono" style={{ minWidth: 78, fontSize: 14 }}>Absprung</span>
            <button disabled={busy} onClick={() => correct(-1, 0)}
                    data-testid="takeoff-minus">−1</button>
            <button disabled={busy} onClick={() => correct(+1, 0)}
                    data-testid="takeoff-plus">+1</button>
            <span className="mono" style={{ minWidth: 78, marginLeft: 14, fontSize: 14 }}>
              Landung
            </span>
            <button disabled={busy} onClick={() => correct(0, -1)}>−1</button>
            <button disabled={busy} onClick={() => correct(0, +1)}>+1</button>
          </div>

          {result ? (<>
          <h2>Alle Schätzer</h2>
      <div className="grid">
        <div><dt>E2 Hüfte (Referenz)</dt><dd className="mono">{cm(result.h_e2)} cm</dd></div>
        <div><dt>E1 Flugzeit</dt><dd className="mono">{cm(result.h_e1)} cm</dd></div>
        <div><dt>E3 Schwerpunkt</dt><dd className="mono">{cm(result.h_e3)} cm</dd></div>
        <div><dt>E4 Fallphase</dt><dd className="mono">{cm(result.h_e4)} cm</dd></div>
        <div><dt>Absprunggeschw.</dt><dd className="mono">{num(result.v0, 2)} m/s</dd></div>
        <div><dt>Flugzeit</dt><dd className="mono">
          {typeof result.t_f === "number" ? (result.t_f * 1000).toFixed(0) : "-"} ms</dd></div>
        <div><dt>Delta Landung</dt><dd className="mono">{cm(result.delta)} cm</dd></div>
        <div><dt>Abdruckdauer</dt><dd className="mono">
          {typeof result.push_time === "number" ? (result.push_time * 1000).toFixed(0) : "-"} ms</dd></div>
        <div><dt>Güte</dt><dd className="mono">{num(result.quality, 2)}</dd></div>
      </div>

      {/* Impuls und mittlere Kraft gibt es nur mit Körpermasse - ohne sie fehlen
          schlicht die Eingangsgrössen. Keine Wattzahlen: Peak Power braucht den
          Kraft-Zeit-Verlauf einer Kraftmessplatte. */}
      {typeof result.impulse === "number" ? (
        <>
          <h2>Mit deinem Gewicht</h2>
          <p className="sub" style={{ fontSize: 13.5, marginTop: -4 }}>
            J = m · v₀ und F̄ = m · v₀ / t_abdruck + m · g. Keine Wattzahlen:
            Peak Power braucht eine Kraftmessplatte.
          </p>
          <div className="grid">
            <div><dt>Absprungimpuls</dt><dd className="mono">
              {num(result.impulse, 1)} Ns</dd></div>
            <div><dt>Mittlere Kraft</dt><dd className="mono">
              {num(result.mean_force, 0)} N</dd></div>
            {typeof result.mean_force === "number" && typeof result.mass_kg === "number" ? (
              <div><dt>entspricht</dt><dd className="mono">
                {(result.mean_force / (result.mass_kg * 9.81)).toFixed(2)} Körpergewichte</dd></div>
            ) : null}
          </div>
        </>
      ) : (
        <p className="sub" style={{ fontSize: 14 }}>
          Ohne Gewicht keine Impuls- und Kraftwerte (J = m · v₀ und
          F̄ = m · v₀ / t_abdruck + m · g). Höhe, Geschwindigkeit und Flugzeit
          brauchen es nicht.
        </p>
      )}

      <p className="sub mono" style={{ marginTop: 10, fontSize: 13 }}>
        R2 {num(result.r2_hip, 4)} &middot; |E2-E3| {cm(result.disagreement)} cm &middot;{" "}
        {num(result.fps_measured, 0)} fps &middot; {String(result.pipeline_version)}
        {result.corrected ? " · korrigiert" : ""}
      </p>
          {(result.warnings as string[] | null)?.map((w) => (
            <div key={w} className="panel warn">{w}</div>
          ))}
          </>) : null}
        </div>
      </div>
    </>
  );
}
