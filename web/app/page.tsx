"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiBase } from "./api-base";

export default function Upload() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [warn, setWarn] = useState<string | null>(null);
  const [mass, setMass] = useState("");
  const [fps, setFps] = useState("");

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const input = (e.currentTarget.elements.namedItem("file") as HTMLInputElement);
    if (!input.files?.[0]) return;
    setBusy(true); setErr(null); setWarn(null);
    try {
      const fd = new FormData();
      fd.append("file", input.files[0]);
      // Nur mitschicken, wenn wirklich etwas drinsteht. Ein leeres Feld heisst
      // "nicht angegeben", nicht "0" - genau wie beim Gewicht.
      if (fps) fd.append("capture_fps", fps);
      const up = await fetch(apiBase() + "/api/videos", { method: "POST", body: fd });
      if (!up.ok) throw new Error((await up.json()).detail ?? "Upload fehlgeschlagen");
      const video = await up.json();
      if (video.warning) setWarn(video.warning);

      const an = await fetch(apiBase() + "/api/analyses", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: video.video_id, jump_type: "cmj",
          mass_kg: mass ? Number(mass) : null,
        }),
      });
      if (!an.ok) throw new Error("Analyse konnte nicht gestartet werden");
      router.push(`/analysis/${(await an.json()).job_id}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <>
      <h1>jumpheightcalculator</h1>
      <p className="sub">Sprunghöhe aus Video, Frame für Frame.</p>

      <form onSubmit={submit} className="panel">
        <div className="row">
          <input type="file" name="file" accept="video/*" required data-testid="file" />
          <input type="number" placeholder="Gewicht kg" value={mass} min={20} max={250}
                 onChange={(e) => setMass(e.target.value)} style={{ width: 120 }}
                 title="Optional - nur für Impuls und mittlere Kraft" />
          <input type="number" placeholder="fps der Aufnahme" value={fps}
                 min={20} max={1000} data-testid="capture-fps"
                 onChange={(e) => setFps(e.target.value)} style={{ width: 150 }}
                 title="Nur nötig, wenn das Slo-Mo bereits verlangsamt exportiert wurde" />
          <button type="submit" disabled={busy} data-testid="submit">
            {busy ? "lädt hoch…" : "Analysieren"}
          </button>
        </div>
      </form>

      <div className="panel" style={{ fontSize: 14.5 }}>
        <b>Das Gewicht ist optional.</b> Es wird für genau zwei Grössen gebraucht:
        <div className="mono" style={{ margin: "10px 0", lineHeight: 1.8 }}>
          J = m · v₀ &nbsp;&nbsp;<span style={{ opacity: 0.6 }}>Absprungimpuls [Ns]</span>
          <br />
          F̄ = m · v₀ / t_abdruck + m · g &nbsp;&nbsp;
          <span style={{ opacity: 0.6 }}>mittlere Kraft [N]</span>
        </div>
        Höhe, Absprunggeschwindigkeit und Flugzeit kommen ohne aus.
        <p style={{ margin: "10px 0 0" }}>
          <b>Wattzahlen gibt es bewusst nicht.</b> Peak Power braucht eine
          Kraftmessplatte - aus Video ist nur der Mittelwert über die Abdruckphase
          seriös.
        </p>
      </div>

      <div className="panel" style={{ fontSize: 14.5 }}>
        <b>Die Bildrate brauchst du fast nie.</b> Nur wenn dein Slo-Mo schon
        <i> verlangsamt</i> exportiert wurde: dann steht im Video die
        Abspielrate, nicht die Aufnahmerate, und die Zeitachse ist gestreckt.
        Die Höhe hängt <b>quadratisch</b> daran - vierfach gestreckt heisst
        sechzehnfach falsch. Die App merkt es und sagt es dir; hier trägst du
        dann nach, mit wie vielen fps du gefilmt hast (meist 120 oder 240).
      </div>

      {warn && <div className="panel warn" data-testid="warning">{warn}</div>}
      {err && <div className="panel err" data-testid="error">{err}</div>}

      <h2>So aufnehmen</h2>
      <ul>
        <li><b>Slo-Mo einschalten</b> - 240 fps wenn möglich, mindestens 120.</li>
        <li><b>Kurze Belichtung.</b> Hell aufnehmen, sonst verschmiert der Fuss.</li>
        <li><b>Handy quer auf ein Stativ.</b> Abstand und Kamerahöhe sind
            unkritisch - nachgerechnet von 2 bis 10 m und von 5 cm bis 2.5 m Höhe
            ohne messbare Abweichung, weil der Massstab aus der Physik kommt und
            nicht angenommen wird. Nur zu weit weg wird das Skelett unsicher.</li>
        <li><b>Senkrecht zur Sprungebene filmen und senkrecht springen.</b> Wer
            auf die Kamera zu springt, wird im Bild grösser - dann ändert sich der
            Massstab mitten im Sprung und die Höhe ist nicht mehr bestimmbar. Das
            ist der häufigste Grund, weshalb eine Aufnahme abgelehnt wird.</li>
        <li><b>Ganzer Körper im Bild</b>, auch die Füsse im Scheitelpunkt.</li>
        <li><b>Enge Kleidung, kontrastreicher Boden.</b> Weite Hosen sind der
            häufigste Grund für schlechte Knöchel-Keypoints.</li>
      </ul>
    </>
  );
}
