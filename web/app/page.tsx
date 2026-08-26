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

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const input = (e.currentTarget.elements.namedItem("file") as HTMLInputElement);
    if (!input.files?.[0]) return;
    setBusy(true); setErr(null); setWarn(null);
    try {
      const fd = new FormData();
      fd.append("file", input.files[0]);
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
      <p className="sub">Sprunghoehe aus Video, Frame fuer Frame.</p>

      <form onSubmit={submit} className="panel">
        <div className="row">
          <input type="file" name="file" accept="video/*" required data-testid="file" />
          <input type="number" placeholder="Gewicht kg" value={mass} min={20} max={250}
                 onChange={(e) => setMass(e.target.value)} style={{ width: 120 }} />
          <button type="submit" disabled={busy} data-testid="submit">
            {busy ? "laedt hoch..." : "Analysieren"}
          </button>
        </div>
      </form>

      {warn && <div className="panel warn" data-testid="warning">{warn}</div>}
      {err && <div className="panel err" data-testid="error">{err}</div>}

      <h2>So aufnehmen</h2>
      <ul>
        <li><b>Slo-Mo einschalten</b> - 240 fps wenn moeglich, mindestens 120.</li>
        <li><b>Kurze Belichtung.</b> Hell aufnehmen, sonst verschmiert der Fuss.</li>
        <li><b>Handy quer auf ein Stativ</b>, so hoch wie moeglich, 3-4 m Abstand.</li>
        <li><b>Senkrecht zur Sprungebene</b> filmen, nicht vor- oder zurueckspringen.</li>
        <li><b>Ganzer Koerper im Bild</b>, auch die Fuesse im Scheitelpunkt.</li>
        <li><b>Enge Kleidung, kontrastreicher Boden.</b> Weite Hosen sind der
            haeufigste Grund fuer schlechte Knoechel-Keypoints.</li>
      </ul>
    </>
  );
}
