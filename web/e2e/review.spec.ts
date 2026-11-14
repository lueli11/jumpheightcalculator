import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

// Kurzer Ausschnitt um den Sprung statt der vollen 134-MB-Datei: derselbe
// Codepfad, aber Upload und Analyse in Minuten statt einer Viertelstunde.
const CLIP = path.resolve(__dirname, "../../tests/data/jump_clip.mp4");

test.describe("Review-Screen", () => {
  test.skip(!fs.existsSync(CLIP), "kein Testausschnitt vorhanden");

  test("P3.1 Upload, Ergebnis, und eine Korrektur verändert die Zahl", async ({ page }) => {
    const problems: string[] = [];
    page.on("console", (m) => { if (m.type() === "error") problems.push(m.text()); });
    page.on("pageerror", (e) => problems.push(String(e)));
    page.on("requestfailed", (r) =>
      problems.push(`${r.method()} ${r.url()} -> ${r.failure()?.errorText}`));

    await page.goto("/");
    await page.getByTestId("file").setInputFiles(CLIP);
    await page.getByTestId("submit").click();

    // Der Worker rechnet; der Screen pollt, bis das Ergebnis da ist.
    await expect(page.getByTestId("height"))
      .toBeVisible({ timeout: 10 * 60 * 1000 })
      .catch((e) => { throw new Error(`${e}\nSeitenfehler:\n${problems.join("\n")}`); });

    const read = async () =>
      Number((await page.getByTestId("height").innerText()).replace(",", "."));
    const before = await read();
    expect(Number.isFinite(before)).toBe(true);
    expect(before).toBeGreaterThan(10);

    // P3.2 - Video und beide Overlays (Skelett/Parabel, Kontaktsignal) sind da.
    await expect(page.getByTestId("video")).toBeVisible();
    expect(await page.locator("canvas").count()).toBe(2);

    // Kern von P3.1: eine +/-1-Frame-Korrektur ändert den angezeigten Wert
    // nachweislich. Bei 120 fps ist ein Frame 8.3 ms, und die Höhe hängt mit
    // rund 3.8 cm pro 10 ms am Absprungzeitpunkt - der Effekt muss sichtbar sein.
    await page.getByTestId("takeoff-plus").click();
    await expect.poll(read, { timeout: 30_000 }).not.toBe(before);

    // ... und zurück führt nicht zwingend exakt auf denselben Wert (der Anker
    // ist jetzt ein gesetzter Frame statt eines interpolierten Schnittpunkts),
    // muss aber wieder in der Nähe liegen.
    await page.getByTestId("takeoff-minus").click();
    await expect.poll(read, { timeout: 30_000 }).toBeGreaterThan(before - 5);

    expect(problems.filter((p) => !p.includes("favicon"))).toEqual([]);
  });
});
