import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 15 * 60 * 1000,          // eine echte Analyse dauert Minuten
  expect: { timeout: 15_000 },
    // Hinweis: Playwrights Chromium ist ein Open-Source-Build ohne H.264. Das
  // Vorschauvideo ist H.264, weil es das fuer iOS sein muss - im Test bleibt die
  // Videospur deshalb leer. Das Overlay wird trotzdem geprueft, weil die Buehne
  // ihr Seitenverhaeltnis aus den Trackdaten bezieht, nicht aus dem Video.
  use: { baseURL: "http://localhost:3000", trace: "retain-on-failure" },
  reporter: [["list"]],
});
