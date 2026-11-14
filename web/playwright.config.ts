import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 15 * 60 * 1000,          // eine echte Analyse dauert Minuten
  expect: { timeout: 15_000 },
    // Hinweis: Playwrights Chromium ist ein Open-Source-Build ohne H.264. Das
  // Vorschauvideo ist H.264, weil es das für iOS sein muss - im Test bleibt die
  // Videospur deshalb leer. Das Overlay wird trotzdem geprüft, weil die Bühne
  // ihr Seitenverhältnis aus den Trackdaten bezieht, nicht aus dem Video.
  use: { baseURL: "http://localhost:3050", trace: "retain-on-failure" },
  reporter: [["list"]],
});
