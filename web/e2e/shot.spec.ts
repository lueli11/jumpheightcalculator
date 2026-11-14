import { test } from "@playwright/test";
const AID = process.env.JUMPHEIGHTCALCULATOR_AID!;

test("P3.2 Overlay ansehen", async ({ page }) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 760, height: 1500 });
  await page.goto(`/analysis/${AID}`);
  await page.getByTestId("height").waitFor({ timeout: 60_000 });
  await page.waitForTimeout(2500);
  console.log("Höhe:", await page.getByTestId("height").innerText());
  await page.locator(".stage").screenshot({ path: "shot-takeoff.png" });
  for (let i = 0; i < 45; i++) await page.getByRole("button", { name: /Frame\s*→/ }).click();
  await page.waitForTimeout(1500);
  await page.locator(".stage").screenshot({ path: "shot-apex.png" });
  await page.screenshot({ path: "shot-full.png", fullPage: true });
});
