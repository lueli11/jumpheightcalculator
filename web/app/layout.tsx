import "./globals.css";
import type { Metadata } from "next";
import { baseUrl } from "./sitemap";

export const metadata: Metadata = {
  metadataBase: new URL(baseUrl),
  title: "JumpLab – Sprunghöhe aus Video messen",
  description:
    "Sprunghöhe aus einem Slo-Mo-Video messen: JumpLab verfolgt den Sprung Frame für Frame und berechnet Höhe, Flugzeit und Absprunggeschwindigkeit.",
  alternates: { canonical: "/" },
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="de">
      <body>
        <main>{children}</main>
        {/* Umami (zentral im Dokploy-Projekt infra), cookielos. data-domains: lokale Aufrufe zählen nicht. */}
        <script
          async
          src="https://t.jonasleuenberger.com/t.js"
          data-website-id="4134173b-b22e-462a-86f5-38d5a11ab3f8"
          data-domains="jumplab.jonasleuenberger.com"
        />
      </body>
    </html>
  );
}
