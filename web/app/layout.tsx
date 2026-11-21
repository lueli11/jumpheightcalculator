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
  return <html lang="de"><body><main>{children}</main></body></html>;
}
