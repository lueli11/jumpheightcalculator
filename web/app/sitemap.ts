import type { MetadataRoute } from "next";

// Nur statische, oeffentlich sinnvolle Seiten. /analysis/[id] gehoert bewusst
// nicht hinein: die Seiten entstehen pro Upload, sind nicht auffindbar und
// haetten fuer eine Suchmaschine keinen Wert.
export const baseUrl =
  process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "") ??
  "https://jumplab.dokploy.jonasleuenberger.com";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: `${baseUrl}/`,
      lastModified: new Date(),
      changeFrequency: "monthly",
      priority: 1,
    },
  ];
}
