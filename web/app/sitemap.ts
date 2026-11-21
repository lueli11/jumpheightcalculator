import type { MetadataRoute } from "next";

// Nur statische, öffentlich sinnvolle Seiten. /analysis/[id] gehört bewusst
// nicht hinein: die Seiten entstehen pro Upload, sind nicht auffindbar und
// hätten für eine Suchmaschine keinen Wert.
export const baseUrl =
  process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "") ??
  "https://jumplab.jonasleuenberger.com";

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
