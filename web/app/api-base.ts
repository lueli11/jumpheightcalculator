/** Basis-URL der API.
 *
 * Der Browser spricht die API direkt an statt über den Next-Proxy: ein
 * 134-MB-Upload durch die Rewrite-Schicht riss reproduzierbar mit ECONNRESET ab,
 * weil der Körper zweimal gepuffert wird. In Compose läuft die API auf Port
 * 8000 desselben Hosts, deshalb der Rückfall auf die aktuelle Hostadresse.
 */
export function apiBase(): string {
  const fromEnv = process.env.NEXT_PUBLIC_API_BASE;
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
}
