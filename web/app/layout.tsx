import "./globals.css";
export const metadata = { title: "jumpheightcalculator" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="de"><body><main>{children}</main></body></html>;
}
