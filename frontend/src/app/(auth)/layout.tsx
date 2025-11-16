import "~/styles/globals.css";
import { type Metadata } from "next";
import { Geist } from "next/font/google";
import { Providers } from "~/components/provider"
import { Toaster } from "sonner";

export const metadata: Metadata = {
  title: "Synthoria",
  description: "A full-stack AI music platform",
  icons: [{ rel: "icon", url: "/favicon.png" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable}`}>
      <body className="flex flex-col min-h-svh" style={{
        backgroundImage: "url(/background.jpg)",
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}>
        <Providers>
            {children}
            <Toaster/>
        </Providers>
      </body>
    </html>
  );
}
