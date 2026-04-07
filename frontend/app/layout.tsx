import type { Metadata } from 'next';
import "./globals.css";
import { Toaster } from 'sonner';

export const metadata: Metadata = {
  title: "FraudSense AI | Command Center",
  description: "Enterprise fraud detection and monitoring console",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="app-shell font-sans">
        <Toaster position="top-right" richColors />
        {children}
      </body>
    </html>
  );
}

