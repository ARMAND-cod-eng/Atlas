import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';

import { THEME_KEY } from '@/lib/theme';

import './globals.css';

// Runs before the first paint, so a viewer who chose a theme never sees the
// other one flash in. Storage can be blocked; the system theme applies then.
const THEME_SCRIPT = `try{var t=localStorage.getItem('${THEME_KEY}');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}`;

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Atlas',
  description: 'Compliance research on the EU AI Act and GDPR, with every citation checked against the law',
};

export default function RootLayout({
  children,
}: {
  readonly children: React.ReactNode;
}): React.JSX.Element {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      // The theme script sets data-theme before React hydrates.
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col bg-canvas text-ink">{children}</body>
    </html>
  );
}
