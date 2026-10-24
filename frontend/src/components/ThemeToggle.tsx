'use client';

import { useEffect, useState } from 'react';

import { THEME_KEY } from '@/lib/theme';

type Theme = 'light' | 'dark';

function resolved(): Theme {
  const chosen = document.documentElement.dataset.theme;
  if (chosen === 'light' || chosen === 'dark') return chosen;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/**
 * Light or dark. The system decides until the viewer does; the choice is kept in
 * this browser only (a per-viewer convenience, so storage failing is harmless).
 */
export function ThemeToggle(): React.JSX.Element {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    setTheme(resolved());
  }, []);

  function toggle(): void {
    const next: Theme = (theme ?? resolved()) === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // Private window, blocked storage: the choice lasts this page view.
    }
    setTheme(next);
  }

  const label = theme === 'dark' ? 'Switch to the light theme' : 'Switch to the dark theme';
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={label}
      title={label}
      className="rounded p-1 text-ink-faint hover:bg-raised hover:text-ink"
    >
      <svg viewBox="0 0 16 16" className="h-4 w-4" aria-hidden="true">
        {theme === 'dark' ? (
          <g fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
            <circle cx="8" cy="8" r="3" />
            <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
          </g>
        ) : (
          <path
            d="M13.5 9.6A5.5 5.5 0 0 1 6.4 2.5a5.5 5.5 0 1 0 7.1 7.1Z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinejoin="round"
          />
        )}
      </svg>
    </button>
  );
}
