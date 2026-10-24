'use client';

import { useSyncExternalStore } from 'react';

/**
 * Whether a media query matches, kept in sync as the window changes.
 *
 * `useSyncExternalStore` rather than state set in an effect: the browser owns
 * this value, and reading it this way cannot tear between renders. On the server
 * there is no window, so `serverDefault` stands in until hydration.
 */
export function useMediaQuery(query: string, serverDefault: boolean): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const list = window.matchMedia(query);
      list.addEventListener('change', onChange);
      return () => list.removeEventListener('change', onChange);
    },
    () => window.matchMedia(query).matches,
    () => serverDefault,
  );
}
