'use client';

import { useEffect, useState } from 'react';

/**
 * The open agent and tab, in the page's URL: `?node=researcher:s1&tab=prompt`.
 *
 * So a link can point at one agent's prompt, and a reload keeps it open. The
 * run itself stays the path (`/runs/{id}`); this is only where you are in it.
 */

export const TABS = ['task', 'prompt', 'output', 'attempts'] as const;
export type InspectorTab = (typeof TABS)[number];

function isTab(value: string | null): value is InspectorTab {
  return TABS.some((tab) => tab === value);
}

export interface InspectorLink {
  readonly node: string | null;
  readonly tab: InspectorTab;
}

/** What a query string asks to open; an unknown tab is the first one. */
export function readLink(search: string): InspectorLink {
  const params = new URLSearchParams(search);
  const tab = params.get('tab');
  return { node: params.get('node'), tab: isTab(tab) ? tab : 'task' };
}

/** The URL with the link written in, every other parameter left as it was. */
export function writeLink(href: string, link: InspectorLink): string {
  const url = new URL(href);
  if (link.node === null) url.searchParams.delete('node');
  else url.searchParams.set('node', link.node);
  // The default tab is left out, so an agent's plain link stays short.
  if (link.node === null || link.tab === 'task') url.searchParams.delete('tab');
  else url.searchParams.set('tab', link.tab);
  return url.toString();
}

/**
 * The selection and tab, read from the URL once mounted and written back on
 * every change. Read in an effect, not in the initial state: the server renders
 * with no selection, and a first client render that differed would not hydrate.
 */
export function useInspectorLink(): readonly [
  InspectorLink,
  (node: string | null) => void,
  (tab: InspectorTab) => void,
] {
  const [link, setLink] = useState<InspectorLink>({ node: null, tab: 'task' });
  const [read, setRead] = useState(false);

  useEffect(() => {
    // Once, after hydration: a one-off sync from the URL, not derived state.
    const fromUrl = readLink(window.location.search);
    setLink(fromUrl);
    setRead(true);
  }, []);

  useEffect(() => {
    // Until the URL has been read, writing would erase what it asked for.
    if (!read) return;
    const next = writeLink(window.location.href, link);
    if (next !== window.location.href) window.history.replaceState(window.history.state, '', next);
  }, [link, read]);

  return [
    link,
    // Another agent keeps the open tab: reading two prompts is the usual reason to switch.
    (node) => setLink((current) => ({ ...current, node })),
    (tab) => setLink((current) => ({ ...current, tab })),
  ] as const;
}
