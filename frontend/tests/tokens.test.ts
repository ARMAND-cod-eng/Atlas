/**
 * Every colour used for text is readable on every surface it sits on, in both
 * themes: WCAG AA for normal text, a contrast of at least 4.5:1.
 *
 * Read from globals.css itself, so a token edit that makes a label unreadable
 * fails here rather than in someone's eyes.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const CSS = readFileSync(join(__dirname, '../src/app/globals.css'), 'utf-8');

function tokens(selector: string): Record<string, string> {
  const start = CSS.indexOf(selector);
  const block = CSS.slice(start, CSS.indexOf('}', start));
  return Object.fromEntries([...block.matchAll(/--([a-z-]+):\s*(#[0-9a-f]{6})/g)].map((m) => [m[1], m[2]]));
}

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const channel = Number.parseInt(hex.slice(i, i + 2), 16) / 255;
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * (r ?? 0) + 0.7152 * (g ?? 0) + 0.0722 * (b ?? 0);
}

export function contrast(a: string, b: string): number {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return ((light ?? 0) + 0.05) / ((dark ?? 0) + 0.05);
}

// The tokens text is drawn in. `idle` and `line` are only ever dots and rules.
const TEXT = ['ink', 'ink-muted', 'ink-faint', 'accent', 'running', 'done', 'failed', 'retry', 'edited', 'law', 'guidance', 'secondary'];
const BACKGROUNDS = ['surface', 'canvas', 'raised'];

describe.each([
  ['light', tokens(':root {')],
  ['dark', tokens(':root[data-theme="dark"] {')],
])('the %s theme', (_, theme) => {
  it.each(TEXT)('%s is readable on every surface', (text) => {
    const below = BACKGROUNDS.map((bg) => [bg, contrast(theme[text] ?? '', theme[bg] ?? '')] as const).filter(
      ([, ratio]) => ratio < 4.5,
    );

    expect(below.map(([bg, ratio]) => `${bg} ${ratio.toFixed(2)}`)).toEqual([]);
  });

  it('puts readable text on the accent', () => {
    expect(contrast(theme['accent-ink'] ?? '', theme.accent ?? '')).toBeGreaterThanOrEqual(4.5);
  });
});

it('the dark theme is the same in both places it is written', () => {
  // Once for a system preference, once for the viewer's choice.
  const system = CSS.slice(CSS.indexOf('@media (prefers-color-scheme: dark)'));
  expect(tokens(':root:not([data-theme="light"]) {')).toEqual(tokens(':root[data-theme="dark"] {'));
  expect(system.length).toBeGreaterThan(0);
});
