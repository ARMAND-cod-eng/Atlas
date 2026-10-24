/**
 * Where the viewer's theme choice is kept. A plain module, not a client one: the
 * server-rendered layout reads it too, for the script that applies the choice
 * before the first paint.
 */
export const THEME_KEY = 'atlas-theme';
