import { defineConfig } from 'vitest/config';

export default defineConfig({
  // tsconfig says `jsx: preserve` because Next compiles JSX itself; under
  // Vitest nothing else would, so component tests compile it here.
  oxc: { jsx: { runtime: 'automatic' } },
  // The `@/` imports components use, read from tsconfig's paths.
  resolve: { tsconfigPaths: true },
});
