import { dirname } from 'path';
import { fileURLToPath } from 'url';

import { FlatCompat } from '@eslint/eslintrc';

// eslint-config-next 15 ships the classic shareable-config shape, so it is
// bridged into flat config with FlatCompat. (The scaffold assumed 16, whose
// flat-native entry points do not exist here.)
const compat = new FlatCompat({ baseDirectory: dirname(fileURLToPath(import.meta.url)) });

const config = [
  { ignores: ['.next/**', 'out/**', 'build/**', 'next-env.d.ts'] },
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  {
    rules: {
      // The project rule is no `any`, enforced rather than trusted.
      '@typescript-eslint/no-explicit-any': 'error',
    },
  },
];

export default config;
