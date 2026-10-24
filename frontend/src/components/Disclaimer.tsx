/**
 * The product's boundary, said wherever an answer is asked for or shown.
 *
 * The law's text is the Official Journal's as retrieved on the date below
 * (backend/atlas/library/sources/manifest.json); a later amendment is not in it.
 */
export const LAW_AS_OF = '23 October 2026';

export function Disclaimer({ className = '' }: { readonly className?: string }): React.JSX.Element {
  return (
    <p className={`text-2xs leading-relaxed text-ink-faint ${className}`}>
      A research aid, not legal advice. The AI Act (Regulation (EU) 2024/1689) and the GDPR
      (Regulation (EU) 2016/679) are read from EUR-Lex as published in the Official Journal,
      retrieved {LAW_AS_OF}; only the Official Journal&apos;s version is authentic, and later
      amendments are not included.
    </p>
  );
}
