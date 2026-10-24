/**
 * Node sizes, as React Flow measured them, kept across re-renders.
 *
 * GraphCanvas rebuilds its nodes from the run state on every event, and React
 * Flow keeps a node hidden until it knows the node's size. A rebuilt node object
 * carries no size, so each rebuild hid the graph again until something resized
 * a node; a run that had stopped changing stayed invisible. The sizes React Flow
 * reports are kept here and put back on every rebuilt node.
 */

export interface Size {
  readonly width: number;
  readonly height: number;
}

export type Measurements = ReadonlyMap<string, Size>;

/** The changes GraphCanvas receives that matter here; others are ignored. */
export interface DimensionChange {
  readonly type: string;
  readonly id: string;
  readonly dimensions?: Size;
}

/**
 * The measurements after these changes. The same map back when nothing changed:
 * a new one each time would re-render on every report, and React Flow reports
 * on every render.
 */
export function recordMeasurements(
  current: Measurements,
  changes: readonly DimensionChange[],
): Measurements {
  let next: Map<string, Size> | null = null;
  for (const change of changes) {
    if (change.type !== 'dimensions' || change.dimensions === undefined) continue;
    const known = (next ?? current).get(change.id);
    if (known?.width === change.dimensions.width && known.height === change.dimensions.height) {
      continue;
    }
    next ??= new Map(current);
    next.set(change.id, { width: change.dimensions.width, height: change.dimensions.height });
  }
  return next ?? current;
}
