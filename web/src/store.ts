import type { Recognition } from './core/recognize';

/** What the current count is for (set before opening the camera). */
export interface CountContext { playerId?: string; sessionId?: string; round?: number }

/** In-memory hand-over between camera and result screen. */
export const current: {
  context: CountContext;
  work: HTMLCanvasElement | null;
  recognition: Recognition | null;
} = { context: {}, work: null, recognition: null };

export function startCount(ctx: CountContext) {
  current.context = ctx;
  current.work = null;
  current.recognition = null;
}
