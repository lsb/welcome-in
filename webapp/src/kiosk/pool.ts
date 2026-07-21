// The card pool: pre-generated question cards, served instantly — port of
// pool.py's serve path. The cards are baked into the bundle at build time
// (scripts/prepare_assets.mjs reads ../pool/*.txt, the same files the desktop
// 27B producer writes). View counts are in-memory, reset per load, exactly as
// the desktop resets them per run.

export interface PoolCard {
  topic: string;
  text: string;
  model: string;
}

export class CardPool {
  private cards: PoolCard[];
  private views: number[];

  constructor(cards: PoolCard[]) {
    this.cards = cards;
    this.views = cards.map(() => 0);
  }

  get size(): number {
    return this.cards.length;
  }

  /** The globally least-viewed card (random tie-break); bumps its view count
   * so repeat visitors get fresh cards. Null only if the pool is empty. */
  takeLeastViewed(): PoolCard | null {
    if (this.cards.length === 0) return null;
    const lo = Math.min(...this.views);
    const ties = this.views.flatMap((v, i) => (v === lo ? [i] : []));
    const pick = ties[Math.floor(Math.random() * ties.length)];
    this.views[pick] += 1;
    return this.cards[pick];
  }
}
