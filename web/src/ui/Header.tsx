import type { ComponentChildren } from 'preact';
import { back } from '../router';

export function Header({ title, right }: { title: ComponentChildren; right?: ComponentChildren }) {
  return (
    <header class="header">
      <button class="icon-btn" onClick={back} aria-label="Zurück">←</button>
      <div class="header-title">{title}</div>
      <div class="header-right">{right}</div>
    </header>
  );
}
