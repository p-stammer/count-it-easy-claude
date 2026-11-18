import { useEffect, useState } from 'preact/hooks';

/** Minimal hash router – keeps the Android/iOS back gesture working. */
export function useRoute(): string[] {
  const read = () => (location.hash.replace(/^#\/?/, '') || 'home').split('/');
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => { setRoute(read()); window.scrollTo(0, 0); };
    addEventListener('hashchange', on);
    return () => removeEventListener('hashchange', on);
  }, []);
  return route;
}

export const go = (path: string, replace = false) => {
  const h = `#/${path}`;
  if (replace) history.replaceState(null, '', h), dispatchEvent(new HashChangeEvent('hashchange'));
  else location.hash = h;
};
export const back = () => (history.length > 1 ? history.back() : go('home', true));

/** Go back `steps` history entries, then navigate to `path` (drops e.g. camera + result). */
export function backThenGo(steps: number, path: string) {
  if (history.length <= steps) { go(path, true); return; }
  const on = () => { removeEventListener('popstate', on); go(path, true); };
  addEventListener('popstate', on);
  history.go(-steps);
}
