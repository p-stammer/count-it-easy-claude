import { render } from 'preact';
import { registerSW } from 'virtual:pwa-register';
import { App } from './app';
import { loadModel } from './core/model';
import './styles.css';

render(<App />, document.getElementById('app')!);
registerSW({ immediate: true });
// warm up the model in the background so the first count is fast
setTimeout(() => loadModel().catch(() => undefined), 500);
if (navigator.storage?.persist) navigator.storage.persist().catch(() => undefined);
