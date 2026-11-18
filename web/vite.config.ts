import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';
import { VitePWA } from 'vite-plugin-pwa';
import { readFileSync } from 'node:fs';

const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8'));
const model = JSON.parse(readFileSync(new URL('./public/models/model.json', import.meta.url), 'utf8'));

export default defineConfig({
  // GitHub Pages serves the app from /<repo>/; override with BASE=/ for other hosts
  base: process.env.BASE ?? '/count-it-easy/',
  define: { __APP_VERSION__: JSON.stringify(pkg.version), __MODEL_VERSION__: JSON.stringify(model.version) },
  plugins: [
    preact(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon-192.png', 'apple-touch-icon.png'],
      manifest: {
        name: 'Count it Easy!',
        short_name: 'Count it Easy',
        description: 'Punkte zählen für Take it Easy! – per Kamera, offline, auf dem Gerät.',
        lang: 'de',
        theme_color: '#121212',
        background_color: '#121212',
        display: 'standalone',
        orientation: 'portrait',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,png,svg,onnx,wasm,mjs,json}'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
      },
    }),
  ],
  build: { target: 'es2022' },
  test: { environment: 'node' },
} as never);
