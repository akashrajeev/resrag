import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'resrag-browser-document-fix',
      transform(code, id) {
        if (id.endsWith('/src/main.jsx')) {
          return code.replace('document.documentElement.dataset.theme', 'window.document.documentElement.dataset.theme');
        }
        return null;
      },
    },
  ],
  server: {
    port: 5173,
  },
  preview: {
    port: 5173,
  },
});
