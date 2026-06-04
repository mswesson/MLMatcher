/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { createServer as createViteServer } from 'vite';

const app = express();
const PORT = Number(process.env.PORT) || 3000;

// Адрес реального FastAPI-бэкенда (в docker-compose — http://backend:8000).
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Проксируем все /api/* запросы на бэкенд.
// pathFilter (а не монтирование на '/api') сохраняет полный путь /api/v1/...
// Регистрируем ДО любых body-парсеров, чтобы не «съесть» multipart/SSE-потоки.
app.use(
  createProxyMiddleware({
    pathFilter: '/api',
    target: BACKEND_URL,
    changeOrigin: true,
    // SSE: отключаем буферизацию, чтобы события шли в реальном времени.
    on: {
      proxyReq: (proxyReq) => {
        proxyReq.setHeader('X-Accel-Buffering', 'no');
      },
    },
  }),
);

// Serve Vite dev server or production assets
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Frontend server running on http://0.0.0.0:${PORT} (proxying /api -> ${BACKEND_URL})`);
  });
}

startServer();
