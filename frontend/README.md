# MLMatcher Frontend

React-SPA (Vite) для сервиса ML-матчинга строк. Две вкладки: **Обучение** (Шаг 1) и **Тестирование** (Шаг 2).

## Стек

React 19 · TypeScript · Vite · Tailwind CSS · lucide-react · motion · jszip · Express (node-сервер + прокси).

## Роль node-сервера (`server.ts`)

`server.ts` — лёгкий Express-сервер, который:
- в **dev** отдаёт приложение через Vite-middleware (HMR);
- в **prod** отдаёт собранную статику из `dist/`;
- в обоих режимах **проксирует все запросы `/api/*` на backend** (`http-proxy-middleware`),
  корректно пробрасывая multipart-загрузки, SSE-поток и бинарный download.

Благодаря прокси UI-компоненты обращаются к относительным `/api/v1/*` и не знают адреса бэкенда —
он задаётся переменной `BACKEND_URL`.

## Структура

```
frontend/
├── server.ts                 # Express: статика/Vite + прокси /api → backend
├── vite.config.ts
├── index.html
└── src/
    ├── App.tsx               # корневой компонент, переключение вкладок
    ├── types.ts              # FeatureId, статусы, DTO
    ├── components/
    │   ├── Header.tsx
    │   ├── StepTraining.tsx  # Шаг 1: CSV + выбор фич + SSE-лог + скачивание
    │   ├── StepInference.tsx # Шаг 2: загрузка ZIP + две строки + результат
    │   └── Documentation.tsx # справка по API
    └── utils/metrics.ts      # локальный расчёт фич (demo-режим)
```

## Запуск

### Через Docker (рекомендуется)
Из корня проекта: `docker compose up --build` → http://localhost:3000.

### Локально (dev)
```bash
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm run dev
```
Откройте http://localhost:3000 (предварительно подняв backend на `:8000`).

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `BACKEND_URL` | `http://localhost:8000` | Адрес FastAPI-бэкенда для прокси |
| `PORT` | `3000` | Порт node-сервера |
| `NODE_ENV` | — | `production` → отдаёт `dist/` вместо Vite-middleware |

## Поток UI

- **Шаг 1.** Загрузка CSV → выбор фич (чекбоксы) → «Обучить» (`POST /training/start`) →
  подписка на SSE (`EventSource` на `/training/status/{id}`) с живым логом и прогресс-баром →
  «Скачать модель» (`/training/download/{id}`).
- **Шаг 2.** Загрузка ZIP → ввод двух строк → «Сравнить» (`POST /inference/predict`) →
  вывод вероятности совпадения и значений фич.

> В demo-режиме (без реальной модели) фронт считает метрики локально через `src/utils/metrics.ts`.
> При наличии реального ZIP инференс идёт на backend.
