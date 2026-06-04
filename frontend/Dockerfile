# --- build stage ---
FROM node:22-alpine AS builder

WORKDIR /app

# Сначала зависимости (для кеша слоёв).
COPY package.json package-lock.json ./
RUN npm ci

# Затем исходники и сборка (vite build + esbuild server.cjs).
COPY . .
RUN npm run build

# --- runtime stage ---
FROM node:22-alpine

WORKDIR /app
ENV NODE_ENV=production

# node_modules нужны в рантайме: server.cjs собран с --packages=external
# (express, http-proxy-middleware, vite резолвятся из node_modules).
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000

CMD ["node", "dist/server.cjs"]
