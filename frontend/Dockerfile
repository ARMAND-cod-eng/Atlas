# Frontend image. Built by `docker compose --profile app up --build`.
#
# Next's standalone output: the runtime stage carries the compiled server and
# only the node_modules it actually imports, not the whole dependency tree.

# --- dependencies -------------------------------------------------------------
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# --- build --------------------------------------------------------------------
FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Baked in at build time — NEXT_PUBLIC_ values are inlined into the browser
# bundle. The browser calls the API directly, so this is the address as seen
# from the user's machine (the published port), not the compose service name.
# 127.0.0.1 rather than localhost: see the note in src/lib/useRunStream.ts.
ARG NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# --- runtime ------------------------------------------------------------------
FROM node:22-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Non-root, as in the backend image.
RUN addgroup -S atlas && adduser -S atlas -G atlas

COPY --from=build --chown=atlas:atlas /app/.next/standalone ./
COPY --from=build --chown=atlas:atlas /app/.next/static ./.next/static
# No public/ directory: the app serves no static files of its own. Add a
# `COPY ... /app/public ./public` here if it ever does.

USER atlas
EXPOSE 3000
CMD ["node", "server.js"]
