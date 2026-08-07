-- Runs once, on first initialization of the postgres data volume.
-- Mounted read-only into /docker-entrypoint-initdb.d/ by docker-compose.yml.
--
-- pgvector ships with the pgvector/pgvector image but is not enabled by default;
-- the extension has to be created per-database. Phase 0 only enables it — the
-- embedding tables and indexes arrive with the retrieval layer in a later phase.

CREATE EXTENSION IF NOT EXISTS vector;
