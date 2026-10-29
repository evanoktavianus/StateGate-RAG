-- One-time setup of PostgreSQL for docflow: a login role `docflow_app` and a database `docflow`, with pgvector enabled in it.
-- Run once, as a PostgreSQL superuser. Safe to run again: it creates only what is missing and changes nothing else.
--
-- It touches nothing else on the server: no other database, role or setting, and no pg_hba.conf.
--
-- The role's password never appears in this file or on a command line. Make a random password, keep it in a file only you can
-- read, and give this script a SCRAM verifier of it in the environment:
--
--     umask 077
--     python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/.docflow_password
--     export DOCFLOW_APP_VERIFIER="$(DOCFLOW_APP_PASSWORD="$(cat ~/.docflow_password)" python3 scripts/pg_password_verifier.py)"
--     psql -X -v ON_ERROR_STOP=1 -f src/docflow/stores/setup_postgres.sql          # as a superuser, e.g. runuser -u postgres --
--
-- The application then reads DATABASE_URL=postgresql://docflow_app:<password>@localhost:5432/docflow

\set ON_ERROR_STOP on
\getenv verifier DOCFLOW_APP_VERIFIER
\if :{?verifier}
\else
  \echo 'DOCFLOW_APP_VERIFIER is not set: nothing was done.'
  \quit
\endif

-- the role: created only if it does not exist (an existing one, and its password, are left alone)
SELECT format('CREATE ROLE docflow_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD %L', :'verifier')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'docflow_app')
\gexec

-- the database: created only if it does not exist
SELECT 'CREATE DATABASE docflow OWNER docflow_app'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'docflow')
\gexec

-- pgvector, inside the new database only
\connect docflow
CREATE EXTENSION IF NOT EXISTS vector;

\echo 'docflow_app and the docflow database are in place.'
