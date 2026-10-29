-- Global tables of one docflow installation (one schema, the namespace). {ns} is that schema.
-- The people store, the workflow definitions, and the map from a document to its unit.

CREATE FUNCTION {ns}.forbid_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'docflow: % on %.% is not allowed: the table is append-only', TG_OP, TG_TABLE_SCHEMA, TG_TABLE_NAME
    USING ERRCODE = 'integrity_constraint_violation';
END
$$;

CREATE TABLE {ns}.orgs (
  org_id  text PRIMARY KEY,
  ordinal bigserial
);

CREATE TABLE {ns}.units (
  unit_id     text PRIMARY KEY,
  schema_name text NOT NULL UNIQUE,
  dir_name    text NOT NULL UNIQUE,
  org         text,
  ordinal     bigserial
);

CREATE TABLE {ns}.people (
  user_id   text PRIMARY KEY,
  ordinal   bigserial,
  org       text,
  unit      text REFERENCES {ns}.units (unit_id),
  groups    jsonb   NOT NULL,
  org_admin boolean NOT NULL,
  positions jsonb   NOT NULL,
  attrs     jsonb   NOT NULL
);

CREATE TABLE {ns}.workspaces (
  ws_id       text PRIMARY KEY,
  org         text,
  members     jsonb NOT NULL,
  group_roles jsonb NOT NULL
);

CREATE TABLE {ns}.workflows (
  name      text PRIMARY KEY,
  yaml      text NOT NULL,
  sha256    text NOT NULL CHECK (length(sha256) = 64),
  loaded_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE {ns}.document_units (
  doc_id  text PRIMARY KEY,
  unit    text NOT NULL REFERENCES {ns}.units (unit_id),
  ordinal bigserial
);


