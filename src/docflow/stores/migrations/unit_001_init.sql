-- The tables of one unit. {schema} is the unit's schema, {ns} the installation's.
-- A version, its passages and the history only ever grow: a trigger refuses to change or remove a row.
-- Where a document stands is not a column anywhere: it is read off the history.

CREATE TABLE {schema}.documents (
  doc_id     text PRIMARY KEY,
  workflow   text  NOT NULL,
  owner      text  NOT NULL,
  workspace  text  NOT NULL,
  invited    jsonb NOT NULL
);

CREATE TABLE {schema}.versions (
  doc_id     text    NOT NULL REFERENCES {schema}.documents (doc_id),
  version    integer NOT NULL CHECK (version >= 1),
  sha256     text    NOT NULL CHECK (length(sha256) = 64),
  size       bigint  NOT NULL,
  rel_path   text    NOT NULL CHECK (rel_path !~ '^/' AND rel_path !~ '[.][.]'),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (doc_id, version)
);

CREATE TABLE {schema}.chunks (
  doc_id     text    NOT NULL,
  version    integer NOT NULL,
  seq        integer NOT NULL,
  row_no     bigserial,
  pages      jsonb   NOT NULL,
  breadcrumb text    NOT NULL,
  content    text    NOT NULL,
  embedding  vector(1536),
  PRIMARY KEY (doc_id, version, seq),
  FOREIGN KEY (doc_id, version) REFERENCES {schema}.versions (doc_id, version)
);

CREATE TABLE {schema}.history (
  subject_id text    NOT NULL,
  seq        integer NOT NULL CHECK (seq >= 0),
  event      jsonb   NOT NULL,
  at         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (subject_id, seq)
);

CREATE TRIGGER history_append_only BEFORE UPDATE OR DELETE ON {schema}.history
  FOR EACH ROW EXECUTE FUNCTION {ns}.forbid_change();
CREATE TRIGGER history_no_truncate BEFORE TRUNCATE ON {schema}.history
  FOR EACH STATEMENT EXECUTE FUNCTION {ns}.forbid_change();
CREATE TRIGGER versions_append_only BEFORE UPDATE OR DELETE ON {schema}.versions
  FOR EACH ROW EXECUTE FUNCTION {ns}.forbid_change();
CREATE TRIGGER versions_no_truncate BEFORE TRUNCATE ON {schema}.versions
  FOR EACH STATEMENT EXECUTE FUNCTION {ns}.forbid_change();
CREATE TRIGGER chunks_append_only BEFORE UPDATE OR DELETE ON {schema}.chunks
  FOR EACH ROW EXECUTE FUNCTION {ns}.forbid_change();
CREATE TRIGGER chunks_no_truncate BEFORE TRUNCATE ON {schema}.chunks
  FOR EACH STATEMENT EXECUTE FUNCTION {ns}.forbid_change();
