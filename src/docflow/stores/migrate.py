"""Applies the SQL files in `migrations/`, in order, once each.

    global_NNN_*.sql   the installation's tables, in the namespace schema
    unit_NNN_*.sql     the tables of one unit, in the unit's schema

Plain SQL, no ORM. `{ns}` and `{schema}` in a file are replaced by quoted schema names. What was applied is kept in
`<ns>.schema_migrations`, so running it again changes nothing.

    DATABASE_URL=... python -m docflow.stores.migrate [namespace]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MIGRATIONS = Path(__file__).parent / "migrations"


def _files(prefix: str) -> list[Path]:
    return sorted(MIGRATIONS.glob(f"{prefix}_*.sql"))


def _apply(conn, scope: str, prefix: str, ns: str, schema: str) -> list[str]:
    from psycopg import sql

    done = []
    for path in _files(prefix):
        with conn.transaction():
            if conn.execute(sql.SQL("SELECT 1 FROM {}.schema_migrations WHERE scope = %s AND version = %s").format(sql.Identifier(ns)), (scope, path.name)).fetchone():
                continue
            conn.execute(sql.SQL(path.read_text(encoding="utf-8")).format(ns=sql.Identifier(ns), schema=sql.Identifier(schema)))
            conn.execute(sql.SQL("INSERT INTO {}.schema_migrations (scope, version) VALUES (%s, %s)").format(sql.Identifier(ns)), (scope, path.name))
            done.append(path.name)
    return done


def apply_global(conn, ns: str) -> list[str]:
    from psycopg import sql

    with conn.transaction():
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(ns)))
        conn.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {}.schema_migrations (scope text NOT NULL, version text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (scope, version))").format(sql.Identifier(ns)))
    return _apply(conn, "global", "global", ns, ns)


def apply_unit(conn, ns: str, schema: str) -> list[str]:
    return _apply(conn, schema, "unit", ns, schema)


if __name__ == "__main__":
    import psycopg

    namespace = sys.argv[1] if len(sys.argv) > 1 else "docflow"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        print("applied:", apply_global(connection, namespace) or "nothing new")
