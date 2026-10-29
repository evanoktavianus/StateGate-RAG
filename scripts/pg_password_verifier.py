"""Prints the SCRAM-SHA-256 verifier PostgreSQL stores for a password, so a role can be created without the password
itself reaching an SQL statement, a log or a command line. The password is read from the environment variable
DOCFLOW_APP_PASSWORD (never from the command line) and is not printed. See src/docflow/stores/setup_postgres.sql."""

import base64, hashlib, hmac, os, secrets, sys

password = os.environ.get("DOCFLOW_APP_PASSWORD")
if not password:
    sys.exit("DOCFLOW_APP_PASSWORD is not set")
iterations, salt = 4096, secrets.token_bytes(16)
salted = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
b64 = lambda b: base64.b64encode(b).decode()
print(f"SCRAM-SHA-256${iterations}:{b64(salt)}${b64(hashlib.sha256(client_key).digest())}:{b64(server_key)}")
