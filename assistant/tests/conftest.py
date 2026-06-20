"""Redirect the memory store to a throwaway dir BEFORE config/store import,
so tests never touch the real ./assistant/memory/memory_db."""
import os
import tempfile

os.environ.setdefault("MEMORY_DB_PATH", tempfile.mkdtemp(prefix="kara-test-mem-"))
# Point local-scope memory at a non-existent temp path so tests don't pick up a real
# .kara_memory in the working directory.
os.environ.setdefault("LOCAL_MEMORY_DB_PATH",
                      os.path.join(tempfile.mkdtemp(prefix="kara-test-local-"), "mem"))
