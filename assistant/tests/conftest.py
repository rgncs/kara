"""Redirect the memory store to a throwaway dir BEFORE config/store import,
so tests never touch the real ./assistant/memory/memory_db."""
import os
import tempfile

os.environ.setdefault("MEMORY_DB_PATH", tempfile.mkdtemp(prefix="kara-test-mem-"))
# Point local-scope memory at a non-existent temp path so tests don't pick up a real
# .kara_memory in the working directory.
os.environ.setdefault("LOCAL_MEMORY_DB_PATH",
                      os.path.join(tempfile.mkdtemp(prefix="kara-test-local-"), "mem"))
# Isolate Kara's own state files (self-facts, spam lists) to a throwaway dir so tests
# never read or write the real user state (e.g. a set birthday or keep-list).
_state = tempfile.mkdtemp(prefix="kara-test-state-")
for _var, _name in (("SELF_FACTS_PATH", "kara_self.json"),
                    ("SPAM_KEEP_PATH", "spam_keep.json"),
                    ("SPAM_AUTODELETE_PATH", "spam_autodelete.json"),
                    ("SPAM_LOG_PATH", "spam_candidates.json"),
                    ("SPAM_SCAN_STATE_PATH", "spam_scan_state.json")):
    os.environ.setdefault(_var, os.path.join(_state, _name))
