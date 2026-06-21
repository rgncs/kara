---
name: spam-cleanup
description: Review senders with many unread emails and delete/unsubscribe by number.
triggers: spam cleanup, deep (email )?(spam )?clean, clean (up )?(my )?(in)?box, clean my email, spam clean, review spam, too many emails from, junk mail, declutter (my )?(in)?box, unsubscribe from
---
When I ask to clean up spam / declutter my inbox:

1. Pick the scan:
   - If I say a **DEEP** cleanup (or "scan everything / whole history"), call
     `deep_spam_cleanup`. It scans my entire history, auto-trashes everyone already
     on my confirmed auto-delete list, and returns the NEW candidates numbered.
   - Otherwise call `find_spam_candidates` (add full=true if I ask to "go deeper").
2. Show me the result as a **numbered list** exactly as the tool returns it (sender,
   unread count, and whether it can unsubscribe). Don't reorder or renumber it.
3. Then I'll respond by NUMBER. Map each number to its sender from the list you just
   showed and act:
   - **"always delete from 1, 3"** → call `auto_delete_sender` for senders #1 and #3
     (adds them to the auto-delete list AND trashes their unread now; future unread is
     auto-trashed with no prompts).
   - **"delete 2"** → `trash_from_sender` for #2 (trash its unread once; confirms).
   - **"unsubscribe 4"** → `unsubscribe` on a message from #4.
   - **"keep 5"** → `keep_sender` for #5 (never flag it again).
   Handle several numbers/commands in one reply; act on each, then briefly confirm
   what you did.
4. Always ignore senders ending in **@regenics.com** (and its subdomains) — they're on
   the keep-list; never flag, delete, or unsubscribe those.
5. Only unread mail is ever deleted, always to Trash (recoverable). Never invent
   message contents. If a number doesn't match the list, ask me to clarify rather than
   guessing.
