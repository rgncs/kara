---
name: email
description: Read, search, send, delete, or unsubscribe from email (Gmail).
triggers: \bemail(s)?\b, \binbox\b, \bunread\b, gmail, \bunsubscribe\b, check my mail, any (new )?(mail|messages), reply to, send (an? )?(email|message), delete (this|that|the) email
---
I may have MULTIPLE Gmail accounts connected (e.g. work, personal). For a general
question ("any new email?", "summarize my unread"), search ALL of them — call
`list_messages` with no `account` (results are tagged and ids come back as
'account:id'; pass that whole tag to read/reply/trash). Only pass `account` when I name
one. To send, send FROM the relevant account (the one I'm replying from, or that I name);
the confirmation will say which account.

When I ask about email, use the Gmail tools:

1. To check or search — `list_messages` (Gmail search syntax in `query`, e.g.
   `from:kevin newer_than:7d`, `is:unread`, `subject:invoice`). Summarize the
   results in plain language; keep the message ids so you can act on one next.
   Use `read_message` to open a specific one before summarizing or replying.
2. To send or reply — `send_message` (to, subject, body). For a reply, read the
   original first so you quote/address it correctly and use a "Re:" subject. The
   tool confirms with me before anything is sent — that's expected.
3. To delete — `trash_message` (moves to Trash, recoverable). Confirm which
   message if more than one could match; it asks me before trashing.
4. To unsubscribe — `unsubscribe` on the message from that sender; it uses the
   email's List-Unsubscribe link and asks me first.
5. Never send, delete, or unsubscribe without the confirmation, and never invent
   message contents — read the real message when you need its details. For bulk
   actions ("delete all from X", "unsubscribe from these"), find them with
   `list_messages`, then act on each id; each one is confirmed separately.
