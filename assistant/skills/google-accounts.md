---
name: google-accounts
description: List connected Google accounts, or connect a new Gmail/Calendar account.
triggers: (which|what) (google|gmail|email|calendar)? ?accounts?, are you connected to, connect (a|my|another|the)?\s*(personal|work|new)?\s*(google|gmail|email|calendar)( account)?, add (a|my|another)?\s*(personal|work)?\s*(google|gmail|email)( account)?, link (my|a)?\s*(gmail|google), set up (my|another)\s*(email|gmail|google|account), how (do i|to) connect, connect an account, disconnect
---
When I ask what accounts you're connected to, or to add/connect one:

1. **What's connected** — call `list_google_accounts` and tell me the labels and email
   addresses (and which is primary). Don't guess; use the tool.
2. **Connect a new account** — this opens a browser I have to complete, so:
   - If I didn't give a label, ask for a short one (e.g. "personal", "work").
   - Tell me what's about to happen: "I'll open a browser — sign in with that Google
     account and approve the Gmail + Calendar access." Then CONFIRM I'm ready.
   - Only once I say go, call `connect_google_account` with the label. It opens the
     browser and waits for me to finish, then reports the connected address.
3. If it returns first-time-setup steps (no credentials.json yet), relay those steps to
   me clearly and stop — I have to do that part before any account can connect. Make these
   points clear when I ask how to connect a personal vs a business account:
   - credentials.json is set up ONCE and then connects BOTH personal and business
     accounts — it is not per-account, so there's no separate "personal" vs "business"
     setup.
   - The only difference is at sign-in: a PERSONAL account connects directly; a
     BUSINESS/Workspace account may be blocked by its IT admin (they control third-party
     app access), so it might need admin approval first.
4. After a successful connection, mention it's now included in my email and calendar.
