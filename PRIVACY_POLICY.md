# Sightline Privacy Policy

**Last updated:** 2026-07-01
**Operator:** Antoni Baum (Yard1)

Sightline is a moderation bot that helps Discord servers detect scam images posted by compromised accounts. This policy explains what data the bot handles and how to have it removed.

**Sightline stores your server's data inside your own Discord server, not on our systems.** Everything the bot keeps — its configuration and its database of known scam images — lives in a private channel that your server's administrators create and control. We run no external database, keep no user profiles, and never use data for advertising, analytics, or anything beyond scam detection.

## 1. What data does Sightline collect?

**Scanned in real time, then discarded.** To do its job, Sightline uses Discord's Message Content intent to check messages for images. Images are compared against the server's list of known scam images and then discarded — they are not saved unless a moderator marks them as a scam. Message text is not read or stored.

**Stored inside your server.** When a moderator marks an image as a scam (or the server has enabled automatic collection of matches), Sightline saves a record to the server's private database channel. That record contains the scam image, where it was posted, the user ID of the account that posted it, and the moderator who added it.

**Moderation logs.** When a scam image is detected, Sightline posts a log for your moderators — in your server's own log channel — identifying the flagged account (mention, user ID, username), the image, and the actions taken.

## 2. Why does Sightline need this data?

Reading images is the entire purpose of the bot: recognizing scam images even after they've been resized or edited. User IDs in records and logs let your moderators see which account posted a scam, review the evidence, and take or reverse action. Nothing is collected beyond what detection and moderation require.

## 3. How is the data used?

Only to detect scam images in servers where the bot is installed, apply the moderation actions that server's admins have configured, and keep logs for that server's moderators. Each server's scam database is separate — data from one server is never used in another.

## 4. Who is the data shared with?

No one, with a single optional exception. If your server's admins turn on the optional text-verification feature, a small cropped piece of a suspected scam image is sent to the OCR.space text-reading service to confirm the scam wording. No usernames, IDs, or messages are included, and nothing is kept afterward — on our side or theirs: per the [OCR.space privacy policy](https://ocr.space/privacypolicy), uploaded images are deleted after processing. If the feature is off (the default), no data leaves Discord at all.

We never sell data or share it with any other party.

## 5. How can users contact us?

For privacy questions or concerns:

- antoni.baum@protonmail.com

## 6. How can data be removed?

Because everything is stored in your own server, your administrators can remove it directly, instantly, and without asking us:

- **One record:** delete its message in the bot's database channel.
- **Everything:** delete the database channel and remove the bot.
- **Logs:** delete the log messages like any other Discord message.

If you're an individual user (for example, you've recovered a hacked account) and believe a server's records include you, ask that server's moderators — the data is theirs to manage.

## 7. Changes

We may update this policy as the bot evolves. The current version is always available at [URL].