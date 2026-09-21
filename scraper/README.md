# OpenRice scraper notes

`scrape_reviews.py` pulls restaurant reviews from OpenRice into the same
DB the FastAPI app uses (`app/local.db` locally). This file documents the
non-obvious stuff discovered while building it - the kind of thing that
isn't visible just from reading the code.

## How to run it

```bash
conda activate gist
python scraper/scrape_reviews.py
```

Run it manually, whenever you're at the machine. This is **not** a
scheduled/cron job on purpose - see "Why manual, not scheduled" below.

## Site behavior discovered by hand, not guessed

- **Reviews are server-rendered**, not fetched via a separate API call -
  there's no JSON endpoint to reverse-engineer.
- The main restaurant URL only ever server-renders **3 teaser reviews**,
  with a link to a dedicated `/reviews` sub-page for the rest.
- **A hard page load straight to `/reviews` caps out at 15 reviews, full
  stop** - scrolling never loads more. Reaching the rest requires landing
  on `/reviews` via an in-app click (`a.show-more-reviews`) from the
  overview page, not a direct URL load. This is why `Venue.source_url`
  should point at the overview page, and the scraper does the click
  itself - confirmed by testing both paths side by side.
- The site shows a live **"食評 (N)"** label with its own advertised
  review count. It drifts upward over time as real users post (watched it
  go from 86 to 87 during development) - never hardcode it, always
  re-read it each run (`read_total_review_count`).

## Confirmed selectors (verified against real markup, not guessed)

| Field | Selector |
|---|---|
| Review card | `.review-post-desktop.poi-detail-review` (an `<article>`, not a `<div>` - don't assume the tag) |
| Reviewer name | `.info-top a` |
| Level / date / views | the three `.with-dot` divs inside `.info-bottom`, in that order |
| Rating | count `.poi-detail-rating-star-full-y` (1.0 each) + `.poi-detail-rating-star-half-y` (0.5 each), out of 5 - there's no plain numeric field |
| Review text | `.review-post-extract` - **this is a truncated excerpt only** ("…查看更多"). Full text would need a separate visit to each review's permalink; deliberately not done, to keep requests-per-session low |

## The scroll-mechanics bug (the big one)

Early versions scrolled in a few giant jumps (`page.mouse.wheel(0, 4000)`)
with multi-second pauses between them. This reliably **stalled around
30 cards no matter how long we waited** - looked exactly like a
server-side session throttle, and was misdiagnosed as one for a while.

It wasn't. Manually scrolling the page with a real mouse loaded all ~87
reviews in about 20 seconds. The actual cause: real mouse-wheel scrolling
fires many small, frequent scroll events, and the site's lazy-load
apparently needs something closer to that pattern, not one huge
instantaneous jump. Switching to small, frequent ticks
(`SCROLL_TICK_PX = 350`, every `SCROLL_TICK_PAUSE_SECONDS = 0.15`) fixed
it completely - confirmed growth of 15 → 30 → 45 → 60 → 75 → 87 in one
continuous run.

**Lesson**: if a future site's infinite scroll seems to mysteriously cap
out, try matching real scroll mechanics (many small ticks) before
assuming it's a deliberate server-side limit.

## Completion logic (`scrape_status`)

`Venue.scrape_status` (`pending` / `in_progress` / `done` /
`blocked_captcha`) is what `main()` uses to decide whether to pick up a
restaurant again. Getting "done" right took two attempts:

1. First attempt: mark `done` when a run finds cards but zero new ones.
   **This was wrong** - a run finding nothing new doesn't reliably mean
   nothing more exists; it can also mean this particular session's
   scrolling stalled early (see above). This caused a venue to be marked
   `done` at 75/86 reviews.
2. Current approach: compare the actual saved count against the site's
   live-read `total_reviews_expected`. Only mark `done` when our count
   genuinely reaches that total. If we can't read the total this run, or
   haven't reached it yet, stay `pending` - safer to keep retrying than to
   guess completion without real evidence.

## CAPTCHA handling

OpenRice serves a slide-puzzle CAPTCHA ("Verify to continue") to automated
traffic after enough requests, and blocks headless Chromium outright with
a hard connection failure. There is no legitimate bypass for this without
a human, and none is attempted here - `looks_like_captcha()` just detects
the CAPTCHA text and the script **stops immediately** (sets
`blocked_captcha`, exits) rather than retrying or waiting.

## Why manual, not scheduled

Given the above, a cron/GitHub-Actions scheduled job would routinely hit
the CAPTCHA with nobody around to clear it. Instead:
- Run manually, whenever convenient.
- `launch_persistent_context` keeps a real browser profile on disk so
  cookies carry over between sessions instead of looking like a fresh
  visitor every time.
- Each new review commits immediately (not buffered to the end), so an
  interrupted or CAPTCHA'd run never loses progress.
- `MAX_NEW_REVIEWS_PER_RUN` caps each run regardless of outcome, so a
  session can't turn into a long unattended crawl even by accident.

## Known limitations

- Review text is the truncated listing excerpt, not the full review.
- If a CAPTCHA appears mid-run, the script does not pause for you to
  solve it - it just stops cleanly. Worth revisiting if this becomes
  annoying in practice.
- Only tested against one restaurant (富臨飯店) so far - a second
  restaurant with a differently structured page (e.g. very few reviews,
  no "show all" link) hasn't been tried.
