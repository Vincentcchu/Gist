"""
Phase 2 scraper: checkpointed drip sessions against OpenRice.

Run manually whenever you're at the machine - this is intentionally NOT a
scheduled/unattended job. OpenRice serves a slide-puzzle CAPTCHA to
automated traffic after a handful of requests, with no legitimate way to
solve it without a human. So each run:
  - resumes from Venue.scrape_status / last_review_external_id (never
    restarts from scratch)
  - commits each new review immediately (never buffers to the end)
  - stops the instant a CAPTCHA is detected, leaving state clean to resume
    next time
  - caps itself at a small number of new reviews per run regardless, so a
    session never turns into a long unattended crawl

Each new review gets one extra page visit (its OpenRice permalink) to pull
the full, untruncated text instead of the "...查看更多" listing excerpt -
this means more requests per run than just scrolling the listing page, so
CAPTCHA risk is a bit higher than it used to be, still bounded by
MAX_NEW_REVIEWS_PER_RUN and still fails safe (stops cleanly, resumable).

Usage: python scraper/scrape_reviews.py
"""
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))
# db.py defaults to a cwd-relative sqlite path; pin it to app/local.db
# explicitly so this script works no matter where it's invoked from.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{APP_DIR / 'local.db'}")

import db  # noqa: E402
import models  # noqa: E402

from playwright.sync_api import Locator, Page, sync_playwright  # noqa: E402

CARD_SELECTOR = ".review-post-desktop.poi-detail-review"
# Confirmed wording from the CAPTCHA we actually hit in session 1 (English,
# even on a zh-locale page - likely an unlocalized third-party widget).
CAPTCHA_MARKERS = ["verify to continue", "puzzle piece", "captcha"]
# Confirmed against a real permalink page (scraper/README.md's ".review-post-extract"
# is only a truncated "...查看更多" excerpt on the listing card; this is the
# full, untruncated text, on the review's own permalink page).
FULL_TEXT_SELECTOR = ".review-post-body"


class CaptchaDetected(Exception):
    pass


PROFILE_DIR = Path(__file__).resolve().parent / ".browser_profile"

MAX_NEW_REVIEWS_PER_RUN = 500
MAX_SCROLL_STEPS = 40
# The next batch of reviews loads slowly and in bursts, not smoothly -
# too short a pause here makes "no new cards yet" look like "no more
# cards exist," which previously caused a run to falsely mark a venue
# done after only reaching 75 of 86 reviews.
SCROLL_PAUSE_SECONDS = 10
# A single big jump (page.mouse.wheel(0, 4000)) reliably stalls OpenRice's
# lazy-load around ~30 cards, no matter how long you wait afterward - the
# site's lazy-load needs frequent small scroll events, not one huge one.
# Confirmed by hand: switching to many small ticks fixed it (growth
# 15 -> 30 -> 45 -> 60 -> 75 -> 87 in one continuous run).
SCROLL_TICK_PX = 350
SCROLL_TICK_PAUSE_SECONDS = 0.15
TICKS_PER_SCROLL_STEP = 12  # ~4000px total per step, same reach as before


def read_total_review_count(page: Page) -> int | None:
    # The overview page shows a "食評 (N)" label with the site's own
    # advertised review count - the real completion target, since it
    # drifts as real users post and isn't something we should hardcode.
    labels = page.locator(".detail-title-common")
    for i in range(labels.count()):
        text = labels.nth(i).inner_text().strip()
        match = re.search(r"食評\s*\((\d+)\)", text)
        if match:
            return int(match.group(1))
    return None


def looks_like_captcha(page: Page) -> bool:
    try:
        body_text = page.inner_text("body").lower()
    except Exception:
        return False
    return any(marker in body_text for marker in CAPTCHA_MARKERS)


def extract_external_id(permalink: str | None) -> str | None:
    if not permalink:
        return None
    # permalinks look like /zh/hongkong/review/--e6502606
    match = re.search(r"-e?(\d+)$", permalink)
    return match.group(1) if match else None


def extract_card_fields(card: Locator) -> dict | None:
    try:
        permalink = card.locator("a.wrapper-left").first.get_attribute("href")
    except Exception:
        permalink = None
    external_id = extract_external_id(permalink)
    if not external_id:
        return None

    try:
        name = card.locator(".info-top a").first.inner_text().strip()
    except Exception:
        name = None

    dots = card.locator(".with-dot")
    date_text = dots.nth(1).inner_text().strip() if dots.count() > 1 else None

    full_stars = card.locator(".poi-detail-rating-star-full-y").count()
    half_stars = card.locator(".poi-detail-rating-star-half-y").count()
    rating = full_stars + 0.5 * half_stars

    try:
        # Truncated "...查看更多" excerpt - kept only as a fallback if
        # fetch_full_review_text() fails for a non-CAPTCHA reason.
        excerpt_text = card.locator(".review-post-extract").first.inner_text().strip()
    except Exception:
        excerpt_text = ""

    return {
        "external_id": external_id,
        "permalink": permalink,
        "name": name,
        "date_text": date_text,
        "rating": rating,
        "excerpt_text": excerpt_text,
    }


def fetch_full_review_text(context, permalink: str) -> str:
    # Separate tab so this doesn't disturb the main listing page's scroll
    # position/state.
    full_url = f"https://www.openrice.com{permalink}"
    page = context.new_page()
    try:
        page.goto(full_url, wait_until="load", timeout=30000)
        time.sleep(1)
        if looks_like_captcha(page):
            raise CaptchaDetected()
        return page.locator(FULL_TEXT_SELECTOR).first.inner_text().strip()
    finally:
        page.close()


def scrape_venue(session, venue: models.Venue) -> None:
    print(f"\n=== Scraping {venue.name} ({venue.source_url}) ===")
    venue.scrape_status = "in_progress"
    session.commit()

    existing_ids = {
        row.external_id
        for row in session.query(models.Review.external_id).filter(
            models.Review.venue_id == venue.id,
            models.Review.external_id.isnot(None),
        )
    }
    seen_this_run: set[str] = set()
    new_count = 0
    max_cards_seen = 0

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # Persistent context so cookies/session carry over between manual
        # sessions instead of looking like a brand new visitor every time.
        context = p.chromium.launch_persistent_context(
            str(PROFILE_DIR), headless=False, slow_mo=50
        )
        page = context.pages[0] if context.pages else context.new_page()

        page.goto(venue.source_url, wait_until="load", timeout=60000)
        time.sleep(2)

        if looks_like_captcha(page):
            print("CAPTCHA detected on initial load - stopping this session.")
            venue.scrape_status = "blocked_captcha"
            session.commit()
            context.close()
            return

        total_expected = read_total_review_count(page)
        if total_expected is not None:
            venue.total_reviews_expected = total_expected
            session.commit()
            print(f"Site advertises {total_expected} total reviews for {venue.name}.")
        else:
            print("Could not read the site's advertised review total this run.")

        # A hard page load straight to the /reviews URL only ever
        # server-renders the first 15 cards - scrolling never loads more.
        # Clicking the "show all N reviews" link as an in-app SPA
        # navigation is what actually wires up infinite scroll (confirmed
        # by hand: direct load capped at 15, click-through reached 30+).
        show_more = page.locator("a.show-more-reviews")
        if show_more.count() > 0:
            show_more.first.click()
            time.sleep(2)
            if looks_like_captcha(page):
                print("CAPTCHA detected after clicking through to reviews - stopping.")
                venue.scrape_status = "blocked_captcha"
                session.commit()
                context.close()
                return

        prev_card_count = -1
        for _ in range(MAX_SCROLL_STEPS):
            cards = page.locator(CARD_SELECTOR)
            card_count = cards.count()
            if card_count != prev_card_count:
                print(f"  (scroll progress: {card_count} cards visible)")
                prev_card_count = card_count
            max_cards_seen = max(max_cards_seen, card_count)
            for i in range(card_count):
                if new_count >= MAX_NEW_REVIEWS_PER_RUN:
                    break

                fields = extract_card_fields(cards.nth(i))
                if not fields or fields["external_id"] in seen_this_run:
                    continue
                seen_this_run.add(fields["external_id"])
                if fields["external_id"] in existing_ids:
                    continue

                try:
                    raw_text = fetch_full_review_text(context, fields["permalink"])
                except CaptchaDetected:
                    print("CAPTCHA detected while fetching a review's full "
                          "text - stopping this session.")
                    venue.scrape_status = "blocked_captcha"
                    session.commit()
                    context.close()
                    return
                except Exception as e:
                    print(f"  ! couldn't fetch full text for review "
                          f"{fields['external_id']} ({e}), falling back to "
                          f"the truncated excerpt.")
                    raw_text = fields["excerpt_text"]

                session.add(models.Review(
                    venue_id=venue.id,
                    external_id=fields["external_id"],
                    raw_text=raw_text,
                    source="openrice",
                    scraped_at=datetime.now(UTC),
                    status="pending",
                ))
                venue.last_review_external_id = fields["external_id"]
                venue.last_scraped_at = datetime.now(UTC)
                session.commit()

                existing_ids.add(fields["external_id"])
                new_count += 1
                print(f"  + saved review {fields['external_id']} "
                      f"({fields['rating']}/5, {fields['name']}, {fields['date_text']})")

            if new_count >= MAX_NEW_REVIEWS_PER_RUN:
                print(f"Reached per-run cap of {MAX_NEW_REVIEWS_PER_RUN} new reviews, stopping.")
                break

            for _ in range(TICKS_PER_SCROLL_STEP):
                page.mouse.wheel(0, SCROLL_TICK_PX)
                time.sleep(SCROLL_TICK_PAUSE_SECONDS)
            time.sleep(SCROLL_PAUSE_SECONDS)  # let lazy-load network calls catch up

            if looks_like_captcha(page):
                print("CAPTCHA detected mid-scroll - stopping this session.")
                venue.scrape_status = "blocked_captcha"
                session.commit()
                context.close()
                return

        context.close()

    # Scrolled to the cap or to the end without a CAPTCHA. "0 new found this
    # run" is NOT reliable proof there are no more - OpenRice's lazy-load
    # depth per session appears throttled independent of scroll patience
    # (confirmed: a 40-scroll/5s-pause run still stalled well short of the
    # count already reached across earlier separate sessions). So instead of
    # trusting this run's behavior, compare our actual saved count against
    # the site's own advertised total - the only real completion signal.
    if max_cards_seen == 0:
        print("WARNING: no review cards matched at all this session - "
              "selector may be broken or the page didn't render as expected. "
              "Leaving status as pending for a retry.")
        venue.scrape_status = "pending"
    else:
        current_total = session.query(models.Review).filter(
            models.Review.venue_id == venue.id
        ).count()
        if venue.total_reviews_expected is not None and current_total >= venue.total_reviews_expected:
            venue.scrape_status = "done"
        else:
            # Includes the case where we couldn't read the site's total this
            # run - safer to keep retrying than to guess done without one.
            venue.scrape_status = "pending"
    session.commit()
    total_str = f"{current_total}/{venue.total_reviews_expected}" if max_cards_seen else "n/a"
    print(f"Session done: {new_count} new review(s) saved for {venue.name} "
          f"(saw up to {max_cards_seen} cards on page, total {total_str}, "
          f"status now '{venue.scrape_status}').")


def main() -> None:
    db.init_db()
    session = db.SessionLocal()
    try:
        venue = (
            session.query(models.Venue)
            .filter(models.Venue.scrape_status.in_(["pending", "in_progress", "blocked_captcha"]))
            .filter(models.Venue.source_url.isnot(None))
            .first()
        )
        if venue is None:
            print("Nothing to scrape (all done, or no venue has a source_url configured).")
            return
        scrape_venue(session, venue)
    finally:
        session.close()


if __name__ == "__main__":
    main()
