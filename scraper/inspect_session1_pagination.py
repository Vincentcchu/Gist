"""
Phase 2, session 1 (part 3): determine how the /reviews sub-page loads beyond
the first ~15 reviews (infinite scroll vs numbered pages), using the correct
selector found from earlier inspection: div.review-post-desktop.poi-detail-review
Still inspection-only, no DB writes.
"""
from playwright.sync_api import sync_playwright

URL = (
    "https://www.openrice.com/zh/hongkong/r-"
    "%E5%AF%8C%E8%87%A8%E9%A3%AF%E5%BA%97-%E9%8A%85%E9%91%BC%E7%81%A3-"
    "%E7%B2%B5%E8%8F%9C-%E5%BB%A3%E6%9D%B1-%E7%84%A1%E8%82%89%E9%A4%90%E5%96%AE-r161154/reviews"
)

CARD_SELECTOR = "div.review-post-desktop.poi-detail-review"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=30)
        page = browser.new_page()

        api_hits = []

        def on_response(response):
            url = response.url
            if "openrice.com/api" in url or "openrice.com/orga/api" in url:
                api_hits.append((response.status, url))

        page.on("response", on_response)

        page.goto(URL, wait_until="load", timeout=60000)
        page.wait_for_load_state("networkidle", timeout=15000)
        page.wait_for_timeout(2000)

        count = page.locator(CARD_SELECTOR).count()
        print(f"Initial card count ({CARD_SELECTOR}): {count}")

        prev_count = -1
        stable_rounds = 0
        for i in range(25):
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(1500)
            cur_count = page.locator(CARD_SELECTOR).count()
            print(f"  scroll {i+1}: card count = {cur_count}")
            if cur_count == prev_count:
                stable_rounds += 1
                if stable_rounds >= 3:
                    print("  count stable for 3 rounds, stopping scroll loop")
                    break
            else:
                stable_rounds = 0
            prev_count = cur_count

        print(f"\nFinal card count: {page.locator(CARD_SELECTOR).count()} (restaurant page says 86 total)")

        print("\n=== openrice.com API calls seen ===")
        for status, url in api_hits:
            print(f"  [{status}] {url}")

        # Print full outerHTML of first card for selector design
        first_card_html = page.locator(CARD_SELECTOR).first.evaluate("el => el.outerHTML")
        with open("scraper/first_review_card.html", "w", encoding="utf-8") as f:
            f.write(first_card_html)
        print(f"\nSaved first review card HTML ({len(first_card_html)} chars) to scraper/first_review_card.html")

        # Extract first 5 reviews' key fields via targeted sub-selectors
        print("\n=== Extracted fields for first 5 reviews ===")
        for i in range(min(5, page.locator(CARD_SELECTOR).count())):
            card = page.locator(CARD_SELECTOR).nth(i)
            try:
                name = card.locator(".review-post-writer-name").first.inner_text()
            except Exception as e:
                name = f"<error: {e}>"
            try:
                text = card.locator(".review-post-extract").first.inner_text()
            except Exception as e:
                text = f"<error: {e}>"
            try:
                writer_info = card.locator(".review-post-writer-info").first.inner_text()
            except Exception as e:
                writer_info = f"<error: {e}>"
            try:
                full_stars = card.locator(".poi-detail-rating-star-full-y").count()
                half_stars = card.locator(".poi-detail-rating-star-half-y").count()
            except Exception as e:
                full_stars, half_stars = -1, -1

            print(f"\n--- Review {i+1} ---")
            print(f"Name: {name}")
            print(f"Writer info (level/date/views): {writer_info}")
            print(f"Rating stars: {full_stars} full, {half_stars} half (out of 5)")
            print(f"Text ({len(text)} chars): {text[:500]}{'...' if len(text) > 500 else ''}")

        print("\nBrowser staying open 10s...")
        page.wait_for_timeout(10000)
        browser.close()


if __name__ == "__main__":
    main()
