"""
Phase 2, session 1 (continued): inspect the dedicated /reviews sub-page,
which is where OpenRice actually paginates all reviews (the overview page
only server-renders 3). Still inspection-only, no DB writes.
"""
from playwright.sync_api import sync_playwright

URL = (
    "https://www.openrice.com/zh/hongkong/r-"
    "%E5%AF%8C%E8%87%A8%E9%A3%AF%E5%BA%97-%E9%8A%85%E9%91%BC%E7%81%A3-"
    "%E7%B2%B5%E8%8F%9C-%E5%BB%A3%E6%9D%B1-%E7%84%A1%E8%82%89%E9%A4%90%E5%96%AE-r161154/reviews"
)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        page = browser.new_page()

        xhr_hits = []

        def on_response(response):
            url = response.url
            if any(k in url.lower() for k in ["review", "comment", "graphql"]) and "orstatic" not in url:
                xhr_hits.append((response.status, url))

        page.on("response", on_response)

        print(f"Navigating to: {URL}")
        response = page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        print(f"Initial response status: {response.status if response else 'N/A'}")
        page.wait_for_timeout(3000)

        initial_html = page.content()
        print(f"Initial /reviews HTML length: {len(initial_html)}")
        with open("scraper/reviews_page_initial.html", "w", encoding="utf-8") as f:
            f.write(initial_html)

        review_card_count = page.locator(".review-post-trim-desktop, [class*='review-post']:not([class*='review-post-']) ").count()
        print(f"'.review-post-trim-desktop' count on initial load: "
              f"{page.locator('.review-post-trim-desktop').count()}")

        # Scroll multiple times to see if infinite scroll loads more reviews
        counts_over_scroll = [page.locator(".review-post-trim-desktop").count()]
        for i in range(8):
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(1200)
            counts_over_scroll.append(page.locator(".review-post-trim-desktop").count())
        print(f"Review card count after each scroll step: {counts_over_scroll}")

        # look for numbered pagination controls
        pagination_candidates = [
            "[class*='pagination']",
            "[class*='Pagination']",
            "a[href*='page=']",
            "button[class*='page']",
            "[class*='load-more']",
            "[class*='LoadMore']",
        ]
        print("\n=== Pagination probe ===")
        for sel in pagination_candidates:
            count = page.locator(sel).count()
            print(f"  {sel!r}: {count} matches")
            if count > 0:
                for i in range(min(count, 3)):
                    try:
                        el = page.locator(sel).nth(i)
                        print(f"    -> outerHTML[:300]: {el.evaluate('e => e.outerHTML')[:300]}")
                    except Exception as e:
                        print(f"    -> error reading element: {e}")

        print("\n=== XHR/fetch responses touching review/comment/graphql ===")
        for status, url in xhr_hits:
            print(f"  [{status}] {url}")
        if not xhr_hits:
            print("  (none captured)")

        after_scroll_html = page.content()
        with open("scraper/reviews_page_after_scroll.html", "w", encoding="utf-8") as f:
            f.write(after_scroll_html)
        print(f"\nAfter-scroll HTML length: {len(after_scroll_html)} "
              f"(delta: {len(after_scroll_html) - len(initial_html)})")

        print("\nCurrent URL after scroll interactions:", page.url)

        print("\nBrowser staying open 10s for manual look...")
        page.wait_for_timeout(10000)
        browser.close()


if __name__ == "__main__":
    main()
