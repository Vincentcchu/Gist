"""
Phase 2, session 1: inspection-only script for OpenRice review scraping.
No DB writes. Launches a headed browser, navigates to the restaurant page,
and prints diagnostics to console so we can decide on selectors + approach.
"""
from playwright.sync_api import sync_playwright

URL = (
    "https://www.openrice.com/zh/hongkong/r-"
    "%E5%AF%8C%E8%87%A8%E9%A3%AF%E5%BA%97-%E9%8A%85%E9%91%BC%E7%81%A3-"
    "%E7%B2%B5%E8%8F%9C-%E5%BB%A3%E6%9D%B1-%E7%84%A1%E8%82%89%E9%A4%90%E5%96%AE-r161154"
)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=100)
        page = browser.new_page()

        # Capture XHR/fetch responses so we can tell if reviews load via API
        xhr_hits = []

        def on_response(response):
            url = response.url
            if any(k in url.lower() for k in ["review", "comment", "graphql", "api"]):
                xhr_hits.append((response.status, url))

        page.on("response", on_response)

        print(f"Navigating to: {URL}")
        response = page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        print(f"Initial response status: {response.status if response else 'N/A'}")

        # Give the page a moment to settle / fire any XHRs
        page.wait_for_timeout(4000)

        # 1. Check initial HTML for review-looking content before any interaction
        initial_html = page.content()
        print(f"\n=== Initial HTML length: {len(initial_html)} chars ===")
        # crude heuristic: look for common review container hints
        for hint in ["review", "comment", "Review", "star"]:
            print(f"  occurrences of '{hint}': {initial_html.count(hint)}")

        with open("scraper/page_initial.html", "w", encoding="utf-8") as f:
            f.write(initial_html)
        print("Saved initial HTML to scraper/page_initial.html")

        # 2. Scroll down to trigger lazy loading / infinite scroll if present
        for i in range(5):
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(1000)

        page.wait_for_timeout(2000)
        after_scroll_html = page.content()
        with open("scraper/page_after_scroll.html", "w", encoding="utf-8") as f:
            f.write(after_scroll_html)
        print(f"\n=== After-scroll HTML length: {len(after_scroll_html)} chars "
              f"(delta: {len(after_scroll_html) - len(initial_html)}) ===")

        print("\n=== XHR/fetch responses touching review/comment/api/graphql ===")
        for status, url in xhr_hits:
            print(f"  [{status}] {url}")
        if not xhr_hits:
            print("  (none captured)")

        # 3. Try a handful of plausible selectors and report which one hits
        candidate_selectors = [
            "[class*='review']",
            "[class*='Review']",
            "[data-testid*='review']",
            "section[id*='review']",
            "div[id*='review']",
            "li[class*='review']",
        ]
        print("\n=== Selector probe ===")
        working_selector = None
        for sel in candidate_selectors:
            count = page.locator(sel).count()
            print(f"  {sel!r}: {count} matches")
            if count > 0 and working_selector is None:
                working_selector = sel

        if working_selector:
            print(f"\n=== Outer HTML of first match for {working_selector!r} ===")
            outer = page.locator(working_selector).first.evaluate("el => el.outerHTML")
            print(outer[:3000])
        else:
            print("\nNo candidate selector matched anything — reviews may be in an iframe, "
                  "shadow DOM, or require login/different route. Inspect saved HTML files.")

        print("\n=== Page title ===")
        print(page.title())

        print("\nBrowser will stay open for 15s for manual inspection via devtools. "
              "Press Ctrl+C in terminal to close early.")
        page.wait_for_timeout(15000)

        browser.close()


if __name__ == "__main__":
    main()
