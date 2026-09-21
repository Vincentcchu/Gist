from datetime import UTC, datetime, timedelta

import db
import models

# Each entry: (venue_index, review_text, source, days_ago, status, aspects)
# aspects: list of (aspect_text, sentiment, span) or [] for reviews left
# "pending" to preview the pre-labeling state.
REVIEWS = [
    (0, "Check-in was a breeze and the staff upgraded us for free. Room had a "
        "stunning harbour view.", "manual-seed", 30, "processed", [
            ("check-in speed", "positive", "Check-in was a breeze"),
            ("room view", "positive", "stunning harbour view"),
        ]),
    (0, "Waited 40 minutes to check in and the AC in our room was broken all "
        "night.", "manual-seed", 28, "processed", [
            ("check-in speed", "negative", "Waited 40 minutes to check in"),
            ("air conditioning", "negative", "AC in our room was broken all night"),
        ]),
    (0, "The gym was tiny but the breakfast buffet had a huge variety, "
        "especially the dim sum.", "manual-seed", 27, "processed", [
            ("gym size", "negative", "gym was tiny"),
            ("breakfast", "positive", "huge variety, especially the dim sum"),
        ]),
    (0, "Room was clean but 房間好細, barely fit two suitcases.", "manual-seed",
        25, "processed", [
            ("room cleanliness", "positive", "Room was clean"),
            ("room size", "negative", "房間好細, barely fit two suitcases"),
        ]),
    (0, "Front desk staff spoke perfect English and Cantonese, super helpful "
        "with restaurant recommendations.", "manual-seed", 22, "processed", [
            ("staff helpfulness", "positive",
             "super helpful with restaurant recommendations"),
        ]),
    (0, "Pool was closed for maintenance the entire stay, nobody told us in "
        "advance.", "manual-seed", 20, "processed", [
            ("pool availability", "negative", "Pool was closed for maintenance"),
            ("communication", "negative", "nobody told us in advance"),
        ]),
    (0, "Good value for the price, nothing fancy but does the job.",
        "manual-seed", 18, "processed", [
            ("value for money", "positive", "Good value for the price"),
        ]),
    (0, "Noisy street outside, could hear traffic until 2am.", "manual-seed",
        3, "pending", []),
    (0, "呢間酒店都幾ok, location convenient but wifi kept dropping.",
        "manual-seed", 1, "pending", []),
    (1, "Boutique feel, loved the rooftop bar and the personalized "
        "welcome note.", "manual-seed", 29, "processed", [
            ("rooftop bar", "positive", "loved the rooftop bar"),
            ("welcome experience", "positive", "personalized welcome note"),
        ]),
    (1, "Small rooms but the design is gorgeous, very Instagrammable.",
        "manual-seed", 26, "processed", [
            ("room size", "negative", "Small rooms"),
            ("interior design", "positive", "design is gorgeous, very Instagrammable"),
        ]),
    (1, "Breakfast was mediocre, same three items every day.", "manual-seed",
        24, "processed", [
            ("breakfast", "negative", "mediocre, same three items every day"),
        ]),
    (1, "Honestly a mixed bag - great cocktails downstairs, but the "
        "elevator broke twice during our stay.", "manual-seed", 21,
        "processed", [
            ("bar", "positive", "great cocktails downstairs"),
            ("elevator reliability", "negative", "elevator broke twice"),
        ]),
    (1, "Staff went above and beyond, arranged a late checkout with no "
        "hassle.", "manual-seed", 19, "processed", [
            ("staff helpfulness", "positive",
             "arranged a late checkout with no hassle"),
        ]),
    (1, "Bit pricey for what you get, but the location is unbeatable.",
        "manual-seed", 15, "processed", [
            ("value for money", "negative", "Bit pricey for what you get"),
            ("location", "positive", "location is unbeatable"),
        ]),
    (1, "Booked for one night, front desk was closed when we arrived late.",
        "manual-seed", 2, "pending", []),
]

VENUES = [
    ("Harbourview Grand", "Hong Kong"),
    ("Central Boutique Inn", "Hong Kong"),
]

# Real restaurant targeted by the Phase 2 scraper (scraper/scrape_reviews.py).
# No fake reviews are seeded for this one - the scraper populates it.
# NOTE: source_url is the overview page, not /reviews directly - a hard
# page load straight to /reviews only ever renders the first 15 reviews
# with no further infinite scroll. The scraper clicks the "show all N
# reviews" link itself, which is what actually wires up pagination.
SCRAPE_TARGETS = [
    (
        "富臨飯店",
        "銅鑼灣, Hong Kong",
        "https://www.openrice.com/zh/hongkong/r-%E5%AF%8C%E8%87%A8%E9%A3%AF%E5%BA%97-"
        "%E9%8A%85%E9%91%BC%E7%81%A3-%E7%B2%B5%E8%8F%9C-%E5%BB%A3%E6%9D%B1-"
        "%E7%84%A1%E8%82%89%E9%A4%90%E5%96%AE-r161154",
    ),
    (
        # ~3,223 reviews at time of writing (vs. 87 for the target above) -
        # expect this one to stay "pending" across many manual scrape
        # sessions before it ever reaches "done".
        "澳洲牛奶公司",
        "佐敦, Hong Kong",
        "https://www.openrice.com/zh/hongkong/r-australia-dairy-company-"
        "jordan-hong-kong-style-dessert-r90",
    ),
]


def run():
    db.init_db()
    session = db.SessionLocal()
    try:
        # Fake demo data (VENUES/REVIEWS) only needs seeding once.
        if session.query(models.Venue).first():
            print("DB already seeded with demo data, skipping VENUES/REVIEWS.")
        else:
            venues = [models.Venue(name=name, location=location) for name, location in VENUES]
            session.add_all(venues)
            session.flush()  # populate venue.id

            for venue_index, text, source, days_ago, status, aspects in REVIEWS:
                review = models.Review(
                    venue_id=venues[venue_index].id,
                    raw_text=text,
                    source=source,
                    scraped_at=datetime.now(UTC) - timedelta(days=days_ago),
                    status=status,
                )
                session.add(review)
                session.flush()  # populate review.id

                for aspect_text, sentiment, span in aspects:
                    session.add(models.AspectExtraction(
                        review_id=review.id,
                        aspect_text=aspect_text,
                        sentiment=sentiment,
                        span=span,
                        model_version="manual-seed-v1",
                    ))
            session.commit()
            print(f"Seeded {len(venues)} demo venues, {len(REVIEWS)} reviews.")

        # SCRAPE_TARGETS is the source of truth for real restaurants the
        # scraper should pick up - added independently of the demo-data
        # check above so a new target here gets picked up by re-running
        # this script even on an already-seeded DB.
        existing_urls = {
            url for (url,) in session.query(models.Venue.source_url)
            if url is not None
        }
        new_targets = [
            models.Venue(name=name, location=location, source_url=source_url,
                         scrape_status="pending")
            for name, location, source_url in SCRAPE_TARGETS
            if source_url not in existing_urls
        ]
        session.add_all(new_targets)
        session.commit()
        print(f"Added {len(new_targets)} new scrape target venue(s).")
    finally:
        session.close()


if __name__ == "__main__":
    run()
