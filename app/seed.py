from datetime import UTC, datetime, timedelta

import db
import models

# Each entry: (hotel_index, review_text, source, days_ago, status, aspects)
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

HOTELS = [
    ("Harbourview Grand", "Hong Kong"),
    ("Central Boutique Inn", "Hong Kong"),
]


def run():
    db.init_db()
    session = db.SessionLocal()
    try:
        if session.query(models.Hotel).first():
            print("DB already seeded, skipping.")
            return

        hotels = [models.Hotel(name=name, location=location) for name, location in HOTELS]
        session.add_all(hotels)
        session.flush()  # populate hotel.id

        for hotel_index, text, source, days_ago, status, aspects in REVIEWS:
            review = models.Review(
                hotel_id=hotels[hotel_index].id,
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
        print(f"Seeded {len(hotels)} hotels, {len(REVIEWS)} reviews.")
    finally:
        session.close()


if __name__ == "__main__":
    run()
