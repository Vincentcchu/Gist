"""The description hierarchy: area -> seed -> refinement, from description_hierarchy.yaml.

    python ml/eval/descriptions.py --write-readme   # regenerate the README's rule-1 table
    python ml/eval/descriptions.py --check          # exit 1 if the README table is out of date

Used two ways:
- **The README.** Rule 1's table of seeds and refinements in ml/data/real/README.md is generated
  from the YAML, between the `<!-- descriptions:begin/end -->` markers, so the written standard
  and the file can't drift apart.
- **Scoring.** `level(description, "seed")` and `level(description, "area")` coarsen a description,
  so descriptions can be compared at three levels: exact, seed (a refinement counts as its parent
  seed) and area. A description that isn't in the hierarchy keeps its own text at seed level and
  has no area (None).
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HIERARCHY_PATH = REPO_ROOT / "ml" / "data" / "real" / "description_hierarchy.yaml"
README_PATH = REPO_ROOT / "ml" / "data" / "real" / "README.md"
BEGIN, END = "<!-- descriptions:begin -->", "<!-- descriptions:end -->"

# The same format every description must have (verify_dataset.py checks labels against it).
DESCRIPTION_FORMAT = re.compile(r"[a-z]+( [a-z]+){0,3}")
LEVELS = ("exact", "seed", "area")


@dataclass
class Refinement:
    name: str
    parent: str  # a seed of the same area, or the area itself
    definition: str
    area: str


@dataclass
class Hierarchy:
    seeds: dict[str, list[str]]  # area -> seeds, in file order
    refinements: dict[str, Refinement] = field(default_factory=dict)

    def area_of(self, description: str) -> str | None:
        for area, seeds in self.seeds.items():
            if description in seeds:
                return area
        if description in self.refinements:
            return self.refinements[description].area
        return None

    def seed_of(self, description: str) -> str:
        """The seed a description counts as. Unknown descriptions, seeds, and refinements
        attached directly to an area count as themselves."""
        refinement = self.refinements.get(description)
        if refinement and refinement.parent != refinement.area:
            return refinement.parent
        return description

    def level(self, description: str, level: str) -> str | None:
        if level == "exact":
            return description
        if level == "seed":
            return self.seed_of(description)
        if level == "area":
            return self.area_of(description)
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")

    def all_seeds(self) -> list[str]:
        return [seed for seeds in self.seeds.values() for seed in seeds]


def load_hierarchy(path: Path = HIERARCHY_PATH) -> Hierarchy:
    """Read and validate the YAML. Raises ValueError on any inconsistency."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    hierarchy = Hierarchy(seeds={})
    seen: set[str] = set()

    def claim(name: str) -> None:
        if not DESCRIPTION_FORMAT.fullmatch(name):
            raise ValueError(
                f"{name!r} doesn't have the description format (1-4 lowercase words)"
            )
        if name in seen:
            raise ValueError(f"{name!r} appears more than once")
        seen.add(name)

    for area, entry in raw.items():
        hierarchy.seeds[area] = list(entry["seeds"])
        for seed in hierarchy.seeds[area]:
            claim(seed)

    for area, entry in raw.items():
        for name, spec in (entry.get("refinements") or {}).items():
            claim(name)
            parent = spec["parent"]
            if parent != area and parent not in hierarchy.seeds[area]:
                raise ValueError(
                    f"refinement {name!r}: parent {parent!r} is neither a seed of {area!r} "
                    "nor the area itself"
                )
            hierarchy.refinements[name] = Refinement(
                name, parent, spec["definition"], area
            )
    return hierarchy


def render_table(hierarchy: Hierarchy) -> str:
    """The README's rule-1 table: seeds by area, then the refinements with their definitions."""
    lines = ["  | Area | Seeds |", "  |---|---|"]
    for area, seeds in hierarchy.seeds.items():
        lines.append(f"  | {area.capitalize()} | {', '.join(seeds)} |")
    if hierarchy.refinements:
        lines += [
            "",
            "  **Refinements** are more specific than a seed. Use one when it fits exactly; when",
            "  descriptions are compared at seed level, it counts as its parent seed (one placed",
            "  directly under an area counts as itself).",
            "",
            "  | Refinement | Parent | Use for |",
            "  |---|---|---|",
        ]
        for refinement in hierarchy.refinements.values():
            parent = (
                f"{refinement.parent} (the area)"
                if refinement.parent == refinement.area
                else refinement.parent
            )
            lines.append(
                f"  | {refinement.name} | {parent} | {refinement.definition} |"
            )
    return "\n".join(lines)


def readme_with_table(readme: str, table: str) -> str:
    """The README text with the block between the markers replaced by `table`."""
    start, end = readme.index(BEGIN), readme.index(END)
    return readme[: start + len(BEGIN)] + "\n" + table + "\n" + readme[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-readme", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    hierarchy = load_hierarchy()
    readme = README_PATH.read_text(encoding="utf-8")
    updated = readme_with_table(readme, render_table(hierarchy))
    if args.check:
        if updated != readme:
            sys.exit("README rule-1 table is out of date: run --write-readme")
        print(f"README table matches {HIERARCHY_PATH.name}")
        return
    README_PATH.write_text(updated, encoding="utf-8")
    seeds = len(hierarchy.all_seeds())
    print(
        f"wrote {seeds} seeds and {len(hierarchy.refinements)} refinements into {README_PATH}"
    )


if __name__ == "__main__":
    main()
