"""Tests for the description hierarchy and the README table generated from it.

pytest ml/eval/test_descriptions.py
"""

import pytest

from descriptions import (
    README_PATH,
    load_hierarchy,
    readme_with_table,
    render_table,
)


@pytest.fixture(scope="module")
def hierarchy():
    return load_hierarchy()


def test_counts(hierarchy):
    assert len(hierarchy.all_seeds()) == 50
    assert len(hierarchy.refinements) == 5
    assert list(hierarchy.seeds)[:2] == ["food", "drinks"]


@pytest.mark.parametrize(
    "description, exact, seed, area",
    [
        ("food taste", "food taste", "food taste", "food"),  # a seed is itself
        (
            "food availability",
            "food availability",
            "menu variety",
            "food",
        ),  # refinement -> parent
        ("queue seating", "queue seating", "queue management", "getting in"),
        (
            "location ambience",
            "location ambience",
            "location ambience",
            "location",
        ),  # under the area
        ("view quality", "view quality", "view quality", None),  # not in the hierarchy
    ],
)
def test_three_levels(hierarchy, description, exact, seed, area):
    assert hierarchy.level(description, "exact") == exact
    assert hierarchy.level(description, "seed") == seed
    assert hierarchy.level(description, "area") == area


def test_readme_table_is_generated_from_the_yaml(hierarchy):
    readme = README_PATH.read_text(encoding="utf-8")
    assert (
        readme_with_table(readme, render_table(hierarchy)) == readme
    ), "README rule-1 table is out of date: python ml/eval/descriptions.py --write-readme"


def write(tmp_path, text):
    path = tmp_path / "hierarchy.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "text, problem",
    [
        ("a:\n  seeds: [food taste, food taste]\n", "more than once"),
        ("a:\n  seeds: [Food Taste]\n", "description format"),
        (
            "a:\n  seeds: [x y]\n  refinements:\n    z w: {parent: nope, definition: d}\n",
            "neither a seed",
        ),
        (
            "a:\n  seeds: [x y]\nb:\n  seeds: [p q]\n  refinements:\n    z w: {parent: x y, definition: d}\n",
            "neither a seed",
        ),
    ],
)
def test_inconsistent_files_are_rejected(tmp_path, text, problem):
    with pytest.raises(ValueError, match=problem):
        load_hierarchy(write(tmp_path, text))
