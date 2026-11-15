"""Tests for the County explorer's "Download these cards (Excel)" button.

Run:
    python -m pytest tests/test_cards_download.py -v

The workbook's Cards sheet is built from the same county fact sheet as the assistant, so these
tests check it against that sheet rather than against hard-coded numbers.
Every test reads real local data and is skipped when
data/raw/acs5_2024_usdash_county.parquet is absent.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "Streamlit" / "us_county_explorer.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dashboard import RAW_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "acs5_2024_usdash_county.parquet").exists(),
    reason="needs data/raw/acs5_2024_usdash_county.parquet (run ingestion/pull_usdash.py)",
)

JEFFERSON_CO = "08059"  # Jefferson County, Colorado


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("county_explorer_cards_download", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data(app):
    return app._load_all()


def _read(xlsx: bytes) -> pd.DataFrame:
    """The Cards sheet; also checks the Data dictionary defines every column."""
    sheets = pd.read_excel(io.BytesIO(xlsx), sheet_name=None)
    assert list(sheets) == ["Cards", "Data dictionary"]
    cards, dictionary = sheets["Cards"], sheets["Data dictionary"]
    assert list(dictionary["Column"]) == list(cards.columns)
    assert dictionary["What it means"].str.len().gt(20).all()
    return cards


def test_rows_follow_the_chosen_cards_and_match_the_fact_sheet(app, data):
    sheet = app._county_fact_sheet(data, JEFFERSON_CO, set())
    df = _read(app._cards_xlsx(sheet, ["Total population", "Median household income"]))

    assert list(df["Measure"]) == ["Total population", "Median household income"]
    assert (df["County"] == "Jefferson County, Colorado").all()
    income = df.iloc[1]
    v = sheet["measures"]["median_household_income"]["values"]
    for header, key in [("Estimate", "estimate"), ("Margin of error (90%)", "moe"),
                        ("90% range low", "ci_low"), ("90% range high", "ci_high"),
                        ("Reliability score", "score"), ("CV rank in state (1 = lowest CV)", "cv_rank")]:
        assert income[header] == pytest.approx(v[key], abs=0.5), header  # rounded like the card
    assert income["CV (%)"] == pytest.approx(v["cv"] * 100, abs=0.05)
    assert income["Reliability band"] == sheet["measures"]["median_household_income"]["blanks"]["band"]
    assert income["Status"] == "Published"
    assert "B19013" in income["Source"]


def test_a_card_without_local_data_has_a_status_and_no_numbers(app, data):
    sheet = app._county_fact_sheet(data, JEFFERSON_CO, {"Unemployed"})
    row = _read(app._cards_xlsx(sheet, ["Unemployed"])).iloc[0]
    assert row["Status"] == "Not in this dashboard's local data"
    assert pd.isna(row["Estimate"]) and pd.isna(row["CV (%)"])


def test_works_without_the_anthropic_package(app, data, monkeypatch):
    monkeypatch.setattr(app, "assistant", None)
    sheet = app._county_fact_sheet(data, JEFFERSON_CO, set())
    assert len(_read(app._cards_xlsx(sheet, list(app.MEASURES)))) == len(app.MEASURES)


def test_button_shows_below_the_cards_and_hides_with_no_cards():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["main_tabs"] = "County explorer"
    at.session_state["us_geo"] = {"level": "county", "code": JEFFERSON_CO, "state_scope": "08"}
    at.run()
    assert not at.exception
    assert len(at.get("download_button")) == 1

    at.multiselect(key="card_checklist").set_value([]).run()
    assert not at.exception
    assert len(at.get("download_button")) == 0
