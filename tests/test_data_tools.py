from pathlib import Path

import pytest

from data_tools import PremiumDataStore


DATA_FILE = Path(__file__).parents[1] / "data" / "Premium Variance Data.xlsx"


@pytest.fixture(scope="module")
def store():
    return PremiumDataStore(DATA_FILE)


def test_dataset_shape(store):
    assert store.dataset_stats == {
        "Rows": 5000,
        "Clients": 250,
        "Treaties": 1250,
        "Quarters": 4,
    }


def test_python_recalculates_formula_driven_fields(store):
    row = store.data.iloc[0]
    expected_unexplained = (
        row["Overall Variance"]
        - row["Change in mortality (COB1)"]
        - row["Change in lapse (COB2)"]
    )
    expected_persistency = expected_unexplained - row["Accrual True Up"]
    assert row["Variance Unexplained through COBS"] == expected_unexplained
    assert row["LeftOver Persistency"] == expected_persistency
    assert bool(row["Reconciled"]) is True


def test_client_query_lists_five_treaties(store):
    result = store.analyse_data(client="Client 001")
    assert result["status"] == "success"
    assert result["group_by"] == "Treaty"
    assert len(result["rows"]) == 5
    assert result["totals"]["Treaties"] == 5
    assert result["totals"]["Records"] == 20


def test_treaty_query_breaks_down_four_quarters(store):
    result = store.analyse_data(treaty=10001)
    assert result["status"] == "success"
    assert result["group_by"] == "Quarter"
    assert [row["Quarter"] for row in result["rows"]] == ["Q1", "Q2", "Q3", "Q4"]


def test_portfolio_and_quarter_filter(store):
    result = store.analyse_data(portfolio="Term", quarter="Q3")
    assert result["status"] == "success"
    assert result["filters"]["portfolio"] == "Term"
    assert result["filters"]["quarter"] == "Q3"
    assert result["totals"]["Records"] > 0


def test_exact_age_filter(store):
    result = store.analyse_data(age_min=45, age_max=45)
    assert result["status"] == "success"
    assert result["filters"]["age_min"] == 45
    assert result["filters"]["age_max"] == 45


def test_quarter_comparison_math(store):
    result = store.compare_quarters("Q1", "Q4", client="Client 001")
    assert result["status"] == "success"
    rows = result["rows"]
    q1 = rows[0]["Overall Variance"]
    q4 = rows[1]["Overall Variance"]
    movement = rows[2]["Overall Variance"]
    assert movement == q4 - q1
    assert len(result["detail_rows"]) == 5


def test_persistency_ranking_is_sorted(store):
    result = store.rank_results(
        metric="LeftOver Persistency",
        group_by="Treaty",
        direction="ascending",
        top_n=10,
        quarter="Q4",
    )
    assert result["status"] == "success"
    values = [row["LeftOver Persistency"] for row in result["rows"]]
    assert len(values) == 10
    assert values == sorted(values)
    assert result["persistency_note"] is not None


def test_unknown_portfolio_is_not_invented(store):
    result = store.analyse_data(portfolio="Universal Life")
    assert result["status"] == "not_found"
    assert "Available portfolios" in result["message"]


def test_invalid_quarter_is_rejected(store):
    result = store.analyse_data(quarter="Q7")
    assert result["status"] == "not_found"
    assert "Q1, Q2, Q3 and Q4" in result["message"]
