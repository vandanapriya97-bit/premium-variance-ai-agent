from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when the uploaded workbook does not match the expected structure."""


class PremiumDataStore:
    """Deterministic analysis layer for the Premium Variance Agent.

    Gemini never reads Excel directly and never performs the financial arithmetic.
    This class owns all filtering, aggregation, comparison, ranking, reconciliation,
    and persistency interpretation.
    """

    SHEET_NAME = "Premium Variance Data"

    REQUIRED_COLUMNS = {
        "Quarter",
        "Treaty",
        "Portfolio",
        "Overall Variance",
        "Client",
        "Change in mortality (COB1)",
        "Change in lapse (COB2)",
        "Age",
        "Gender",
        "Accrual True Up",
    }

    DIMENSIONS = ["Client", "Treaty", "Portfolio", "Age", "Gender", "Quarter"]
    METRICS = [
        "Overall Variance",
        "Change in mortality (COB1)",
        "Change in lapse (COB2)",
        "Variance Unexplained through COBS",
        "Accrual True Up",
        "LeftOver Persistency",
    ]

    METRIC_ALIASES = {
        "overall variance": "Overall Variance",
        "overall var": "Overall Variance",
        "variance": "Overall Variance",
        "mortality": "Change in mortality (COB1)",
        "cob1": "Change in mortality (COB1)",
        "change in mortality": "Change in mortality (COB1)",
        "lapse": "Change in lapse (COB2)",
        "cob2": "Change in lapse (COB2)",
        "change in lapse": "Change in lapse (COB2)",
        "variance unexplained through cobs": "Variance Unexplained through COBS",
        "unexplained through cobs": "Variance Unexplained through COBS",
        "unexplained variance": "Variance Unexplained through COBS",
        "accrual true up": "Accrual True Up",
        "true up": "Accrual True Up",
        "persistency": "LeftOver Persistency",
        "leftover persistency": "LeftOver Persistency",
    }

    def __init__(self, excel_source: str | Path | BinaryIO):
        self.data = self._load_and_validate(excel_source)

    @staticmethod
    def _clean_number(value: float | int) -> int | float:
        value = float(value)
        return int(round(value)) if value.is_integer() else round(value, 2)

    @staticmethod
    def _to_number(series: pd.Series, column: str) -> pd.Series:
        converted = pd.to_numeric(series, errors="coerce")
        if converted.isna().any():
            bad_rows = list(converted[converted.isna()].index + 2)
            raise DataValidationError(
                f"Column '{column}' contains non-numeric values at Excel rows {bad_rows[:20]}."
            )
        return converted.astype(float)

    def _load_and_validate(self, excel_source: str | Path | BinaryIO) -> pd.DataFrame:
        try:
            df = pd.read_excel(excel_source, sheet_name=self.SHEET_NAME)
        except Exception as exc:
            raise DataValidationError(
                f"Could not read sheet '{self.SHEET_NAME}' from the workbook."
            ) from exc

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Workbook is missing required columns: {sorted(missing)}"
            )

        df = df.copy()
        df["Quarter"] = df["Quarter"].astype(str).str.strip().str.upper()
        df["Client"] = df["Client"].astype(str).str.strip()
        df["Portfolio"] = df["Portfolio"].astype(str).str.strip()
        df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()

        numeric = [
            "Treaty",
            "Overall Variance",
            "Change in mortality (COB1)",
            "Change in lapse (COB2)",
            "Age",
            "Accrual True Up",
        ]
        for column in numeric:
            df[column] = self._to_number(df[column], column)

        df["Treaty"] = df["Treaty"].astype(int)
        df["Age"] = df["Age"].astype(int)

        allowed_quarters = {"Q1", "Q2", "Q3", "Q4"}
        invalid_quarters = sorted(set(df["Quarter"]) - allowed_quarters)
        if invalid_quarters:
            raise DataValidationError(
                f"Invalid quarter values found: {invalid_quarters}. Expected Q1-Q4."
            )

        duplicate_key = df.duplicated(["Quarter", "Treaty"], keep=False)
        if duplicate_key.any():
            examples = df.loc[duplicate_key, ["Quarter", "Treaty"]].head(20).to_dict("records")
            raise DataValidationError(
                f"Duplicate Quarter + Treaty rows found. Examples: {examples}"
            )

        # Treaty-level descriptive attributes should remain stable across quarters.
        for column in ["Client", "Portfolio", "Age", "Gender"]:
            counts = df.groupby("Treaty")[column].nunique(dropna=False)
            bad = counts[counts > 1]
            if not bad.empty:
                raise DataValidationError(
                    f"Column '{column}' changes across quarters for treaties: {bad.index.tolist()[:20]}"
                )

        # Recalculate formula-driven fields in Python. We deliberately do not trust
        # Excel's cached formula results.
        df["Variance Unexplained through COBS"] = (
            df["Overall Variance"]
            - df["Change in mortality (COB1)"]
            - df["Change in lapse (COB2)"]
        )
        df["LeftOver Persistency"] = (
            df["Variance Unexplained through COBS"] - df["Accrual True Up"]
        )

        df["Explained Total"] = (
            df["Change in mortality (COB1)"]
            + df["Change in lapse (COB2)"]
            + df["Accrual True Up"]
            + df["LeftOver Persistency"]
        )
        df["Reconciliation Difference"] = df["Overall Variance"] - df["Explained Total"]
        df["Reconciled"] = df["Reconciliation Difference"].abs() <= 1.0
        df["Result"] = df["Overall Variance"].apply(self._result_label)

        if not df["Reconciled"].all():
            bad = df.loc[
                ~df["Reconciled"],
                ["Quarter", "Treaty", "Overall Variance", "Explained Total"],
            ].head(20).to_dict("records")
            raise DataValidationError(f"Rows do not reconcile. Examples: {bad}")

        quarter_order = pd.CategoricalDtype(["Q1", "Q2", "Q3", "Q4"], ordered=True)
        df["Quarter"] = df["Quarter"].astype(quarter_order)
        return df.sort_values(["Treaty", "Quarter"]).reset_index(drop=True)

    @staticmethod
    def _result_label(value: float) -> str:
        if value > 0:
            return "Favourable"
        if value < 0:
            return "Unfavourable"
        return "Neutral"

    @property
    def dataset_stats(self) -> dict[str, int]:
        return {
            "Rows": int(len(self.data)),
            "Clients": int(self.data["Client"].nunique()),
            "Treaties": int(self.data["Treaty"].nunique()),
            "Quarters": int(self.data["Quarter"].nunique()),
        }

    def _normalize_client(self, value: str) -> str:
        raw = str(value).strip()
        if raw.lower().startswith("client"):
            raw = raw[6:].strip()
        if raw.isdigit():
            candidate = f"Client {int(raw):03d}"
        else:
            candidate = str(value).strip()

        available = {c.lower(): c for c in self.data["Client"].astype(str).unique()}
        return available.get(candidate.lower(), candidate)

    def _normalize_portfolio(self, value: str) -> str:
        raw = str(value).strip()
        aliases = {
            "term": "Term",
            "perm": "Perm",
            "permanent": "Perm",
            "group": "Group",
        }
        return aliases.get(raw.lower(), raw)

    @staticmethod
    def _normalize_gender(value: str) -> str:
        raw = str(value).strip().lower()
    
        aliases = {
            "m": "M",
            "male": "M",
            "males": "M",
            "man": "M",
            "men": "M",
    
            "f": "F",
            "female": "F",
            "females": "F",
            "woman": "F",
            "women": "F",
        }
    
        return aliases.get(raw, str(value).strip().upper())

    @staticmethod
    def _normalize_quarter(value: str) -> str:
        return str(value).strip().upper().replace("QUARTER ", "Q")

    def normalize_metric(self, metric: str | None) -> str:
        if not metric:
            return "Overall Variance"
        raw = str(metric).strip()
        if raw in self.METRICS:
            return raw
        normalized = self.METRIC_ALIASES.get(raw.lower())
        if normalized:
            return normalized
        raise DataValidationError(
            f"Metric '{metric}' is not supported. Supported metrics are: {', '.join(self.METRICS)}"
        )

    def _validate_requested_values(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        if filters.get("client") is not None:
            client = self._normalize_client(filters["client"])
            if client not in set(self.data["Client"].astype(str)):
                return {
                    "status": "not_found",
                    "message": f"{client} was not found in the uploaded dataset.",
                    "rows": [],
                }

        if filters.get("treaty") is not None:
            try:
                treaty = int(filters["treaty"])
            except (TypeError, ValueError):
                return {"status": "invalid", "message": "Treaty must be a number.", "rows": []}
            if treaty not in set(self.data["Treaty"]):
                return {
                    "status": "not_found",
                    "message": f"Treaty {treaty} was not found in the uploaded dataset.",
                    "rows": [],
                }

        if filters.get("portfolio") is not None:
            portfolio = self._normalize_portfolio(filters["portfolio"])
            available = sorted(self.data["Portfolio"].unique())
            if portfolio not in available:
                return {
                    "status": "not_found",
                    "message": (
                        f"Portfolio '{filters['portfolio']}' was not found. "
                        f"Available portfolios are: {', '.join(available)}."
                    ),
                    "rows": [],
                }

        if filters.get("gender") is not None:
            gender = self._normalize_gender(filters["gender"])
            available = sorted(self.data["Gender"].unique())
            if gender not in available:
                return {
                    "status": "not_found",
                    "message": (
                        f"Gender '{filters['gender']}' was not found. "
                        f"Available values are: {', '.join(available)}."
                    ),
                    "rows": [],
                }

        if filters.get("quarter") is not None:
            quarter = self._normalize_quarter(filters["quarter"])
            if quarter not in {"Q1", "Q2", "Q3", "Q4"}:
                return {
                    "status": "not_found",
                    "message": f"Quarter {filters['quarter']} is not available. Available quarters are Q1, Q2, Q3 and Q4.",
                    "rows": [],
                }

        ages = set(int(v) for v in self.data["Age"].unique())
        if filters.get("age_min") is not None and filters.get("age_max") is not None:
            if int(filters["age_min"]) > int(filters["age_max"]):
                return {
                    "status": "invalid",
                    "message": "The minimum age cannot be greater than the maximum age.",
                    "rows": [],
                }
        elif filters.get("age_min") is not None and filters.get("age_max") is None:
            # age_min alone means an exact age unless the router explicitly provides a range.
            if int(filters["age_min"]) not in ages:
                return {
                    "status": "not_found",
                    "message": f"No records were found for age {int(filters['age_min'])}.",
                    "rows": [],
                }
        return None

    def _apply_filters(self, filters: dict[str, Any] | None) -> tuple[pd.DataFrame | None, dict[str, Any] | None, dict[str, Any]]:
        filters = {k: v for k, v in (filters or {}).items() if v is not None}
        error = self._validate_requested_values(filters)
        if error:
            return None, error, filters

        df = self.data.copy()
        normalized: dict[str, Any] = {}

        if "client" in filters:
            normalized["client"] = self._normalize_client(filters["client"])
            df = df[df["Client"] == normalized["client"]]
        if "treaty" in filters:
            normalized["treaty"] = int(filters["treaty"])
            df = df[df["Treaty"] == normalized["treaty"]]
        if "portfolio" in filters:
            normalized["portfolio"] = self._normalize_portfolio(filters["portfolio"])
            df = df[df["Portfolio"] == normalized["portfolio"]]
        if "gender" in filters:
            normalized["gender"] = self._normalize_gender(filters["gender"])
            df = df[df["Gender"] == normalized["gender"]]
        if "quarter" in filters:
            normalized["quarter"] = self._normalize_quarter(filters["quarter"])
            df = df[df["Quarter"].astype(str) == normalized["quarter"]]

        age_min = filters.get("age_min")
        age_max = filters.get("age_max")
        if age_min is not None and age_max is not None:
            normalized["age_min"] = int(age_min)
            normalized["age_max"] = int(age_max)
            df = df[(df["Age"] >= int(age_min)) & (df["Age"] <= int(age_max))]
        elif age_min is not None:
            normalized["age_min"] = int(age_min)
            normalized["age_max"] = int(age_min)
            df = df[df["Age"] == int(age_min)]
        elif age_max is not None:
            normalized["age_max"] = int(age_max)
            df = df[df["Age"] <= int(age_max)]

        if df.empty:
            return None, {
                "status": "no_results",
                "message": "The requested values are valid individually, but no rows match that combination of filters.",
                "rows": [],
            }, normalized

        return df, None, normalized

    def _persistency_insight(self, values: dict[str, float | int]) -> dict[str, Any]:
        p = float(values["LeftOver Persistency"])
        adverse_drivers = {
            name: float(values[name])
            for name in [
                "Change in mortality (COB1)",
                "Change in lapse (COB2)",
                "Accrual True Up",
                "LeftOver Persistency",
            ]
            if float(values[name]) < 0
        }
        largest_adverse = None
        if adverse_drivers:
            largest_adverse = min(adverse_drivers, key=adverse_drivers.get)

        if p < 0:
            statement = (
                "Negative LeftOver Persistency may indicate higher lapse activity or lower retention than anticipated. "
                "This is an indicator only; the dataset does not contain observed lapse rates."
            )
            direction = "Adverse"
        elif p > 0:
            statement = (
                "Positive LeftOver Persistency may indicate better retention or lower lapse activity than anticipated. "
                "This is an indicator only; the dataset does not contain observed lapse rates."
            )
            direction = "Favourable"
        else:
            statement = "LeftOver Persistency is neutral for this result."
            direction = "Neutral"

        return {
            "value": self._clean_number(p),
            "direction": direction,
            "is_largest_adverse_driver": bool(p < 0 and largest_adverse == "LeftOver Persistency"),
            "statement": statement,
        }

    def _summary(self, df: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {
            "Records": int(len(df)),
            "Clients": int(df["Client"].nunique()),
            "Treaties": int(df["Treaty"].nunique()),
        }
        for metric in self.METRICS:
            result[metric] = self._clean_number(df[metric].sum())
        result["Result"] = self._result_label(float(result["Overall Variance"]))
        result["Reconciliation Difference"] = self._clean_number(
            result["Overall Variance"]
            - result["Change in mortality (COB1)"]
            - result["Change in lapse (COB2)"]
            - result["Accrual True Up"]
            - result["LeftOver Persistency"]
        )
        result["Reconciled"] = abs(float(result["Reconciliation Difference"])) <= 1.0
        result["Persistency Insight"] = self._persistency_insight(result)
        return result

    def _default_group_by(self, filters: dict[str, Any]) -> str:
        if filters.get("treaty") is not None:
            return "Quarter"
        if filters.get("client") is not None:
            return "Treaty"
        if filters.get("quarter") is not None:
            return "Portfolio"
        return "Quarter"

    def _aggregate_rows(self, df: pd.DataFrame, group_by: str) -> list[dict[str, Any]]:
        if group_by not in self.DIMENSIONS:
            raise DataValidationError(
                f"Breakdown '{group_by}' is not supported. Choose from {', '.join(self.DIMENSIONS)}."
            )

        rows: list[dict[str, Any]] = []
        grouped = df.groupby(group_by, observed=True, sort=False)
        for key, subset in grouped:
            summary = self._summary(subset)
            row: dict[str, Any] = {group_by: int(key) if group_by in {"Treaty", "Age"} else str(key)}

            # Helpful treaty metadata without creating arbitrary LLM joins.
            if group_by == "Treaty":
                row.update({
                    "Client": str(subset["Client"].iloc[0]),
                    "Portfolio": str(subset["Portfolio"].iloc[0]),
                    "Age": int(subset["Age"].iloc[0]),
                    "Gender": str(subset["Gender"].iloc[0]),
                })

            row.update({k: v for k, v in summary.items() if k != "Persistency Insight"})
            rows.append(row)

        if group_by == "Quarter":
            order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
            rows.sort(key=lambda r: order.get(str(r["Quarter"]), 99))
        elif group_by in {"Treaty", "Age"}:
            rows.sort(key=lambda r: r[group_by])
        else:
            rows.sort(key=lambda r: str(r[group_by]))
        return rows

    @staticmethod
    def _filter_title(filters: dict[str, Any]) -> str:
        if not filters:
            return "Dataset Overview"
        parts: list[str] = []
        if filters.get("client"):
            parts.append(str(filters["client"]))
        if filters.get("treaty") is not None:
            parts.append(f"Treaty {filters['treaty']}")
        if filters.get("portfolio"):
            parts.append(f"{filters['portfolio']} Portfolio")
        if filters.get("gender"):
            parts.append(f"Gender {filters['gender']}")
        if filters.get("age_min") is not None and filters.get("age_max") is not None:
            if filters["age_min"] == filters["age_max"]:
                parts.append(f"Age {filters['age_min']}")
            else:
                parts.append(f"Age {filters['age_min']}-{filters['age_max']}")
        if filters.get("quarter"):
            parts.append(str(filters["quarter"]))
        return " | ".join(parts) if parts else "Premium Variance Analysis"

    def analyse_data(
        self,
        *,
        client: str | None = None,
        treaty: int | None = None,
        portfolio: str | None = None,
        age_min: int | None = None,
        age_max: int | None = None,
        gender: str | None = None,
        quarter: str | None = None,
        group_by: str | None = None,
        row_limit: int = 50,
    ) -> dict[str, Any]:
        filters = {
            "client": client,
            "treaty": treaty,
            "portfolio": portfolio,
            "age_min": age_min,
            "age_max": age_max,
            "gender": gender,
            "quarter": quarter,
        }
        df, error, normalized = self._apply_filters(filters)
        if error:
            return error
        assert df is not None

        chosen_group = group_by or self._default_group_by(normalized)
        rows = self._aggregate_rows(df, chosen_group)
        truncated = len(rows) > row_limit
        rows = rows[:row_limit]
        totals = self._summary(df)

        return {
            "status": "success",
            "query_type": "analysis",
            "title": self._filter_title(normalized),
            "filters": normalized,
            "group_by": chosen_group,
            "rows": rows,
            "rows_truncated": truncated,
            "row_limit": row_limit,
            "totals": totals,
            "persistency_insight": totals["Persistency Insight"],
        }

    def compare_quarters(
        self,
        quarter_1: str,
        quarter_2: str,
        *,
        client: str | None = None,
        treaty: int | None = None,
        portfolio: str | None = None,
        age_min: int | None = None,
        age_max: int | None = None,
        gender: str | None = None,
    ) -> dict[str, Any]:
        q1 = self._normalize_quarter(quarter_1)
        q2 = self._normalize_quarter(quarter_2)
        for q in [q1, q2]:
            if q not in {"Q1", "Q2", "Q3", "Q4"}:
                return {
                    "status": "not_found",
                    "message": f"Quarter {q} is not available. Available quarters are Q1, Q2, Q3 and Q4.",
                    "rows": [],
                }
        if q1 == q2:
            return {
                "status": "invalid",
                "message": "Please choose two different quarters for a comparison.",
                "rows": [],
            }

        base_filters = {
            "client": client,
            "treaty": treaty,
            "portfolio": portfolio,
            "age_min": age_min,
            "age_max": age_max,
            "gender": gender,
        }
        base_df, error, normalized = self._apply_filters(base_filters)
        if error:
            return error
        assert base_df is not None

        df1 = base_df[base_df["Quarter"].astype(str) == q1]
        df2 = base_df[base_df["Quarter"].astype(str) == q2]
        if df1.empty or df2.empty:
            missing = q1 if df1.empty else q2
            return {
                "status": "no_results",
                "message": f"No matching records were found for {missing}.",
                "rows": [],
            }

        s1 = self._summary(df1)
        s2 = self._summary(df2)
        movement = {metric: self._clean_number(float(s2[metric]) - float(s1[metric])) for metric in self.METRICS}
        overall_move = float(movement["Overall Variance"])
        movement_label = "Improved" if overall_move > 0 else "Deteriorated" if overall_move < 0 else "No change"

        p_move = float(movement["LeftOver Persistency"])
        if p_move < 0:
            persistency_movement = (
                "Persistency deteriorated between the two quarters. The more negative movement may indicate higher lapse "
                "activity or lower retention, but this dataset does not contain observed lapse rates."
            )
        elif p_move > 0:
            persistency_movement = (
                "Persistency improved between the two quarters. The positive movement may indicate better retention or "
                "lower lapse activity, but this dataset does not contain observed lapse rates."
            )
        else:
            persistency_movement = "There was no movement in LeftOver Persistency between the two quarters."

        comparison_rows = []
        for label, summary in [(q1, s1), (q2, s2)]:
            comparison_rows.append({
                "Quarter": label,
                **{metric: summary[metric] for metric in self.METRICS},
                "Result": summary["Result"],
                "Records": summary["Records"],
                "Clients": summary["Clients"],
                "Treaties": summary["Treaties"],
            })
        comparison_rows.append({
            "Quarter": f"Movement {q2} - {q1}",
            **movement,
            "Result": movement_label,
            "Records": "",
            "Clients": "",
            "Treaties": "",
        })

        detail_rows: list[dict[str, Any]] = []
        if normalized.get("client") or normalized.get("treaty"):
            group_dim = "Treaty"
            g1 = df1.groupby(group_dim, observed=True)[self.METRICS].sum()
            g2 = df2.groupby(group_dim, observed=True)[self.METRICS].sum()
            keys = sorted(set(g1.index).intersection(g2.index))
            for treaty_id in keys:
                meta = base_df[base_df["Treaty"] == treaty_id].iloc[0]
                detail_rows.append({
                    "Treaty": int(treaty_id),
                    "Client": str(meta["Client"]),
                    "Portfolio": str(meta["Portfolio"]),
                    f"Overall Variance {q1}": self._clean_number(g1.loc[treaty_id, "Overall Variance"]),
                    f"Overall Variance {q2}": self._clean_number(g2.loc[treaty_id, "Overall Variance"]),
                    "Overall Movement": self._clean_number(g2.loc[treaty_id, "Overall Variance"] - g1.loc[treaty_id, "Overall Variance"]),
                    f"Persistency {q1}": self._clean_number(g1.loc[treaty_id, "LeftOver Persistency"]),
                    f"Persistency {q2}": self._clean_number(g2.loc[treaty_id, "LeftOver Persistency"]),
                    "Persistency Movement": self._clean_number(g2.loc[treaty_id, "LeftOver Persistency"] - g1.loc[treaty_id, "LeftOver Persistency"]),
                })

        title_filters = dict(normalized)
        title = self._filter_title(title_filters)
        return {
            "status": "success",
            "query_type": "comparison",
            "title": f"{title} | {q1} vs {q2}",
            "filters": normalized,
            "quarter_1": q1,
            "quarter_2": q2,
            "rows": comparison_rows,
            "detail_rows": detail_rows,
            "movement": movement,
            "movement_result": movement_label,
            "persistency_comparison_insight": persistency_movement,
            "quarter_1_persistency_insight": s1["Persistency Insight"],
            "quarter_2_persistency_insight": s2["Persistency Insight"],
        }

    def rank_results(
        self,
        *,
        metric: str,
        group_by: str = "Treaty",
        direction: str = "ascending",
        top_n: int = 10,
        client: str | None = None,
        treaty: int | None = None,
        portfolio: str | None = None,
        age_min: int | None = None,
        age_max: int | None = None,
        gender: str | None = None,
        quarter: str | None = None,
    ) -> dict[str, Any]:
        metric = self.normalize_metric(metric)
        if group_by not in self.DIMENSIONS:
            return {
                "status": "invalid",
                "message": f"Ranking dimension '{group_by}' is not supported.",
                "rows": [],
            }
        direction = str(direction).lower()
        if direction not in {"ascending", "descending"}:
            return {
                "status": "invalid",
                "message": "Ranking direction must be ascending or descending.",
                "rows": [],
            }
        top_n = max(1, min(int(top_n or 10), 50))

        filters = {
            "client": client,
            "treaty": treaty,
            "portfolio": portfolio,
            "age_min": age_min,
            "age_max": age_max,
            "gender": gender,
            "quarter": quarter,
        }
        df, error, normalized = self._apply_filters(filters)
        if error:
            return error
        assert df is not None

        rows = self._aggregate_rows(df, group_by)
        rows.sort(key=lambda r: float(r[metric]), reverse=(direction == "descending"))
        rows = rows[:top_n]
        label = "Lowest" if direction == "ascending" else "Highest"

        return {
            "status": "success",
            "query_type": "ranking",
            "title": f"{label} {top_n} {group_by} values by {metric}",
            "filters": normalized,
            "group_by": group_by,
            "metric": metric,
            "direction": direction,
            "top_n": top_n,
            "rows": rows,
            "persistency_note": (
                "For LeftOver Persistency, more negative values may indicate higher lapse activity or lower retention than anticipated. "
                "This is an indicator only; observed lapse rates are not present in the dataset."
                if metric == "LeftOver Persistency"
                else None
            ),
        }

    def get_dataset_overview(self) -> dict[str, Any]:
        result = self.analyse_data(group_by="Quarter")
        if result.get("status") == "success":
            result["query_type"] = "overview"
            result["title"] = "Dataset Overview"
        return result
