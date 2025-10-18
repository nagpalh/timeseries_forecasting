import json
import os
import secrets
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "app.db"
DEFAULT_FORECAST_STEPS = 96
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "timeseries-forecast-secret")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))


def _get_or_create_csrf_token() -> str:
    token = session.get("_csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _validate_csrf_token() -> None:
    session_token = session.get("_csrf_token")
    form_token = request.form.get("_csrf_token")
    if not session_token or not form_token or not secrets.compare_digest(session_token, form_token):
        abort(400, description="Invalid CSRF token")


@app.context_processor
def inject_globals() -> dict[str, Any]:
    return {
        "current_year": datetime.utcnow().year,
        "csrf_token": _get_or_create_csrf_token,
    }


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        g.db = connection
    return g.db


@app.teardown_appcontext
def close_db(exception: Optional[BaseException]) -> None:  # noqa: D401, ARG001
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            columns_json TEXT NOT NULL,
            original_data_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            missing_json TEXT NOT NULL,
            cleaned_series_json TEXT NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()


init_db()


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def _strip_timezone(series: pd.Series) -> pd.Series:
    if is_datetime64tz_dtype(series):
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


def _stringify_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return ", ".join(_stringify_value(item) for item in value.tolist())

    normalized = _format_value(value)
    if normalized is None:
        return "—"
    if isinstance(normalized, float):
        return f"{normalized:.6g}"
    return str(normalized)


def _format_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _format_index_value(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return str(value)


def _infer_axis_title(index: pd.Index) -> str:
    return "Time" if is_datetime64_any_dtype(index) else "Index"


def _dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in dataframe.to_dict(orient="records"):
        formatted = {str(key): _format_value(val) for key, val in record.items()}
        records.append(formatted)
    return records


def _parse_timestamp_column(column: pd.Series) -> tuple[pd.Series, str]:
    if column.empty:
        empty = pd.Series([], dtype="datetime64[ns]")
        empty.name = "timestamp"
        return empty, "no timestamps provided"

    attempts: list[tuple[int, str, pd.Series]] = []

    def register(candidate: Optional[pd.Series], label: str) -> None:
        if candidate is None:
            return
        candidate = _strip_timezone(candidate)
        valid = int(candidate.notna().sum())
        attempts.append((valid, label, candidate))

    if is_datetime64_any_dtype(column):
        register(pd.to_datetime(column, errors="coerce"), "pre-parsed datetimes")

    numeric = pd.to_numeric(column, errors="coerce")
    numeric_non_na = numeric.dropna()
    if not numeric_non_na.empty:
        diffs = numeric_non_na.diff().dropna()
        if not diffs.empty and np.allclose(diffs.values, diffs.iloc[0]) and abs(diffs.iloc[0] - 1) < 1e-9:
            sequential = numeric_non_na.astype(int)
            sequential.index = numeric_non_na.index
            sequential.name = "timestamp"
            register(sequential, "sequential numeric index")

        epoch_thresholds = {"ns": 1e16, "us": 1e13, "ms": 1e10, "s": 1e7}
        numeric_abs_max = numeric_non_na.abs().max()
        for unit in ("ns", "us", "ms", "s"):
            if numeric_abs_max < epoch_thresholds[unit]:
                continue
            parsed = pd.to_datetime(numeric, unit=unit, errors="coerce")
            register(parsed, f"numeric epoch ({unit})")
            if parsed.notna().all():
                break

    as_string = column.astype(str).str.strip().replace({"": np.nan})
    parse_strategies = [
        ({"dayfirst": True, "infer_datetime_format": True, "cache": True}, "string parse (day-first)"),
        ({"dayfirst": False, "infer_datetime_format": True, "cache": True}, "string parse (month-first)"),
        ({"dayfirst": True, "utc": True, "cache": True}, "string parse (day-first utc)"),
        ({"dayfirst": False, "utc": True, "cache": True}, "string parse (month-first utc)"),
    ]

    for kwargs, label in parse_strategies:
        try:
            parsed = pd.to_datetime(as_string, errors="coerce", **kwargs)
        except TypeError:
            parsed = pd.to_datetime(as_string, errors="coerce")
        register(parsed, label)

    if attempts:
        best_valid, label, series = max(attempts, key=lambda item: item[0])
        if best_valid > 0:
            series.name = "timestamp"
            return series, label

    raise ValueError("Could not parse the first column into valid dates.")


def _summarize_fit_diagnostics(fit) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "converged": bool(getattr(fit, "converged", False)),
        "general": [],
        "details": [],
        "messages": [],
    }

    for attr, label in (("aic", "AIC"), ("bic", "BIC"), ("sse", "SSE")):
        value = getattr(fit, attr, None)
        if value is not None:
            diagnostics["general"].append({"label": label, "value": _stringify_value(value)})

    fit_details = getattr(fit, "fit_details", None)
    if isinstance(fit_details, dict):
        for key, value in fit_details.items():
            if key == "messages":
                continue
            diagnostics["details"].append({
                "label": key.replace("_", " ").title(),
                "value": _stringify_value(value),
            })
        messages = fit_details.get("messages")
        if messages:
            if isinstance(messages, (list, tuple, set)):
                diagnostics["messages"] = [str(item) for item in messages if item]
            else:
                diagnostics["messages"] = [str(messages)]

    return diagnostics


def _load_dataframe(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.filename or ""
    if filename == "":
        raise ValueError("No file selected.")

    if not _allowed_file(filename):
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    extension = os.path.splitext(filename)[1].lower()
    try:
        if extension == ".csv":
            dataframe = pd.read_csv(uploaded_file)
        else:
            dataframe = pd.read_excel(uploaded_file)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("Unable to read the uploaded file. Please check the contents and try again.") from exc

    if dataframe.empty:
        raise ValueError("Uploaded file is empty.")

    return dataframe


def _prepare_series(dataframe: pd.DataFrame) -> tuple[pd.Series, dict[str, Any], dict[str, Any]]:
    if dataframe.shape[1] == 0:
        raise ValueError("Uploaded data must contain at least one column.")

    timestamp_note: str

    if dataframe.shape[1] == 1:
        values = pd.to_numeric(dataframe.iloc[:, 0], errors="coerce")
        timestamp_sequence = pd.Series(
            np.arange(1, len(values) + 1, dtype=int),
            name="timestamp",
        )
        timestamp_note = "Generated sequential index (1..N) because no timestamp column was provided."
    else:
        try:
            timestamp_series, timestamp_note = _parse_timestamp_column(dataframe.iloc[:, 0])
        except ValueError:
            timestamp_series = pd.Series(
                np.arange(1, len(dataframe) + 1, dtype=int),
                name="timestamp",
            )
            timestamp_note = "Generated sequential index (1..N) because timestamp parsing failed."
        values = pd.to_numeric(dataframe.iloc[:, 1], errors="coerce")
        timestamp_sequence = timestamp_series

    base = pd.DataFrame(
        {
            "timestamp": timestamp_sequence.reset_index(drop=True),
            "value": values.reset_index(drop=True),
        }
    )

    missing_timestamp_count = int(base["timestamp"].isna().sum())
    base = base.dropna(subset=["timestamp"])
    if base.empty:
        raise ValueError("No valid timestamps detected and sequence generation failed.")

    base = base.sort_values("timestamp")
    base = base[~base["timestamp"].duplicated(keep="last")]

    series = base.set_index("timestamp")["value"]
    missing_values_before = int(series.isna().sum())

    cleaned_series = series.copy()
    fill_steps: list[str] = []
    if missing_values_before:
        if is_datetime64_any_dtype(cleaned_series.index):
            cleaned_series = cleaned_series.interpolate(method="time")
            fill_steps.append("time interpolation")
        else:
            cleaned_series = cleaned_series.interpolate(method="linear")
            fill_steps.append("linear interpolation")
        if cleaned_series.isna().sum():
            cleaned_series = cleaned_series.ffill().bfill()
            fill_steps.append("forward/back fill")

    cleaned_series = cleaned_series.astype(float)

    if cleaned_series.notna().sum() < 3:
        raise ValueError("Time series must contain at least three numeric observations after cleaning.")

    summary = {
        "observations": int(cleaned_series.count()),
        "mean": float(cleaned_series.mean()),
        "median": float(cleaned_series.median()),
        "std_dev": float(cleaned_series.std(ddof=0)) if cleaned_series.count() > 1 else 0.0,
        "minimum": float(cleaned_series.min()),
        "maximum": float(cleaned_series.max()),
        "value_range": float(cleaned_series.max() - cleaned_series.min()),
        "start": _format_index_value(cleaned_series.index.min()),
        "end": _format_index_value(cleaned_series.index.max()),
    }

    missing_info = {
        "missing_timestamps": missing_timestamp_count,
        "missing_values_before": missing_values_before,
        "missing_values_after": int(cleaned_series.isna().sum()),
        "fill_steps": fill_steps,
        "timestamp_note": timestamp_note,
    }

    return cleaned_series, summary, missing_info


def _serialize_series(series: pd.Series) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, value in series.items():
        if isinstance(index, (datetime, pd.Timestamp)):
            timestamp = index.isoformat()
        else:
            timestamp = str(index)
        payload.append({"timestamp": timestamp, "value": _format_value(value)})
    return payload


def _series_from_records(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        raise ValueError("Stored dataset is empty.")

    timestamps: list[pd.Timestamp | int | float | str] = []
    values: list[float] = []
    for record in records:
        timestamp_raw = record.get("timestamp")
        value_raw = record.get("value")
        if value_raw is None:
            continue

        try:
            timestamp = pd.to_datetime(timestamp_raw)
        except Exception:  # pylint: disable=broad-except
            timestamp = timestamp_raw

        timestamps.append(timestamp)
        values.append(float(value_raw))

    if not values:
        raise ValueError("Stored dataset has no numeric observations.")

    return pd.Series(values, index=timestamps, name="value")


def _persist_dataset(
    filename: str,
    columns: list[str],
    original_data: list[dict[str, Any]],
    summary: dict[str, Any],
    missing: dict[str, Any],
    cleaned_series: list[dict[str, Any]],
) -> int:
    connection = get_db()
    cursor = connection.execute(
        """
        INSERT INTO datasets (
            filename,
            uploaded_at,
            columns_json,
            original_data_json,
            summary_json,
            missing_json,
            cleaned_series_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            datetime.utcnow().isoformat(),
            json.dumps(columns),
            json.dumps(original_data),
            json.dumps(summary),
            json.dumps(missing),
            json.dumps(cleaned_series),
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def _fetch_dataset(dataset_id: int) -> dict[str, Any]:
    connection = get_db()
    row = connection.execute(
        """
        SELECT
            id,
            filename,
            uploaded_at,
            columns_json,
            original_data_json,
            summary_json,
            missing_json,
            cleaned_series_json
        FROM datasets
        WHERE id = ?
        """,
        (dataset_id,),
    ).fetchone()

    if row is None:
        raise ValueError("Dataset not found. Please upload data first.")

    return {
        "id": row["id"],
        "filename": row["filename"],
        "uploaded_at": row["uploaded_at"],
        "columns": json.loads(row["columns_json"]),
        "original_data": json.loads(row["original_data_json"]),
        "summary": json.loads(row["summary_json"]),
        "missing": json.loads(row["missing_json"]),
        "cleaned_series": json.loads(row["cleaned_series_json"]),
    }


def _forecast_series(series: pd.Series, steps: int = DEFAULT_FORECAST_STEPS) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    model_series = pd.Series(series.values)

    seasonal_periods = min(DEFAULT_FORECAST_STEPS, len(series))
    seasonal_component: Optional[str] = "add" if seasonal_periods >= 2 else None
    seasonal_periods_value: Optional[int] = seasonal_periods if seasonal_periods >= 2 else None

    try:
        model = ExponentialSmoothing(
            model_series,
            trend=None,
            seasonal=seasonal_component,
            seasonal_periods=seasonal_periods_value,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
    except Exception:  # pylint: disable=broad-except
        model = ExponentialSmoothing(
            model_series,
            trend=None,
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)

    fitted_values = pd.Series(fit.fittedvalues.values, index=series.index)
    forecast_values = fit.forecast(steps=steps)

    inferred_freq: Optional[str] = pd.infer_freq(series.index)
    if inferred_freq:
        offset = pd.tseries.frequencies.to_offset(inferred_freq)
        start = series.index[-1] + offset
        forecast_index = pd.date_range(start=start, periods=steps, freq=inferred_freq)
    else:
        forecast_index = pd.RangeIndex(start=len(series), stop=len(series) + steps, name="step")

    forecast_series = pd.Series(forecast_values.values, index=forecast_index, name="forecast")

    diagnostics = _summarize_fit_diagnostics(fit)

    return fitted_values, forecast_series, diagnostics


def _build_history_plot(series: pd.Series) -> dict[str, Any]:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(series.index),
            y=list(series.values),
            mode="lines+markers",
            name="Observations",
            marker=dict(size=6, color="#4d7cfe"),
            line=dict(color="#4d7cfe"),
        )
    )
    figure.update_layout(
        title="Time Series Overview",
        xaxis_title=_infer_axis_title(series.index),
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return json.loads(figure.to_json())


def _build_forecast_plot(series: pd.Series, forecast_values: pd.Series) -> dict[str, Any]:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(series.index),
            y=list(series.values),
            mode="lines",
            name="Historical",
            line=dict(color="#172b4d"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=list(forecast_values.index),
            y=list(forecast_values.values),
            mode="lines",
            name="Forecast",
            line=dict(dash="dash", color="#4d7cfe"),
        )
    )
    figure.update_layout(
        title="Forecast Horizon",
        xaxis_title=_infer_axis_title(forecast_values.index),
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return json.loads(figure.to_json())


def _build_summary_cards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": "Observations",
            "value": f"{summary['observations']:,}",
            "icon": "fa-database",
            "description": "Rows retained after cleaning",
        },
        {
            "label": "Mean",
            "value": f"{summary['mean']:.2f}",
            "icon": "fa-chart-line",
            "description": "Average value across the series",
        },
        {
            "label": "Median",
            "value": f"{summary['median']:.2f}",
            "icon": "fa-wave-square",
            "description": "Centre point of the distribution",
        },
        {
            "label": "Std. Deviation",
            "value": f"{summary['std_dev']:.2f}",
            "icon": "fa-chart-area",
            "description": "Variation around the mean",
        },
        {
            "label": "Range",
            "value": f"{summary['value_range']:.2f}",
            "icon": "fa-arrows-left-right",
            "description": "Difference between min and max",
        },
        {
            "label": "Minimum",
            "value": f"{summary['minimum']:.2f}",
            "icon": "fa-arrow-down-wide-short",
            "description": "Lowest observed value",
        },
        {
            "label": "Maximum",
            "value": f"{summary['maximum']:.2f}",
            "icon": "fa-arrow-up-wide-short",
            "description": "Highest observed value",
        },
        {
            "label": "Start",
            "value": summary["start"],
            "icon": "fa-calendar-day",
            "description": "First timestamp after cleaning",
        },
        {
            "label": "End",
            "value": summary["end"],
            "icon": "fa-calendar-check",
            "description": "Latest timestamp available",
        },
    ]


@app.errorhandler(413)
def handle_large_upload(_error):
    flash("The uploaded file exceeds the maximum allowed size (16 MB).", "error")
    return redirect(url_for("upload"))


@app.route("/")
def home() -> str:
    connection = get_db()
    rows = connection.execute(
        "SELECT id, filename, uploaded_at FROM datasets ORDER BY uploaded_at DESC LIMIT 5"
    ).fetchall()
    recent_datasets = [
        {
            "id": row["id"],
            "filename": row["filename"],
            "uploaded_at": row["uploaded_at"],
        }
        for row in rows
    ]
    return render_template("home.html", recent_datasets=recent_datasets)


@app.route("/upload", methods=["GET", "POST"])
def upload() -> str:
    if request.method == "POST":
        _validate_csrf_token()
        uploaded_file = request.files.get("data_file")
        if uploaded_file is None or uploaded_file.filename == "":
            flash("Please select a CSV or Excel file to upload.", "error")
            return redirect(url_for("upload"))

        try:
            dataframe = _load_dataframe(uploaded_file)
            columns = [str(column) for column in dataframe.columns]
            original_records = _dataframe_to_records(dataframe)

            series, summary, missing = _prepare_series(dataframe)
            cleaned_records = _serialize_series(series)

            dataset_id = _persist_dataset(
                filename=uploaded_file.filename,
                columns=columns,
                original_data=original_records,
                summary=summary,
                missing=missing,
                cleaned_series=cleaned_records,
            )

            flash("Upload successful. Your data is ready to explore.", "success")
            return redirect(url_for("visualize", dataset_id=dataset_id))
        except Exception as exc:  # pylint: disable=broad-except
            flash(str(exc), "error")
            return redirect(url_for("upload"))

    return render_template("upload.html")


@app.route("/datasets/<int:dataset_id>/delete", methods=["POST"])
def delete_dataset(dataset_id: int) -> str:
    _validate_csrf_token()
    connection = get_db()
    cursor = connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    connection.commit()

    if cursor.rowcount:
        flash("Dataset removed successfully.", "success")
    else:
        flash("Dataset not found or already deleted.", "error")

    return redirect(url_for("home"))


@app.route("/datasets/delete-all", methods=["POST"])
def delete_all_datasets() -> str:
    _validate_csrf_token()
    connection = get_db()
    cursor = connection.execute("DELETE FROM datasets")
    connection.commit()

    if cursor.rowcount:
        flash("All datasets have been deleted.", "success")
    else:
        flash("No datasets to delete.", "error")

    return redirect(url_for("home"))


@app.route("/visualize/<int:dataset_id>")
def visualize(dataset_id: int) -> str:
    try:
        dataset = _fetch_dataset(dataset_id)
    except Exception as exc:  # pylint: disable=broad-except
        flash(str(exc), "error")
        return redirect(url_for("upload"))

    series = _series_from_records(dataset["cleaned_series"])
    history_plot = _build_history_plot(series)
    summary_cards = _build_summary_cards(dataset["summary"])

    return render_template(
        "visualize.html",
        dataset=dataset,
        original_columns=dataset["columns"],
        original_data=dataset["original_data"],
        summary_cards=summary_cards,
        missing_info=dataset["missing"],
        history_plot=history_plot,
    )


@app.route("/forecast/<int:dataset_id>")
def forecast(dataset_id: int) -> str:
    try:
        dataset = _fetch_dataset(dataset_id)
    except Exception as exc:  # pylint: disable=broad-except
        flash(str(exc), "error")
        return redirect(url_for("upload"))

    series = _series_from_records(dataset["cleaned_series"])
    _, forecast_values, fit_diagnostics = _forecast_series(series)
    forecast_plot = _build_forecast_plot(series, forecast_values)

    forecast_rows = (
        forecast_values.reset_index()
        .rename(columns={forecast_values.index.name or "index": "timestamp"})
        .to_dict("records")
    )
    formatted_rows = [
        {key: _format_value(value) for key, value in row.items()} for row in forecast_rows
    ]

    return render_template(
        "forecast.html",
        dataset=dataset,
        forecast_rows=formatted_rows,
        forecast_plot=forecast_plot,
        forecast_steps=DEFAULT_FORECAST_STEPS,
        fit_diagnostics=fit_diagnostics,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
