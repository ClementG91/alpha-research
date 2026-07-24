from __future__ import annotations

import numpy as np
import pandas as pd

import lse_intraday_macro_alpha as base
import lse_intraday_rate_alpha as campaign


def relative_class_neutral_targets(
    raw: pd.DataFrame,
    classes: pd.Series,
) -> pd.DataFrame:
    """Hedge every shock with instruments from the same economic class.

    Zero-signal class members remain eligible as the hedge basket. This converts
    isolated shocks into relative-value trades instead of market-direction bets.
    Each active class has zero net notional and receives equal gross risk.
    """
    result = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    class_members = {
        str(class_name): raw.columns.intersection(classes.index[classes == class_name])
        for class_name in sorted(classes.dropna().unique())
        if class_name != "benchmark"
    }
    for timestamp in raw.index:
        class_rows: dict[str, pd.Series] = {}
        for class_name, members in class_members.items():
            values = raw.loc[timestamp, members].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 2 or values.abs().max() <= 0.0:
                continue
            centered = values - values.mean()
            gross = centered.abs().sum()
            if gross <= 1e-12 or not (centered.gt(0.0).any() and centered.lt(0.0).any()):
                continue
            class_rows[class_name] = centered / gross
        if not class_rows:
            continue
        class_budget = 1.0 / len(class_rows)
        for values in class_rows.values():
            result.loc[timestamp, values.index] = values * class_budget
    return result


base.normalise_targets = relative_class_neutral_targets

if __name__ == "__main__":
    raise SystemExit(campaign.main())
