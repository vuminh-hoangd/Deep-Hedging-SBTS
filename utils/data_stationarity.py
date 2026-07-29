
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(series, name="Series"):
    print(f"\n--- Stationarity Analysis for: {name} ---")

    # Augmented Dickey-Fuller (ADF) Test
    # Null Hypothesis (H0): Non-stationary (has a unit root)
    adf_res = adfuller(series)
    print(f"ADF Statistic: {adf_res[0]:.4f}")
    print(f"ADF p-value:   {adf_res[1]:.4e}")
    if adf_res[1] < 0.05:
        print("  => ADF Result: Stationary (Reject H0 at 5% significance)")
    else:
        print(
            "  => ADF Result: Non-Stationary / Unit Root (Fail to reject H0)"
        )

    # KPSS Test
    # Null Hypothesis (H0): Stationary around a constant
    kpss_res = kpss(series, regression="c", nlags="auto")
    print(f"KPSS Statistic: {kpss_res[0]:.4f}")
    print(f"KPSS p-value:   {kpss_res[1]:.4e}")
    if kpss_res[1] < 0.05:
        print(
            "  => KPSS Result: Non-Stationary (Reject H0 at 5% significance)"
        )
    else:
        print("  => KPSS Result: Stationary (Fail to reject H0)")
