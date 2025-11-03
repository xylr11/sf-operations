import polars as pl
import datetime as dt

def add_signals(df: pl.DataFrame, IC=0.05):
    return (
        df.lazy()
        .sort(["barrid", "date"])
        .with_columns([ # Convert nasty percents to nice fractions
            pl.col('specific_risk').truediv(100),
            pl.col('return').truediv(100),
            pl.col('specific_return').truediv(100)
        ])
        .with_columns(
            pl.col('return').log1p().alias('log_return')
        )
        .with_columns(
            pl.col("log_return")
                .rolling_sum(230)
                .over("barrid")
                .alias("momentum_temp")
        )
        .with_columns(
            pl.col("momentum_temp").shift(22).over("barrid").alias("momentum")
        )
        .with_columns(
            pl.col("log_return")
                .rolling_sum(22)
                .over("barrid")
                .alias("meanrev_temp")
        )
        .with_columns(
            (-pl.col("meanrev_temp")).alias("meanrev")
        )
        .with_columns(
            (-pl.col("predicted_beta")).alias("bab")
        )
        .with_columns([ # Add signal z-scores
            ((pl.col("momentum") - pl.col("momentum").mean().over("date")) 
        / pl.col("momentum").std().over("date")).alias("momentum_z"),
            ((pl.col("meanrev") - pl.col("meanrev").mean().over("date")) 
        / pl.col("meanrev").std().over("date")).alias("meanrev_z"),
            ((pl.col("bab") - pl.col("bab").mean().over("date")) 
        / pl.col("bab").std().over("date")).alias("bab_z")
        ])
        .with_columns([ # Add signal alphas, using alpha = IC * specific_risk * z-score
            (IC * pl.col("specific_risk") * pl.col("momentum_z")).alias("momentum_alpha"),
            (IC * pl.col("specific_risk") * pl.col("meanrev_z")).alias("meanrev_alpha"),
            (IC * pl.col("specific_risk") * pl.col("bab_z")).alias("bab_alpha")
        ])
        .drop(["momentum_temp", "meanrev_temp"])
        .collect()
    )