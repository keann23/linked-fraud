-- Phase 3: velocity features. Every window looks backward only (past + current
-- transaction), so nothing here can see the future. No labels are used.
-- Rows without a uid (0.2%, missing D1) get NULL uid features.
CREATE OR REPLACE TABLE velocity AS
WITH t AS (
    SELECT
        b.TransactionID,
        b.TransactionDT AS ts,
        b.TransactionAmt AS amt,
        b.card1,
        i.uid,
        b.P_emaildomain,
        NULLIF(concat_ws('|', b.DeviceInfo, b.id_30, b.id_31, b.id_33), '') AS device_fp
    FROM base b
    JOIN identity i USING (TransactionID)
),
card AS (
    SELECT
        TransactionID,
        COUNT(*) OVER w1h  AS card1_txn_1h,
        COUNT(*) OVER w24h AS card1_txn_24h,
        COUNT(*) OVER w7d  AS card1_txn_7d,
        ts - LAG(ts) OVER (PARTITION BY card1 ORDER BY ts, TransactionID) AS card1_secs_since_prev
    FROM t
    WINDOW
        w1h  AS (PARTITION BY card1 ORDER BY ts RANGE BETWEEN 3600   PRECEDING AND CURRENT ROW),
        w24h AS (PARTITION BY card1 ORDER BY ts RANGE BETWEEN 86400  PRECEDING AND CURRENT ROW),
        w7d  AS (PARTITION BY card1 ORDER BY ts RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW)
),
u AS (
    SELECT
        TransactionID,
        COUNT(*) OVER w1h  AS uid_txn_1h,
        COUNT(*) OVER w24h AS uid_txn_24h,
        COUNT(*) OVER w7d  AS uid_txn_7d,
        ROW_NUMBER() OVER wall AS uid_txn_number,
        ts - LAG(ts) OVER wall AS uid_secs_since_prev,
        COUNT(DISTINCT P_emaildomain) OVER wcum AS uid_distinct_emails,
        COUNT(DISTINCT device_fp)     OVER wcum AS uid_distinct_devices,
        AVG(amt) OVER wprior AS uid_prior_avg_amt,
        amt / NULLIF(AVG(amt) OVER wprior, 0) AS uid_amt_vs_avg
    FROM t
    WHERE uid IS NOT NULL
    WINDOW
        w1h    AS (PARTITION BY uid ORDER BY ts RANGE BETWEEN 3600   PRECEDING AND CURRENT ROW),
        w24h   AS (PARTITION BY uid ORDER BY ts RANGE BETWEEN 86400  PRECEDING AND CURRENT ROW),
        w7d    AS (PARTITION BY uid ORDER BY ts RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW),
        wall   AS (PARTITION BY uid ORDER BY ts, TransactionID),
        wcum   AS (PARTITION BY uid ORDER BY ts, TransactionID ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
        wprior AS (PARTITION BY uid ORDER BY ts, TransactionID ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT *
FROM card
LEFT JOIN u USING (TransactionID)
ORDER BY TransactionID;
