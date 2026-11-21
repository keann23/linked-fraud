-- Phase 1: one row per transaction, identity left-joined on, plus a day index.
-- TransactionDT is seconds from an unknown reference point; day 1 starts at 86400.
CREATE OR REPLACE TABLE base AS
SELECT
    t.*,
    CAST(FLOOR(t.TransactionDT / 86400) AS INTEGER) AS day,
    i.* EXCLUDE (TransactionID),
    (i.TransactionID IS NOT NULL) AS has_identity
FROM read_csv_auto('{raw}/train_transaction.csv', sample_size = -1) AS t
LEFT JOIN read_csv_auto('{raw}/train_identity.csv', sample_size = -1) AS i
    USING (TransactionID);
