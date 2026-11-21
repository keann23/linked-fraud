-- Phase 2: rebuild hidden customer identities.
-- D1 = days since the card/account started, measured at transaction time.
-- So day - D1 is the start date itself, which should stay fixed for one customer.
-- addr1 is missing on ~11% of rows; we keep those as an explicit 'na' token
-- rather than dropping them. Rows with no D1 (0.2%) get no uid.
CREATE OR REPLACE TABLE identity AS
SELECT
    TransactionID,
    day,
    isFraud,
    TransactionAmt,
    card1,
    CAST(day - D1 AS INTEGER)                                        AS D1n,
    CAST(card1 AS VARCHAR) AS uid_card1,
    CASE WHEN D1 IS NOT NULL THEN
        CAST(card1 AS VARCHAR) || '_' ||
        COALESCE(CAST(CAST(addr1 AS INTEGER) AS VARCHAR), 'na') || '_' ||
        CAST(CAST(day - D1 AS INTEGER) AS VARCHAR)
    END                                                              AS uid,
    CASE WHEN D1 IS NOT NULL THEN
        CAST(card1 AS VARCHAR) || '_' ||
        COALESCE(CAST(CAST(addr1 AS INTEGER) AS VARCHAR), 'na') || '_' ||
        CAST(CAST(day - D1 AS INTEGER) AS VARCHAR) || '_' ||
        COALESCE(P_emaildomain, 'na')
    END                                                              AS uid_email
FROM base;
