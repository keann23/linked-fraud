-- Phase 5: network features for every transaction, computed "as of" its date.
--
-- Two timing rules keep this honest:
--   1. An edge only counts once both identities have actually shown the shared
--      value (edge_day <= transaction day). No links from the future.
--   2. Neighbor labels are only those already known at a cutoff:
--        test rows  (day > 120): labels up to day 120, the end of training
--        train rows (day <= 120): labels up to day - 30, so training sees a
--        label delay similar to test (where the delay is 1 to 62 days)
--   The identity's OWN labels are never used (that would be uid label leakage).
--
-- Placeholders: {hub_max}, {train_last_day}, {label_lag}

CREATE OR REPLACE TABLE uid_attr_t AS
WITH t AS (
    SELECT i.uid, b.*
    FROM base b JOIN identity i USING (TransactionID)
    WHERE i.uid IS NOT NULL
)
SELECT uid, link_type, val, MIN(day) AS first_day FROM (
    SELECT uid, 'device' AS link_type, concat_ws('|', DeviceInfo, id_30, id_31, id_33) AS val, day
    FROM t WHERE DeviceInfo IS NOT NULL
    UNION ALL SELECT uid, 'card',  CAST(card1 AS VARCHAR), day FROM t
    UNION ALL SELECT uid, 'email', P_emaildomain, day FROM t WHERE P_emaildomain IS NOT NULL
    UNION ALL SELECT uid, 'email', R_emaildomain, day FROM t WHERE R_emaildomain IS NOT NULL
) GROUP BY ALL;

-- directed neighbor list with the day each link became visible
CREATE OR REPLACE TABLE nbr_edges AS
WITH vals AS (
    SELECT link_type, val FROM uid_attr_t GROUP BY ALL
    HAVING COUNT(*) BETWEEN 2 AND {hub_max}
),
kept AS (SELECT a.* FROM uid_attr_t a JOIN vals USING (link_type, val))
SELECT x.uid, y.uid AS nbr,
       MIN(GREATEST(x.first_day, y.first_day))                                   AS edge_day,
       MIN(GREATEST(x.first_day, y.first_day)) FILTER (WHERE x.link_type = 'device') AS device_day,
       MIN(GREATEST(x.first_day, y.first_day)) FILTER (WHERE x.link_type = 'card')   AS card_day
FROM kept x JOIN kept y ON x.link_type = y.link_type AND x.val = y.val AND x.uid <> y.uid
GROUP BY ALL;

-- running label totals per identity per day
CREATE OR REPLACE TABLE uid_cum AS
SELECT uid, day,
       SUM(COUNT(*))     OVER w AS cum_txns,
       SUM(SUM(isFraud)) OVER w AS cum_fraud
FROM identity WHERE uid IS NOT NULL
GROUP BY uid, day
WINDOW w AS (PARTITION BY uid ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

CREATE OR REPLACE TABLE keys AS
SELECT DISTINCT uid, day,
       CASE WHEN day > {train_last_day} THEN {train_last_day} ELSE day - {label_lag} END AS cutoff
FROM identity WHERE uid IS NOT NULL;

-- direct neighbors, as of each (uid, day)
CREATE OR REPLACE TABLE nbr_feats AS
WITH n AS (
    SELECT k.uid, k.day, k.cutoff, e.nbr,
           (e.device_day <= k.day) AS dev, (e.card_day <= k.day) AS crd
    FROM keys k JOIN nbr_edges e ON e.uid = k.uid AND e.edge_day <= k.day
),
lab AS (
    SELECT n.*, c.cum_txns, c.cum_fraud
    FROM n ASOF LEFT JOIN uid_cum c ON c.uid = n.nbr AND c.day <= n.cutoff
)
SELECT uid, day,
       COUNT(*)                                        AS net_degree,
       COUNT(*) FILTER (WHERE dev)                     AS net_device_degree,
       COUNT(*) FILTER (WHERE crd)                     AS net_card_degree,
       COUNT(*) FILTER (WHERE cum_fraud > 0)           AS nbr_fraud_uids,
       COALESCE(SUM(cum_txns), 0)                      AS nbr_known_txns,
       COALESCE(SUM(cum_fraud), 0)                     AS nbr_fraud_txns,
       SUM(cum_fraud) / NULLIF(SUM(cum_txns), 0)       AS nbr_fraud_rate,
       COUNT(*) FILTER (WHERE dev AND cum_fraud > 0)   AS nbr_device_fraud_uids
FROM lab GROUP BY ALL;

-- whole cluster (Louvain group), as of each (uid, day); cluster shape is label-free
CREATE OR REPLACE TABLE clu_feats AS
WITH m AS (
    SELECT k.uid, k.day, k.cutoff, c2.uid AS member
    FROM keys k
    JOIN clusters c1 ON c1.uid = k.uid
    JOIN clusters c2 ON c2.cluster_id = c1.cluster_id AND c2.uid <> k.uid
),
lab AS (
    SELECT m.*, c.cum_txns, c.cum_fraud
    FROM m ASOF LEFT JOIN uid_cum c ON c.uid = m.member AND c.day <= m.cutoff
)
SELECT uid, day,
       COUNT(*) FILTER (WHERE cum_fraud > 0)       AS clu_fraud_uids,
       SUM(cum_fraud) / NULLIF(SUM(cum_txns), 0)   AS clu_fraud_rate
FROM lab GROUP BY ALL;

CREATE OR REPLACE TABLE network_features AS
SELECT i.TransactionID,
       COALESCE(n.net_degree, 0)            AS net_degree,
       COALESCE(n.net_device_degree, 0)     AS net_device_degree,
       COALESCE(n.net_card_degree, 0)       AS net_card_degree,
       COALESCE(n.nbr_fraud_uids, 0)        AS nbr_fraud_uids,
       COALESCE(n.nbr_known_txns, 0)        AS nbr_known_txns,
       COALESCE(n.nbr_fraud_txns, 0)        AS nbr_fraud_txns,
       n.nbr_fraud_rate,
       COALESCE(n.nbr_device_fraud_uids, 0) AS nbr_device_fraud_uids,
       COALESCE(cl.cluster_size, 1)         AS clu_size,
       COALESCE(f.clu_fraud_uids, 0)        AS clu_fraud_uids,
       f.clu_fraud_rate
FROM identity i
LEFT JOIN nbr_feats n  ON n.uid = i.uid AND n.day = i.day
LEFT JOIN clusters cl  ON cl.uid = i.uid
LEFT JOIN clu_feats f  ON f.uid = i.uid AND f.day = i.day
ORDER BY i.TransactionID;
