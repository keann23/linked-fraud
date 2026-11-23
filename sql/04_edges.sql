-- Phase 4: link rebuilt identities (uids) that share an attribute value.
-- No labels are used here. {hub_max} is the hub-suppression threshold: any value
-- shared by more than that many uids (e.g. a popular phone model, gmail.com) is
-- dropped, since it links strangers rather than accomplices.

CREATE OR REPLACE TABLE uid_attr AS
WITH t AS (
    SELECT i.uid, b.*
    FROM base b JOIN identity i USING (TransactionID)
    WHERE i.uid IS NOT NULL
)
SELECT DISTINCT uid, 'device' AS link_type,
       concat_ws('|', DeviceInfo, id_30, id_31, id_33) AS val
FROM t WHERE DeviceInfo IS NOT NULL
UNION
-- every uid already contains card1, so two uids sharing a card1 differ in
-- billing region or account start date: same card, different "customer"
SELECT DISTINCT uid, 'card', CAST(card1 AS VARCHAR) FROM t
UNION
SELECT DISTINCT uid, 'email', P_emaildomain FROM t WHERE P_emaildomain IS NOT NULL
UNION
SELECT DISTINCT uid, 'email', R_emaildomain FROM t WHERE R_emaildomain IS NOT NULL;

CREATE OR REPLACE TABLE attr_values AS
SELECT link_type, val, COUNT(*) AS n_uids
FROM uid_attr GROUP BY ALL;

-- one row per linked uid pair, weight = sum over shared values of 1 / n_uids
CREATE OR REPLACE TABLE edges AS
WITH kept AS (
    SELECT a.uid, a.link_type, a.val, v.n_uids
    FROM uid_attr a JOIN attr_values v USING (link_type, val)
    WHERE v.n_uids BETWEEN 2 AND {hub_max}
)
SELECT
    x.uid AS src,
    y.uid AS dst,
    SUM(1.0 / x.n_uids)                                 AS weight,
    COUNT(*)                                            AS shared_values,
    MAX((x.link_type = 'device')::INT)::BOOLEAN         AS via_device,
    MAX((x.link_type = 'card')::INT)::BOOLEAN           AS via_card,
    MAX((x.link_type = 'email')::INT)::BOOLEAN          AS via_email
FROM kept x
JOIN kept y ON x.link_type = y.link_type AND x.val = y.val AND x.uid < y.uid
GROUP BY ALL;
