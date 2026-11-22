"""Is the velocity result real or seed noise? Retrain each variant with 3 seeds."""
import sys; sys.path.insert(0, 'src')
import pandas as pd, model
df = model.load_frame(); train, test = model.split(df)
rows = []
for seed in [1, 2, 3]:
    model.PARAMS['seed'] = seed
    for name, vel in [('transaction only', False), ('+ velocity', True)]:
        cols = model.feature_cols(df, velocity=vel)
        m = model.fit(train, cols)
        r = model.evaluate(test, m.predict(test[cols])); r.update(seed=seed, variant=name)
        rows.append(r); print(r, flush=True)
out = pd.DataFrame(rows)
print(out.groupby('variant')[['auc', 'recall_at_budget', 'fraud_amt_caught']].agg(['mean', 'std']).round(4).to_string())
out.to_csv('data/processed/seed_check.csv', index=False)
