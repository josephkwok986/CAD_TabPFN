import pandas as pd

PI = "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.for_split.csv"  # 第1步生成的 for_split 表
df = pd.read_csv(PI)

g = (df.groupby('duplicate_canonical')
       .agg(n_rows=('part_id','size'),
            n_content=('content_hash','nunique'),
            n_geom=('geom_hash','nunique')))
top = g.sort_values('n_rows', ascending=False).head(20)
print("Top-20 duplicate_canonical by n_rows:")
print(top.to_string())

print("\nHow many canonical map to >1 content_hash ?")
print((g['n_content']>1).sum(), "/", len(g))
