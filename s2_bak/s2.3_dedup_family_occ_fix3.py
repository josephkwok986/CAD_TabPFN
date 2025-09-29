
import os, random, pandas as pd

PI = "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.for_split.csv"   # 第1步生成
OUT_ROOT = "/workspace/Gjj Local/data/CAD/step_out/s2_temp_out/splits/LODO"
DOMAIN_COL = "source_dataset"
CALIB_FRAC = 0.1
SEED = 42

class DSU:
    def __init__(self, n): self.p=list(range(n)); self.r=[0]*n
    def f(self,x):
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def u(self,a,b):
        ra,rb=self.f(a),self.f(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: ra,rb=rb,ra
        self.p[rb]=ra
        if self.r[ra]==self.r[rb]: self.r[ra]+=1

def build_components(df):
    n=len(df); dsu=DSU(n)
    fam_map={}; ch_map={}
    for i,(fam,ch) in enumerate(zip(df["family_id"], df["content_hash"])):
        fam_map.setdefault(fam, []).append(i)
        if isinstance(ch,str) and len(ch)>0:
            ch_map.setdefault(ch, []).append(i)
    # family 连边
    for lst in fam_map.values():
        if len(lst)>1:
            b=lst[0]
            for j in lst[1:]: dsu.u(b,j)
    # content_hash 连边（真正去重）
    for lst in ch_map.values():
        if len(lst)>1:
            b=lst[0]
            for j in lst[1:]: dsu.u(b,j)
    return [dsu.f(i) for i in range(n)]

def write_ids(path, ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for x in ids: f.write(str(x)+"\n")

def main():
    df=pd.read_csv(PI).reset_index(drop=True)
    rng=random.Random(SEED)
    df["comp"]=build_components(df)

    for dom in sorted(df[DOMAIN_COL].unique()):
        sub_out=os.path.join(OUT_ROOT, f"{DOMAIN_COL}={dom}")
        test=df[df[DOMAIN_COL]==dom]
        train_cand=df[df[DOMAIN_COL]!=dom]
        banned=set(test["comp"].unique().tolist())
        train_cand=train_cand[~train_cand["comp"].isin(banned)]

        ids=train_cand["part_id"].tolist()
        rng.shuffle(ids)
        cut=int(len(ids)*CALIB_FRAC)
        calib_ids=set(ids[:cut])
        train_ids=[x for x in ids if x not in calib_ids]
        test_ids=test["part_id"].tolist()

        write_ids(os.path.join(sub_out,"train.txt"), train_ids)
        write_ids(os.path.join(sub_out,"calib.txt"), list(calib_ids))
        write_ids(os.path.join(sub_out,"test.txt"), test_ids)
        print(f"[LODO] hold {DOMAIN_COL}={dom}: train={len(train_ids)}, calib={len(calib_ids)}, test={len(test_ids)}, banned_comps={len(banned)}")

if __name__=="__main__":
    main()
