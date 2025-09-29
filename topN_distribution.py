import pandas as pd

def top_n_distribution(
    csv_path: str,
    column: str,
    n: int = 10,
    *,
    dropna: bool = True,
    bins: int | list | None = None,
    trim_whitespace: bool = True,
    casefold: bool = False,
    save_to: str | None = None,
):
    """
    统计某一列的数据分布并输出前 N 个。

    参数说明：
    - csv_path: CSV 文件路径
    - column: 列名
    - n: 取 Top-N
    - dropna: 是否丢弃空值
    - bins: 若该列为数值，可给 int（等宽分箱个数）或自定义分箱边界列表；None 表示不分箱
    - trim_whitespace: 对文本值去除首尾空格
    - casefold: 文本不区分大小写（更强的 lower()）
    - save_to: 若给出路径，则把结果另存为 CSV
    """
    # 读取
    df = pd.read_csv(csv_path, low_memory=False)

    if column not in df.columns:
        cols_preview = ", ".join(map(str, df.columns[:50]))
        raise KeyError(f"找不到列名：{column}。可用列示例：{cols_preview}")

    s = df[column]

    # 文本清洗
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if trim_whitespace:
            s = s.astype("string").str.strip()
        if casefold:
            s = s.astype("string").str.casefold()

    # 可选：数值分箱
    if bins is not None and pd.api.types.is_numeric_dtype(s):
        s = pd.cut(s, bins=bins, include_lowest=True)

    # 频数与占比
    counts_all = s.value_counts(dropna=dropna)
    if counts_all.empty:
        raise ValueError("该列没有可统计的数据（可能全为空或被过滤掉）。")

    top = counts_all.head(n)
    pct = (top / counts_all.sum()).round(6)

    result = (
        pd.DataFrame({column: top.index, "count": top.values, "pct": pct.values})
        .reset_index(drop=True)
    )

    # 打印展示
    print(result.to_string(index=False))

    # 可选保存
    if save_to:
        result.to_csv(save_to, index=False)
        print(f"\n已保存到：{save_to}")

    return result


if __name__ == "__main__":
    # === 使用示例 ===
    csv_path = "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.csv"  # 你的文件
    column_name = "family_id"        # TODO: 改成目标列名
    N = 20                                   # Top-N 个数

    # 基础用法：Top-N 频数与占比
    top_n_distribution(csv_path, column_name, n=N)

    # 进阶：对数值列做等宽 20 分箱后再取 Top-N（此时 Top-N 是出现次数最多的区间）
    # top_n_distribution(csv_path, column_name, n=N, bins=20)

    # 进阶：文本列去空格、忽略大小写，并保存结果
    # top_n_distribution(csv_path, column_name, n=N, trim_whitespace=True, casefold=True,
    #                    save_to="/mnt/data/topN_distribution.csv")
