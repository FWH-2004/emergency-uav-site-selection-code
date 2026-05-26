# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
INPUT_FILE = r"C:\Users\22906\Desktop\b1.xlsx"
OUTPUT_FILE = r"C:\Users\22906\Desktop\expanded_100_points.xlsx"
TARGET_N = 100
RANDOM_SEED = 42
LON_NOISE_STD = 0.035
LAT_NOISE_STD = 0.035
rng = np.random.default_rng(RANDOM_SEED)
df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.astype(str).str.strip()
required_cols = ["编号", "点位名称", "经度", "纬度"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Excel文件{missing_cols}")
df = df.copy()
df["编号"] = df["编号"].astype(int)
if "所属区域" not in df.columns:
    df["所属区域"] = "未知区域"
if "点位类型" not in df.columns:
    df["点位类型"] = "原始点位"
df = df.sort_values("编号").reset_index(drop=True)

original_n = len(df)

if original_n >= TARGET_N:
    raise ValueError("原始点位数量已经大于或等于目标数量，不需要扩展。")
print("=" * 80)
print("原始数据读取完成")
print("=" * 80)
print(f"原始点位数量：{original_n}")
print(df[["编号", "点位名称", "所属区域", "点位类型", "经度", "纬度"]].to_string(index=False))
need_generate = TARGET_N - original_n
synthetic_rows = []
lon_min, lon_max = df["经度"].min(), df["经度"].max()
lat_min, lat_max = df["纬度"].min(), df["纬度"].max()
lon_lower = lon_min - 0.05
lon_upper = lon_max + 0.05
lat_lower = lat_min - 0.05
lat_upper = lat_max + 0.05
for k in range(need_generate):
    base_idx = rng.integers(0, original_n)
    base = df.loc[base_idx]
    new_lon = base["经度"] + rng.normal(0, LON_NOISE_STD)
    new_lat = base["纬度"] + rng.normal(0, LAT_NOISE_STD)
    new_lon = float(np.clip(new_lon, lon_lower, lon_upper))
    new_lat = float(np.clip(new_lat, lat_lower, lat_upper))
    new_id = original_n + k + 1
    synthetic_rows.append({
        "编号": new_id,
        "点位名称": f"扩展点{new_id}",
        "所属区域": str(base["所属区域"]),
        "点位类型": "扩展模拟点",
        "经度": new_lon,
        "纬度": new_lat,
        "来源点编号": int(base["编号"]),
        "来源点名称": str(base["点位名称"])
    })
synthetic_df = pd.DataFrame(synthetic_rows)
original_df = df.copy()
original_df["来源点编号"] = original_df["编号"]
original_df["来源点名称"] = original_df["点位名称"]

expanded_df = pd.concat([original_df, synthetic_df], ignore_index=True)
expanded_df = expanded_df.sort_values("编号").reset_index(drop=True)
print("\n" + "=" * 80)
print("扩展数据生成完成")
print("=" * 80)
print(f"扩展后点位数量：{len(expanded_df)}")

print("\n扩展后前40个点位：")
print(
    expanded_df[
        ["编号", "点位名称", "所属区域", "点位类型", "经度", "纬度", "来源点编号", "来源点名称"]
    ].head(40).to_string(index=False)
)
print("\n经纬度范围检查：")
print(f"经度范围：{expanded_df['经度'].min():.6f} ~ {expanded_df['经度'].max():.6f}")
print(f"纬度范围：{expanded_df['纬度'].min():.6f} ~ {expanded_df['纬度'].max():.6f}")
print("\n各区域扩展后数量：")
print(expanded_df["所属区域"].value_counts().to_string())
