# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
INPUT_FILE = r"C:\Users\22906\Desktop\expanded_100_points.xlsx"
P = 20
EARTH_RADIUS_KM = 6371.0
MIP_GAP = 1e-9
TIME_LIMIT = 3600
GUROBI_OUTPUT = 1
df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.astype(str).str.strip()
required_cols = ["编号", "点位名称", "经度", "纬度"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Excel 文件缺少必要列：{missing_cols}")

df = df.copy()
df["编号"] = df["编号"].astype(int)
df = df.sort_values("编号").reset_index(drop=True)
n = len(df)
if P < 2:
    raise ValueError("椭圆覆盖模型至少需要选择2个航空站。")
if P >= n:
    raise ValueError("P 必须小于点位总数。")
print("=" * 80)
print("数据读取完成")
print("=" * 80)
print(f"点位规模 n = {n}")
print(f"开放航空站数量 p = {P}")
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2]
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c
def build_distance_matrix(df):
    n = len(df)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = haversine(
                df.loc[i, "经度"],
                df.loc[i, "纬度"],
                df.loc[j, "经度"],
                df.loc[j, "纬度"]
            )
    return D
D = build_distance_matrix(df)
print("\n距离矩阵计算完成。")
print(f"距离矩阵维度：{D.shape[0]} × {D.shape[1]}")
def site_id(i):
    return int(df.loc[i, "编号"])
def site_name(i):
    return str(df.loc[i, "点位名称"])
def print_selected_sites(selected):
    for i in selected:
        print(f"{site_id(i):>3}  {site_name(i)}")
def evaluate_solution(selected):
    records = []
    costs = np.zeros(n)
    for j in range(n):
        ordered = sorted(selected, key=lambda i: D[i, j])
        i1, i2 = ordered[0], ordered[1]
        d1 = D[i1, j]
        d2 = D[i2, j]
        cost = d1 + d2
        costs[j] = cost
        records.append({
            "需求点编号": site_id(j),
            "需求点名称": site_name(j),
            "服务航空站1编号": site_id(i1),
            "服务航空站1名称": site_name(i1),
            "服务航空站1距离_km": d1,
            "服务航空站2编号": site_id(i2),
            "服务航空站2名称": site_name(i2),
            "服务航空站2距离_km": d2,
            "双站覆盖距离和_km": cost
        })
    return costs, pd.DataFrame(records)
def solve_direct_gurobi():
    start_time = time.time()
    model = gp.Model("Direct_Gurobi_e_pCP")
    model.Params.OutputFlag = GUROBI_OUTPUT
    model.Params.MIPGap = MIP_GAP
    model.Params.TimeLimit = TIME_LIMIT
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")
    model.addConstr(
        gp.quicksum(x[i] for i in range(n)) == P,
        name="select_p_sites"
    )
    for j in range(n):
        model.addConstr(
            gp.quicksum(y[i, j] for i in range(n)) == 2,
            name=f"assign_two_sites_for_demand_{j}"
        )
    for i in range(n):
        for j in range(n):
            model.addConstr(
                y[i, j] <= x[i],
                name=f"open_link_{i}_{j}"
            )
    for j in range(n):
        model.addConstr(
            z >= gp.quicksum(D[i, j] * y[i, j] for i in range(n)),
            name=f"max_two_site_distance_sum_{j}"
        )
    model.setObjective(z, GRB.MINIMIZE)
    print("\n" + "=" * 80)
    print("开始使用 Gurobi 直接求解原始 e-pCP 模型")
    print("=" * 80)
    model.optimize()
    solve_time = time.time() - start_time
    if model.SolCount == 0:
        raise RuntimeError("Gurobi 未找到可行解。")
    selected = [i for i in range(n) if x[i].X > 0.5]
    costs, detail_df = evaluate_solution(selected)
    worst_j = int(np.argmax(costs))
    real_obj = float(np.max(costs))
    if model.status == GRB.OPTIMAL:
        status_text = "OPTIMAL"
    elif model.status == GRB.TIME_LIMIT:
        status_text = "TIME_LIMIT"
    else:
        status_text = str(model.status)
    result = {
        "算法": "Gurobi直接求解",
        "状态": status_text,
        "点位规模": n,
        "开放航空站数量p": P,
        "模型目标值_km": model.ObjVal,
        "校验目标值_km": real_obj,
        "下界Bound": model.ObjBound,
        "最终Gap_%": model.MIPGap * 100 if model.SolCount > 0 else None,
        "求解时间_s": solve_time,
        "变量数量": model.NumVars,
        "约束数量": model.NumConstrs,
        "节点数量": model.NodeCount,
        "选中站点索引": selected,
        "需求点覆盖成本": costs,
        "服务明细": detail_df,
        "最不利需求点索引": worst_j
    }
    return result
result = solve_direct_gurobi()
print("\n" + "=" * 80)
print("Gurobi 直接求解最终结果")
print("=" * 80)
print(f"算法：{result['算法']}")
print(f"求解状态：{result['状态']}")
print(f"点位规模：{result['点位规模']}")
print(f"开放航空站数量 p：{result['开放航空站数量p']}")
print(f"模型目标值：{result['模型目标值_km']:.6f} km")
print(f"校验目标值：{result['校验目标值_km']:.6f} km")
print(f"下界 Bound：{result['下界Bound']:.6f}")
print(f"最终 Gap：{result['最终Gap_%']:.6f}%")
print(f"求解时间：{result['求解时间_s']:.4f} s")
print(f"变量数量：{result['变量数量']}")
print(f"约束数量：{result['约束数量']}")
print(f"搜索节点数量：{result['节点数量']}")
print("\n选中航空站：")
print_selected_sites(result["选中站点索引"])
worst_j = result["最不利需求点索引"]
worst_row = result["服务明细"].iloc[worst_j]
print("\n最不利需求点：")
print(f"需求点编号：{worst_row['需求点编号']}")
print(f"需求点名称：{worst_row['需求点名称']}")
print(f"双站覆盖距离和：{worst_row['双站覆盖距离和_km']:.6f} km")
print(
    f"服务航空站1：{worst_row['服务航空站1编号']} {worst_row['服务航空站1名称']}，"
    f"距离 {worst_row['服务航空站1距离_km']:.6f} km"
)
print(
    f"服务航空站2：{worst_row['服务航空站2编号']} {worst_row['服务航空站2名称']}，"
    f"距离 {worst_row['服务航空站2距离_km']:.6f} km"
)
print("\n" + "=" * 80)
print("服务明细")
print("=" * 80)
print(result["服务明细"].round(6).to_string(index=False))
print("\n" + "=" * 80)
print("论文表格用结果")
print("=" * 80)
paper_table = pd.DataFrame([{
    "算法": result["算法"],
    "点位规模": result["点位规模"],
    "开放航空站数量p": result["开放航空站数量p"],
    "最优目标值/km": result["模型目标值_km"],
    "求解时间/s": result["求解时间_s"],
    "变量数量": result["变量数量"],
    "约束数量": result["约束数量"],
    "搜索节点数量": result["节点数量"],
    "最终Gap/%": result["最终Gap_%"]
}])
print(paper_table.round(6).to_string(index=False))
