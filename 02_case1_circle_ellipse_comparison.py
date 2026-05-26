# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
INPUT_FILE = r"C:\Users\22906\Desktop\b1.xlsx"
P = 6
EARTH_RADIUS_KM = 6371.0
GUROBI_OUTPUT = True
df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.astype(str).str.strip()
required_cols = ["编号", "点位名称", "经度", "纬度"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Excel文件中缺少必要列：{missing_cols}")
df = df.copy()
df["编号"] = df["编号"].astype(int)
df = df.sort_values("编号").reset_index(drop=True)
n = len(df)
if P >= n:
    raise ValueError("开放仓库数量P必须小于候选点总数。")
print("=" * 80)
print("数据读取完成")
print("=" * 80)
print(f"点位总数：{n}")
print(f"开放仓库数量 P：{P}")
print("\n点位列表：")
print(df[["编号", "点位名称", "经度", "纬度"]].to_string(index=False))
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
print("\n" + "=" * 80)
print("距离矩阵计算完成")
print("=" * 80)
print("距离矩阵前5行前5列，单位km：")
print(pd.DataFrame(
    D[:5, :5],
    index=df.loc[:4, "点位名称"],
    columns=df.loc[:4, "点位名称"]
).round(4))
def site_id(i):
    return int(df.loc[i, "编号"])
def site_name(i):
    return str(df.loc[i, "点位名称"])
def get_selected_sites(x_vars):
    return [i for i in range(n) if x_vars[i].X > 0.5]
def print_selected_sites(selected):
    for i in selected:
        print(f"{site_id(i):>2}  {site_name(i)}")
def format_selected_ids(selected):
    return "、".join(str(site_id(i)) for i in selected)
def format_selected_names(selected):
    return "、".join(site_name(i) for i in selected)
def evaluate_circle_solution(selected):
    records = []
    max_value = -1.0
    worst_j = None
    for j in range(n):
        nearest_i = min(selected, key=lambda i: D[i, j])
        one_way = D[nearest_i, j]
        round_trip = 2.0 * one_way
        if round_trip > max_value:
            max_value = round_trip
            worst_j = j
        records.append({
            "需求点编号": site_id(j),
            "需求点名称": site_name(j),
            "服务仓库编号": site_id(nearest_i),
            "服务仓库名称": site_name(nearest_i),
            "单程距离_km": one_way,
            "圆形覆盖距离_往返_km": round_trip
        })
    return max_value, worst_j, pd.DataFrame(records)
def evaluate_ellipse_solution(selected):
    if len(selected) < 2:
        raise ValueError("椭圆覆盖至少需要两个开放仓库。")
    records = []
    max_value = -1.0
    worst_j = None
    for j in range(n):
        ordered = sorted(selected, key=lambda i: D[i, j])
        i1 = ordered[0]
        i2 = ordered[1]
        d1 = D[i1, j]
        d2 = D[i2, j]
        ellipse_dist = d1 + d2
        if ellipse_dist > max_value:
            max_value = ellipse_dist
            worst_j = j
        records.append({
            "需求点编号": site_id(j),
            "需求点名称": site_name(j),
            "服务仓库1编号": site_id(i1),
            "服务仓库1名称": site_name(i1),
            "服务仓库1距离_km": d1,
            "服务仓库2编号": site_id(i2),
            "服务仓库2名称": site_name(i2),
            "服务仓库2距离_km": d2,
            "椭圆覆盖距离_两段和_km": ellipse_dist
        })
    return max_value, worst_j, pd.DataFrame(records)
def solve_circle_model():
    model = gp.Model("Circle_Coverage_Single_Roundtrip")
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z_circle")
    # 选择P个仓库
    model.addConstr(
        gp.quicksum(x[i] for i in range(n)) == P,
        name="select_P_sites"
    )
    for j in range(n):
        model.addConstr(
            gp.quicksum(y[i, j] for i in range(n)) == 1,
            name=f"assign_one_site_for_demand_{j}"
        )
    for i in range(n):
        for j in range(n):
            model.addConstr(
                y[i, j] <= x[i],
                name=f"link_open_assign_{i}_{j}"
            )
    for j in range(n):
        model.addConstr(
            z >= gp.quicksum(2.0 * D[i, j] * y[i, j] for i in range(n)),
            name=f"max_roundtrip_distance_{j}"
        )
    model.setObjective(z, GRB.MINIMIZE)
    model.Params.OutputFlag = 1 if GUROBI_OUTPUT else 0
    model.Params.MIPGap = 1e-9
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError("圆形覆盖模型未求得最优解。")
    selected = get_selected_sites(x)
    obj_value, worst_j, detail_df = evaluate_circle_solution(selected)

    return {
        "selected": selected,
        "obj": obj_value,
        "worst_j": worst_j,
        "detail": detail_df
    }
def solve_ellipse_model():
    model = gp.Model("Ellipse_Coverage_Two_Site")
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z_ellipse")
    # 选择P个仓库
    model.addConstr(
        gp.quicksum(x[i] for i in range(n)) == P,
        name="select_P_sites"
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
                name=f"link_open_assign_{i}_{j}"
            )
    for j in range(n):
        model.addConstr(
            z >= gp.quicksum(D[i, j] * y[i, j] for i in range(n)),
            name=f"max_two_site_distance_sum_{j}"
        )
    model.setObjective(z, GRB.MINIMIZE)
    model.Params.OutputFlag = 1 if GUROBI_OUTPUT else 0
    model.Params.MIPGap = 1e-9
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError("椭圆覆盖模型未求得最优解。")
    selected = get_selected_sites(x)
    obj_value, worst_j, detail_df = evaluate_ellipse_solution(selected)
    return {
        "selected": selected,
        "obj": obj_value,
        "worst_j": worst_j,
        "detail": detail_df
    }
def failure_analysis_circle(selected, base_obj):
    records = []
    for failed in selected:
        remaining = [i for i in selected if i != failed]
        fail_obj, worst_j, _ = evaluate_circle_solution(remaining)
        records.append({
            "失效仓库编号": site_id(failed),
            "失效仓库名称": site_name(failed),
            "失效后目标值_km": fail_obj,
            "较正常状态增加_km": fail_obj - base_obj,
            "增幅_%": (fail_obj - base_obj) / base_obj * 100.0,
            "失效后最不利需求点编号": site_id(worst_j),
            "失效后最不利需求点名称": site_name(worst_j)
        })
    return pd.DataFrame(records)
def failure_analysis_ellipse(selected, base_obj):
    records = []
    for failed in selected:
        remaining = [i for i in selected if i != failed]
        fail_obj, worst_j, _ = evaluate_ellipse_solution(remaining)
        records.append({
            "失效仓库编号": site_id(failed),
            "失效仓库名称": site_name(failed),
            "失效后目标值_km": fail_obj,
            "较正常状态增加_km": fail_obj - base_obj,
            "增幅_%": (fail_obj - base_obj) / base_obj * 100.0,
            "失效后最不利需求点编号": site_id(worst_j),
            "失效后最不利需求点名称": site_name(worst_j)
        })
    return pd.DataFrame(records)
def summarize_failure(failure_df, base_obj):
    return {
        "正常状态目标值_km": base_obj,
        "单点失效后平均目标值_km": failure_df["失效后目标值_km"].mean(),
        "单点失效后最大目标值_km": failure_df["失效后目标值_km"].max(),
        "平均增加值_km": failure_df["较正常状态增加_km"].mean(),
        "最大增加值_km": failure_df["较正常状态增加_km"].max(),
        "平均增幅_%": failure_df["增幅_%"].mean(),
        "最大增幅_%": failure_df["增幅_%"].max()
    }
circle_result = solve_circle_model()
ellipse_result = solve_ellipse_model()

circle_failure_df = failure_analysis_circle(
    circle_result["selected"],
    circle_result["obj"]
)
ellipse_failure_df = failure_analysis_ellipse(
    ellipse_result["selected"],
    ellipse_result["obj"]
)
circle_summary = summarize_failure(circle_failure_df, circle_result["obj"])
ellipse_summary = summarize_failure(ellipse_failure_df, ellipse_result["obj"])
summary_df = pd.DataFrame([
    {"覆盖模式": "圆形覆盖", **circle_summary},
    {"覆盖模式": "椭圆覆盖", **ellipse_summary}
])
print("\n" + "=" * 80)
print("正常状态下模型结果")
print("=" * 80)
print("\n【圆形覆盖模型：单站往返】")
print(f"最优目标值 Z_circle = {circle_result['obj']:.6f} km")
print(f"最不利需求点：{site_id(circle_result['worst_j'])}  {site_name(circle_result['worst_j'])}")
print("选中仓库：")
print_selected_sites(circle_result["selected"])

print("\n【椭圆覆盖模型：双站协同、异站起降】")
print(f"最优目标值 Z_ellipse = {ellipse_result['obj']:.6f} km")
print(f"最不利需求点：{site_id(ellipse_result['worst_j'])}  {site_name(ellipse_result['worst_j'])}")
print("选中仓库：")
print_selected_sites(ellipse_result["selected"])
