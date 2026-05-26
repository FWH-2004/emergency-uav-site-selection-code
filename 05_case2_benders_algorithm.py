# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
INPUT_FILE = r"C:\Users\22906\Desktop\expanded_100_points.xlsx"

P = 20
EARTH_RADIUS_KM = 6371.0

MAX_ITER = 500
EPS = 1e-6
TIME_LIMIT = 3600
MASTER_OUTPUT = 0
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
def get_selected_sites_from_vector(x_vec):
    return [i for i in range(n) if x_vec[i] > 0.5]
def print_selected_sites(selected):
    for i in selected:
        print(f"{site_id(i):>3}  {site_name(i)}")
def compute_two_nearest_for_all_demands(selected):
    if len(selected) < 2:
        raise ValueError("椭圆覆盖至少需要两个开放航空站。")
    costs = np.zeros(n)
    records = []
    for j in range(n):
        ordered = sorted(selected, key=lambda i: D[i, j])
        i1 = ordered[0]
        i2 = ordered[1]
        d1 = D[i1, j]
        d2 = D[i2, j]
        cost = d1 + d2
        costs[j] = cost
        records.append({
            "需求点索引": j,
            "需求点编号": site_id(j),
            "需求点名称": site_name(j),
            "服务站1索引": i1,
            "服务站1编号": site_id(i1),
            "服务站1名称": site_name(i1),
            "服务站1距离": d1,
            "服务站2索引": i2,
            "服务站2编号": site_id(i2),
            "服务站2名称": site_name(i2),
            "服务站2距离": d2,
            "双站距离和": cost
        })
    return costs, records
def global_trivial_lower_bound():
    lb_values = []
    for j in range(n):
        ordered = np.sort(D[:, j])
        lb_values.append(ordered[0] + ordered[1])
    return max(lb_values)
def analytic_dual_cut_coefficients(j, selected):
    ordered = sorted(selected, key=lambda i: D[i, j])
    i1 = ordered[0]
    i2 = ordered[1]
    d1 = D[i1, j]
    d2 = D[i2, j]
    alpha = d2
    beta = np.minimum(0.0, D[:, j] - d2)
    current_cost = d1 + d2
    return current_cost, alpha, beta, i1, i2, d1, d2
def classic_benders_fixed():
    start_time = time.time()
    base_lb = global_trivial_lower_bound()
    master = gp.Model("Classic_Benders_Master_Fixed")
    master.Params.OutputFlag = MASTER_OUTPUT
    master.Params.TimeLimit = TIME_LIMIT
    x = master.addVars(n, vtype=GRB.BINARY, name="x")
    z = master.addVar(lb=base_lb, vtype=GRB.CONTINUOUS, name="z")
    master.addConstr(
        gp.quicksum(x[i] for i in range(n)) == P,
        name="select_p_sites"
    )
    master.setObjective(z, GRB.MINIMIZE)
    UB = float("inf")
    LB = base_lb
    best_selected = None
    best_costs = None
    best_records = None
    total_cuts = 0
    iteration_log = []
    print("\n" + "=" * 80)
    print("开始运行修正后的经典 Benders 算法")
    print("=" * 80)
    print(f"基础下界 base LB = {base_lb:.6f}")
    for it in range(1, MAX_ITER + 1):
        master.optimize()
        if master.status == GRB.INFEASIBLE:
            raise RuntimeError("主问题不可行。")
        if master.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            raise RuntimeError(f"主问题求解异常，状态码：{master.status}")
        LB = master.ObjVal
        x_bar = np.array([x[i].X for i in range(n)])
        selected = get_selected_sites_from_vector(x_bar)

        if len(selected) != P:
            raise RuntimeError(f"当前主问题选中站点数量异常：{len(selected)}，应为 {P}")
        costs, records = compute_two_nearest_for_all_demands(selected)
        current_obj = float(np.max(costs))
        if current_obj < UB:
            UB = current_obj
            best_selected = selected.copy()
            best_costs = costs.copy()
            best_records = records.copy()
        z_val = z.X
        cuts_added = 0
        max_violation = 0.0
        for j in range(n):
            current_cost, alpha, beta, i1, i2, d1, d2 = analytic_dual_cut_coefficients(j, selected)
            violation = current_cost - z_val
            max_violation = max(max_violation, violation)
            if violation > EPS:
                master.addConstr(
                    z >= 2.0 * alpha + gp.quicksum(beta[i] * x[i] for i in range(n)),
                    name=f"benders_cut_it{it}_demand{j}"
                )
                cuts_added += 1
        total_cuts += cuts_added
        gap = (UB - LB) / max(1.0, abs(UB))
        elapsed = time.time() - start_time
        iteration_log.append({
            "迭代次数": it,
            "LB": LB,
            "UB": UB,
            "Gap": gap,
            "当前方案目标值": current_obj,
            "本轮新增割数量": cuts_added,
            "累计割数量": total_cuts,
            "最大违反度": max_violation,
            "当前已用时间_s": elapsed
        })
        print(
            f"Iter {it:>3} | "
            f"LB = {LB:>10.6f} | "
            f"UB = {UB:>10.6f} | "
            f"Gap = {gap * 100:>8.4f}% | "
            f"Cuts = {cuts_added:>4} | "
            f"Total Cuts = {total_cuts:>5} | "
            f"MaxViol = {max_violation:>10.6f} | "
            f"Time = {elapsed:>8.2f}s"
        )
        if gap <= EPS and cuts_added == 0:
            print("\n修正后的经典 Benders 算法收敛。")
            break
        if elapsed >= TIME_LIMIT:
            print("\n达到时间限制，算法停止。")
            break
    total_time = time.time() - start_time
    result = {
        "算法": "经典Benders-修正版",
        "点位规模": n,
        "开放航空站数量p": P,
        "最优目标值_km": UB,
        "下界LB": LB,
        "最终Gap_%": (UB - LB) / max(1.0, abs(UB)) * 100.0,
        "求解时间_s": total_time,
        "迭代次数": len(iteration_log),
        "Benders割数量": total_cuts,
        "选中站点索引": best_selected,
        "需求点覆盖成本": best_costs,
        "需求点服务记录": best_records,
        "迭代日志": pd.DataFrame(iteration_log)
    }
    return result
result = classic_benders_fixed()
print("\n" + "=" * 80)
print("修正后的经典 Benders 算法最终结果")
print("=" * 80)
print(f"算法：{result['算法']}")
print(f"点位规模：{result['点位规模']}")
print(f"开放航空站数量 p：{result['开放航空站数量p']}")
print(f"最优目标值：{result['最优目标值_km']:.6f} km")
print(f"下界 LB：{result['下界LB']:.6f}")
print(f"最终 Gap：{result['最终Gap_%']:.6f}%")
print(f"求解时间：{result['求解时间_s']:.4f} s")
print(f"迭代次数：{result['迭代次数']}")
print(f"Benders 割数量：{result['Benders割数量']}")
print("\n选中航空站：")
print_selected_sites(result["选中站点索引"])
worst_j = int(np.argmax(result["需求点覆盖成本"]))
worst_cost = result["需求点覆盖成本"][worst_j]
worst_record = result["需求点服务记录"][worst_j]
print("\n最不利需求点：")
print(f"需求点编号：{site_id(worst_j)}")
print(f"需求点名称：{site_name(worst_j)}")
print(f"覆盖距离和：{worst_cost:.6f} km")
print(
    f"服务航空站1：{worst_record['服务站1编号']} {worst_record['服务站1名称']}，"
    f"距离 {worst_record['服务站1距离']:.6f} km"
)
print(
    f"服务航空站2：{worst_record['服务站2编号']} {worst_record['服务站2名称']}，"
    f"距离 {worst_record['服务站2距离']:.6f} km"
)
print("\n" + "=" * 80)
print("迭代日志")
print("=" * 80)
print(result["迭代日志"].round(6).to_string(index=False))
print("\n" + "=" * 80)
print("=" * 80)
paper_table = pd.DataFrame([{
    "算法": result["算法"],
    "点位规模": result["点位规模"],
    "开放航空站数量p": result["开放航空站数量p"],
    "最优目标值/km": result["最优目标值_km"],
    "求解时间/s": result["求解时间_s"],
    "迭代次数": result["迭代次数"],
    "Benders割数量": result["Benders割数量"],
    "最终Gap/%": result["最终Gap_%"]
}])
print(paper_table.round(6).to_string(index=False))
