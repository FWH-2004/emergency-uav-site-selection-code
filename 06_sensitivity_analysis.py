# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
INPUT_FILE = r"C:\Users\22906\Desktop\expanded_100_points.xlsx"
P_LIST = [5, 10, 15, 20, 25, 30]
EARTH_RADIUS_KM = 6371.0
MAX_ITER = 500
EPS = 1e-6
TIME_LIMIT = 3600
OUTPUT_FLAG = 0
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
print("=" * 80)
print("数据读取完成")
print("=" * 80)
print(f"点位规模 n = {n}")
def build_distance_matrix_vectorized(df):
    lon = np.radians(df["经度"].to_numpy(dtype=float))
    lat = np.radians(df["纬度"].to_numpy(dtype=float))

    lon_i = lon[:, None]
    lon_j = lon[None, :]
    lat_i = lat[:, None]
    lat_j = lat[None, :]
    dlon = lon_j - lon_i
    dlat = lat_j - lat_i
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_i) * np.cos(lat_j) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c
D = build_distance_matrix_vectorized(df)
print("距离矩阵计算完成")
print(f"距离矩阵维度：{D.shape[0]} × {D.shape[1]}")
def global_trivial_lower_bound():
    two_smallest = np.partition(D, 1, axis=0)[:2, :]
    lb_values = two_smallest.sum(axis=0)
    return float(np.max(lb_values))
def get_selected_sites_from_vector(x_vec):
    return [i for i in range(n) if x_vec[i] > 0.5]
def evaluate_selected_solution(selected):
    selected = np.array(selected, dtype=int)
    subD = D[selected, :]
    part = np.partition(subD, 1, axis=0)
    d1 = part[0, :]
    d2 = part[1, :]
    costs = d1 + d2
    return costs
def analytic_dual_cut_coefficients(j, selected):
    selected = np.array(selected, dtype=int)
    selected_dist = D[selected, j]
    order = np.argsort(selected_dist)
    i1 = selected[order[0]]
    i2 = selected[order[1]]
    d1 = D[i1, j]
    d2 = D[i2, j]
    current_cost = d1 + d2
    alpha = d2
    beta = np.minimum(0.0, D[:, j] - d2)
    return float(current_cost), float(alpha), beta
def solve_by_benders(P):
    start_time = time.time()
    if P < 2:
        raise ValueError("P 必须不小于 2。")
    if P >= n:
        raise ValueError("P 必须小于点位总数。")
    base_lb = global_trivial_lower_bound()
    master = gp.Model(f"Benders_e_pCP_P{P}")
    master.Params.OutputFlag = OUTPUT_FLAG
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
    total_cuts = 0
    iter_count = 0
    for it in range(1, MAX_ITER + 1):
        iter_count = it
        master.optimize()
        if master.status == GRB.INFEASIBLE:
            raise RuntimeError(f"Benders 主问题不可行，P={P}")
        if master.SolCount == 0:
            raise RuntimeError(f"Benders 主问题未找到可行解，P={P}")
        if master.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            raise RuntimeError(f"Benders 主问题求解异常，P={P}，状态码：{master.status}")
        LB = float(master.ObjVal)
        x_bar = np.array([x[i].X for i in range(n)])
        selected = get_selected_sites_from_vector(x_bar)
        if len(selected) != P:
            raise RuntimeError(f"P={P} 当前选中站点数量异常：{len(selected)}")
        costs = evaluate_selected_solution(selected)
        current_obj = float(np.max(costs))
        if current_obj < UB:
            UB = current_obj
        z_val = float(z.X)
        cuts_added = 0
        for j in range(n):
            current_cost, alpha, beta = analytic_dual_cut_coefficients(j, selected)
            violation = current_cost - z_val
            if violation > EPS:
                negative_idx = np.where(beta < -1e-12)[0]
                expr = 2.0 * alpha
                expr += gp.quicksum(float(beta[i]) * x[int(i)] for i in negative_idx)
                master.addConstr(
                    z >= expr,
                    name=f"cut_p{P}_it{it}_j{j}"
                )
                cuts_added += 1
        total_cuts += cuts_added
        gap = (UB - LB) / max(1.0, abs(UB))
        elapsed = time.time() - start_time
        print(
            f"[Benders] P={P:>2} | Iter={it:>3} | "
            f"LB={LB:>10.6f} | UB={UB:>10.6f} | "
            f"Gap={gap*100:>8.4f}% | Cuts={cuts_added:>4} | "
            f"Time={elapsed:>8.2f}s"
        )
        if gap <= EPS and cuts_added == 0:
            break
        if elapsed >= TIME_LIMIT:
            print(f"[Benders] P={P} 达到时间限制。")
            break
    total_time = time.time() - start_time
    return {
        "求解方法": "Benders算法",
        "点位规模": n,
        "P": P,
        "最优值/km": UB,
        "求解时间/s": total_time,
        "迭代次数": iter_count,
        "Benders割数量": total_cuts,
        "最终Gap/%": (UB - LB) / max(1.0, abs(UB)) * 100.0
    }
def solve_by_gurobi_direct(P):
    start_time = time.time()
    if P < 2:
        raise ValueError("P 必须不小于 2。")
    if P >= n:
        raise ValueError("P 必须小于点位总数。")
    model = gp.Model(f"Gurobi_Direct_e_pCP_P{P}")
    model.Params.OutputFlag = OUTPUT_FLAG
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.MIPGap = 1e-6
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
            name=f"assign_two_sites_{j}"
        )
    for i in range(n):
        for j in range(n):
            model.addConstr(
                y[i, j] <= x[i],
                name=f"link_{i}_{j}"
            )
    for j in range(n):
        model.addConstr(
            z >= gp.quicksum(float(D[i, j]) * y[i, j] for i in range(n)),
            name=f"max_cover_{j}"
        )
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    total_time = time.time() - start_time
    if model.SolCount == 0:
        obj = None
        gap = None
    else:
        obj = float(model.ObjVal)
        gap = float(model.MIPGap) * 100 if model.status != GRB.OPTIMAL else 0.0
    return {
        "求解方法": "Gurobi",
        "点位规模": n,
        "P": P,
        "最优值/km": obj,
        "求解时间/s": total_time,
        "迭代次数": None,
        "Benders割数量": None,
        "最终Gap/%": gap
    }
all_results = []
print("\n" + "=" * 80)
print("开始敏感性分析：改变开放航空站数量 P")
print("=" * 80)
for P in P_LIST:
    print("\n" + "-" * 80)
    print(f"当前 P = {P}")
    print("-" * 80)
    benders_result = solve_by_benders(P)
    all_results.append(benders_result)
    gurobi_result = solve_by_gurobi_direct(P)
    all_results.append(gurobi_result)
    print(f"[汇总] P={P} | Benders={benders_result['最优值/km']:.6f} km, "
          f"Time={benders_result['求解时间/s']:.4f}s | "
          f"Gurobi={gurobi_result['最优值/km']:.6f} km, "
          f"Time={gurobi_result['求解时间/s']:.4f}s")
result_df = pd.DataFrame(all_results)

paper_table = result_df[[
    "求解方法",
    "点位规模",
    "P",
    "最优值/km",
    "求解时间/s"
]].copy()
paper_table["最优值/km"] = paper_table["最优值/km"].astype(float).round(6)
paper_table["求解时间/s"] = paper_table["求解时间/s"].astype(float).round(6)

print("\n" + "=" * 80)
print("敏感性分析结果表")
print("=" * 80)
print(paper_table.to_string(index=False))
