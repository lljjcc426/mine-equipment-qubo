from __future__ import annotations

import json
from pathlib import Path

import pulp

from data import (
    FUEL_PRICE_YUAN_PER_L,
    ORE_PRICE_YUAN_PER_M3,
    QUESTION_1_PROFIT_WAN,
    QUESTION_2_EXCAVATORS,
    QUESTION_2_MATCH,
    QUESTION_2_TRUCKS,
    QUESTION_3_EXCAVATORS,
    QUESTION_3_MATCH,
    QUESTION_3_TRUCKS,
    WORK_DAYS_PER_MONTH,
    WORK_HOURS_PER_DAY,
    YEARS,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
HOURS_TOTAL = WORK_DAYS_PER_MONTH * WORK_HOURS_PER_DAY * 12 * YEARS
MONTHS_TOTAL = 12 * YEARS
YUAN_PER_WAN = 10000


def excavator_fixed_cost_wan(excavator):
    monthly = excavator.labor_yuan_per_month + excavator.maintenance_yuan_per_month
    return excavator.purchase_wan + monthly * MONTHS_TOTAL / YUAN_PER_WAN


def excavator_active_margin_wan(excavator):
    revenue = (
        excavator.efficiency_m3_per_hour
        * HOURS_TOTAL
        * ORE_PRICE_YUAN_PER_M3
        / YUAN_PER_WAN
    )
    fuel = (
        excavator.fuel_l_per_hour
        * HOURS_TOTAL
        * FUEL_PRICE_YUAN_PER_L
        / YUAN_PER_WAN
    )
    return revenue - fuel


def truck_active_cost_wan(truck):
    fuel = truck.fuel_l_per_hour * HOURS_TOTAL * FUEL_PRICE_YUAN_PER_L / YUAN_PER_WAN
    monthly = truck.labor_yuan_per_month + truck.maintenance_yuan_per_month
    return fuel + monthly * MONTHS_TOTAL / YUAN_PER_WAN


def solve_question_1():
    excavators = QUESTION_2_EXCAVATORS
    budget_wan = 2400
    problem = pulp.LpProblem("question_1", pulp.LpMaximize)
    x = {
        exc.code: pulp.LpVariable(
            f"x_{exc.code}",
            lowBound=0,
            upBound=int(budget_wan // exc.purchase_wan),
            cat="Integer",
        )
        for exc in excavators
    }
    use = {exc.code: pulp.LpVariable(f"use_{exc.code}", cat="Binary") for exc in excavators}

    problem += pulp.lpSum(QUESTION_1_PROFIT_WAN[exc.code] * x[exc.code] for exc in excavators)
    problem += pulp.lpSum(exc.purchase_wan * x[exc.code] for exc in excavators) <= budget_wan
    problem += pulp.lpSum(use.values()) >= 3
    for exc in excavators:
        max_count = int(budget_wan // exc.purchase_wan)
        problem += x[exc.code] >= use[exc.code]
        problem += x[exc.code] <= max_count * use[exc.code]

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Question 1 not solved optimally: {pulp.LpStatus[status]}")

    counts = {exc.code: int(round(x[exc.code].value() or 0)) for exc in excavators}
    total_purchase = sum(exc.purchase_wan * counts[exc.code] for exc in excavators)
    total_profit = sum(QUESTION_1_PROFIT_WAN[exc.code] * counts[exc.code] for exc in excavators)
    return {
        "status": pulp.LpStatus[status],
        "budget_wan": budget_wan,
        "total_purchase_wan": total_purchase,
        "total_profit_wan": total_profit,
        "counts": counts,
    }


def solve_operation_question(name, excavators, trucks, match_rules, budget_wan, min_types):
    problem = pulp.LpProblem(name, pulp.LpMaximize)
    exc_by_code = {item.code: item for item in excavators}
    truck_by_code = {item.code: item for item in trucks}
    allowed_pairs = [(e, t) for (e, t) in match_rules if e in exc_by_code and t in truck_by_code]

    x = {}
    use = {}
    pair_use = {}
    assigned = {}
    effective = {}

    for exc in excavators:
        max_count = int(budget_wan // exc.purchase_wan)
        x[exc.code] = pulp.LpVariable(
            f"x_{exc.code}",
            lowBound=0,
            upBound=max_count,
            cat="Integer",
        )
        use[exc.code] = pulp.LpVariable(f"use_{exc.code}", cat="Binary")
        problem += x[exc.code] >= use[exc.code]
        problem += x[exc.code] <= max_count * use[exc.code]

    for exc_code, truck_code in allowed_pairs:
        truck = truck_by_code[truck_code]
        exc = exc_by_code[exc_code]
        max_count = int(budget_wan // exc.purchase_wan)
        pair_use[(exc_code, truck_code)] = pulp.LpVariable(
            f"pair_{exc_code}_{truck_code}",
            cat="Binary",
        )
        assigned[(exc_code, truck_code)] = pulp.LpVariable(
            f"assigned_{exc_code}_{truck_code}",
            lowBound=0,
            upBound=truck.available,
            cat="Integer",
        )
        effective[(exc_code, truck_code)] = pulp.LpVariable(
            f"effective_{exc_code}_{truck_code}",
            lowBound=0,
            upBound=max_count,
            cat="Continuous",
        )

        ratio = match_rules[(exc_code, truck_code)]
        problem += assigned[(exc_code, truck_code)] <= truck.available * pair_use[(exc_code, truck_code)]
        problem += assigned[(exc_code, truck_code)] >= pair_use[(exc_code, truck_code)]
        problem += effective[(exc_code, truck_code)] <= x[exc_code]
        problem += ratio * effective[(exc_code, truck_code)] <= assigned[(exc_code, truck_code)]
        problem += effective[(exc_code, truck_code)] <= max_count * pair_use[(exc_code, truck_code)]

    for exc in excavators:
        pair_vars = [pair_use[(exc.code, truck.code)] for truck in trucks if (exc.code, truck.code) in pair_use]
        if pair_vars:
            problem += pulp.lpSum(pair_vars) == use[exc.code]
        else:
            problem += use[exc.code] == 0
            problem += x[exc.code] == 0

    for truck in trucks:
        truck_assignments = [
            assigned[(exc.code, truck.code)]
            for exc in excavators
            if (exc.code, truck.code) in assigned
        ]
        if truck_assignments:
            problem += pulp.lpSum(truck_assignments) <= truck.available

    problem += pulp.lpSum(exc.purchase_wan * x[exc.code] for exc in excavators) <= budget_wan
    problem += pulp.lpSum(use.values()) >= min_types

    objective_terms = []
    for exc in excavators:
        objective_terms.append(-excavator_fixed_cost_wan(exc) * x[exc.code])
    for exc_code, truck_code in allowed_pairs:
        exc = exc_by_code[exc_code]
        truck = truck_by_code[truck_code]
        objective_terms.append(excavator_active_margin_wan(exc) * effective[(exc_code, truck_code)])
        objective_terms.append(-truck_active_cost_wan(truck) * assigned[(exc_code, truck_code)])
    problem += pulp.lpSum(objective_terms)

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"{name} not solved optimally: {pulp.LpStatus[status]}")

    counts = {exc.code: int(round(x[exc.code].value() or 0)) for exc in excavators}
    selected_types = [code for code, value in counts.items() if value > 0]
    assignments = []
    total_volume = 0.0
    revenue_wan = 0.0
    exc_fuel_wan = 0.0
    truck_fuel_wan = 0.0
    exc_fixed_operating_wan = 0.0
    truck_fixed_operating_wan = 0.0
    purchase_wan = 0.0

    for exc in excavators:
        count = counts[exc.code]
        if count <= 0:
            continue
        purchase_wan += exc.purchase_wan * count
        exc_fixed_operating_wan += (
            (exc.labor_yuan_per_month + exc.maintenance_yuan_per_month)
            * MONTHS_TOTAL
            / YUAN_PER_WAN
            * count
        )

    for (exc_code, truck_code), value in assigned.items():
        truck_count = int(round(value.value() or 0))
        active_equivalent = float(effective[(exc_code, truck_code)].value() or 0.0)
        if truck_count <= 0:
            continue
        exc = exc_by_code[exc_code]
        truck = truck_by_code[truck_code]
        ratio = match_rules[(exc_code, truck_code)]
        volume = exc.efficiency_m3_per_hour * HOURS_TOTAL * active_equivalent
        total_volume += volume
        revenue_wan += volume * ORE_PRICE_YUAN_PER_M3 / YUAN_PER_WAN
        exc_fuel_wan += (
            exc.fuel_l_per_hour
            * HOURS_TOTAL
            * active_equivalent
            * FUEL_PRICE_YUAN_PER_L
            / YUAN_PER_WAN
        )
        truck_fuel_wan += (
            truck.fuel_l_per_hour
            * HOURS_TOTAL
            * truck_count
            * FUEL_PRICE_YUAN_PER_L
            / YUAN_PER_WAN
        )
        truck_fixed_operating_wan += (
            (truck.labor_yuan_per_month + truck.maintenance_yuan_per_month)
            * MONTHS_TOTAL
            / YUAN_PER_WAN
            * truck_count
        )
        assignments.append(
            {
                "excavator": exc_code,
                "truck": truck_code,
                "excavator_count": counts[exc_code],
                "truck_count": truck_count,
                "required_ratio": ratio,
                "effective_active_equivalent": round(active_equivalent, 4),
                "utilization_ratio": round(active_equivalent / counts[exc_code], 4),
                "five_year_output_m3": round(volume, 2),
            }
        )

    total_profit_wan = (
        revenue_wan
        - purchase_wan
        - exc_fixed_operating_wan
        - exc_fuel_wan
        - truck_fixed_operating_wan
        - truck_fuel_wan
    )

    return {
        "status": pulp.LpStatus[status],
        "budget_wan": budget_wan,
        "min_types": min_types,
        "selected_types": selected_types,
        "counts": counts,
        "assignments": assignments,
        "summary": {
            "purchase_wan": round(purchase_wan, 2),
            "exc_fixed_operating_wan": round(exc_fixed_operating_wan, 2),
            "exc_fuel_wan": round(exc_fuel_wan, 2),
            "truck_fixed_operating_wan": round(truck_fixed_operating_wan, 2),
            "truck_fuel_wan": round(truck_fuel_wan, 2),
            "revenue_wan": round(revenue_wan, 2),
            "total_profit_wan": round(total_profit_wan, 2),
            "total_output_m3": round(total_volume, 2),
        },
    }


def format_question_1(result):
    lines = [
        "## 问题1",
        "",
        f"- 状态：{result['status']}",
        f"- 总采购成本：{result['total_purchase_wan']} 万元 / 预算 {result['budget_wan']} 万元",
        f"- 最大长期利润折现值：{result['total_profit_wan']} 万元",
        "- 最优采购数量：",
    ]
    for code, count in result["counts"].items():
        lines.append(f"  - {code}: {count} 台")
    return "\n".join(lines)


def format_operation_question(title, result):
    lines = [
        f"## {title}",
        "",
        f"- 状态：{result['status']}",
        f"- 预算：{result['budget_wan']} 万元",
        f"- 购置的挖掘机类型数：{len(result['selected_types'])}",
        f"- 购置型号：{', '.join(result['selected_types'])}",
        f"- 五年总产量：{result['summary']['total_output_m3']:.2f} 立方米",
        f"- 五年总收入：{result['summary']['revenue_wan']:.2f} 万元",
        f"- 五年总利润：{result['summary']['total_profit_wan']:.2f} 万元",
        "- 成本拆分：",
        f"  - 挖掘机采购：{result['summary']['purchase_wan']:.2f} 万元",
        f"  - 挖掘机人工维护：{result['summary']['exc_fixed_operating_wan']:.2f} 万元",
        f"  - 挖掘机燃油：{result['summary']['exc_fuel_wan']:.2f} 万元",
        f"  - 矿车人工维护：{result['summary']['truck_fixed_operating_wan']:.2f} 万元",
        f"  - 矿车燃油：{result['summary']['truck_fuel_wan']:.2f} 万元",
        "- 挖掘机采购数量：",
    ]
    for code, count in result["counts"].items():
        if count > 0:
            lines.append(f"  - {code}: {count} 台")
    lines.append("- 匹配方案：")
    for item in result["assignments"]:
        lines.append(
            "  - "
            f"{item['excavator']} -> {item['truck']}，"
            f"挖机 {item['excavator_count']} 台，矿车 {item['truck_count']} 辆，"
            f"匹配系数 {item['required_ratio']}，"
            f"有效作业当量 {item['effective_active_equivalent']} 台，"
            f"利用率 {item['utilization_ratio']:.2%}"
        )
    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    question_1 = solve_question_1()
    question_2 = solve_operation_question(
        "question_2",
        QUESTION_2_EXCAVATORS,
        QUESTION_2_TRUCKS,
        QUESTION_2_MATCH,
        budget_wan=2400,
        min_types=3,
    )
    question_3 = solve_operation_question(
        "question_3",
        QUESTION_3_EXCAVATORS,
        QUESTION_3_TRUCKS,
        QUESTION_3_MATCH,
        budget_wan=4000,
        min_types=5,
    )

    payload = {
        "question_1": question_1,
        "question_2": question_2,
        "question_3": question_3,
        "hours_total": HOURS_TOTAL,
        "months_total": MONTHS_TOTAL,
    }

    (RESULTS_DIR / "solution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = "\n\n".join(
        [
            "# 2024 MathorCup D题结果汇总",
            format_question_1(question_1),
            format_operation_question("问题2", question_2),
            format_operation_question("问题3", question_3),
        ]
    )
    (RESULTS_DIR / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
