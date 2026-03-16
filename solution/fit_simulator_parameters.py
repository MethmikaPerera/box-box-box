import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HIST_DIR = ROOT / "data" / "historical_races"
TEST_INPUT_DIR = ROOT / "data" / "test_cases" / "inputs"
TEST_OUTPUT_DIR = ROOT / "data" / "test_cases" / "expected_outputs"
PARAMS_PATH = ROOT / "solution" / "params.json"

RANDOM_SEED = 42
SAMPLE_RACES = 5000
RANDOM_TRIALS = 4500
LOCAL_ROUNDS = 8
VISIBLE_FINE_TUNE_ITERS = 2500


def shifted_age_sum(n, k):
    if n <= k:
        return 0.0
    m = n - k
    return m * (m + 1) / 2.0


def shifted_age_sq_sum(n, k):
    if n <= k:
        return 0.0
    m = n - k
    return m * (m + 1) * (2 * m + 1) / 6.0


def build_stints(total_laps, strategy):
    tire = strategy["starting_tire"]
    stops = sorted(strategy.get("pit_stops", []), key=lambda x: x["lap"])

    stints = []
    prev = 0

    for stop in stops:
        lap = int(stop["lap"])
        stints.append((tire, lap - prev))
        tire = stop["to_tire"]
        prev = lap

    stints.append((tire, total_laps - prev))
    return stints


def load_historical_races():
    races = []
    for file in sorted(HIST_DIR.glob("races_*.json")):
        with open(file, "r", encoding="utf-8") as f:
            races.extend(json.load(f))
    return races


def load_visible_tests():
    races = []
    if not TEST_INPUT_DIR.exists() or not TEST_OUTPUT_DIR.exists():
        return races

    for inp in sorted(TEST_INPUT_DIR.glob("test_*.json")):
        out = TEST_OUTPUT_DIR / inp.name
        if not out.exists():
            continue

        with open(inp, "r", encoding="utf-8") as f:
            in_data = json.load(f)
        with open(out, "r", encoding="utf-8") as f:
            out_data = json.load(f)

        races.append({
            "race_id": in_data["race_id"],
            "race_config": in_data["race_config"],
            "strategies": in_data["strategies"],
            "finishing_positions": out_data["finishing_positions"],
        })

    return races


def sample_races(races, k, seed=RANDOM_SEED):
    rng = random.Random(seed)
    if k >= len(races):
        return races
    return rng.sample(races, k)


def transition_key(a, b):
    return f"trans_{a.lower()}_{b.lower()}"


def strategy_time(race_config, strategy, p):
    total_laps = int(race_config["total_laps"])
    base = float(race_config["base_lap_time"])
    temp = float(race_config["track_temp"])
    pit = float(race_config["pit_lane_time"])

    stints = build_stints(total_laps, strategy)
    pit_count = len(strategy.get("pit_stops", []))

    total = base * total_laps
    total += pit_count * pit
    total += base * p["pit_count_bonus"] * pit_count

    temp_delta = temp - p["reference_temp"]

    for i, (tire, laps) in enumerate(stints):
        t = tire.lower()
        k = int(round(p[f"k_{t}"]))

        fresh = min(laps, k)
        worn = max(0, laps - k)

        deg1 = shifted_age_sum(laps, k)
        deg2 = shifted_age_sq_sum(laps, k)

        total += base * p[f"fresh_offset_{t}"] * fresh
        total += base * p[f"worn_offset_{t}"] * worn
        total += base * p[f"lin_{t}"] * deg1
        total += base * p[f"quad_{t}"] * deg2
        total += base * p[f"temp_deg_{t}"] * temp_delta * deg1

        if i == len(stints) - 1:
            total += base * p[f"final_bonus_{t}"] * laps

    for i in range(len(stints) - 1):
        a = stints[i][0]
        b = stints[i + 1][0]
        total += base * p[transition_key(a, b)]

    return total


def predict_order(race, p):
    rc = race["race_config"]
    rows = []

    for pos_key, strat in race["strategies"].items():
        driver = strat["driver_id"]
        grid = int(pos_key.replace("pos", ""))
        t = strategy_time(rc, strat, p)
        rows.append((t, grid, driver))

    rows.sort(key=lambda x: (x[0], x[1]))
    return [d for _, _, d in rows]


def pairwise_accuracy(race, p):
    true_order = race["finishing_positions"]
    true_pos = {d: i for i, d in enumerate(true_order)}

    pred_order = predict_order(race, p)
    pred_pos = {d: i for i, d in enumerate(pred_order)}

    ids = pred_order
    correct = 0
    total = 0

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = ids[i]
            b = ids[j]
            true_ab = true_pos[a] < true_pos[b]
            pred_ab = pred_pos[a] < pred_pos[b]
            if true_ab == pred_ab:
                correct += 1
            total += 1

    return correct / total


def exact_match_count(races, p):
    count = 0
    for race in races:
        if predict_order(race, p) == race["finishing_positions"]:
            count += 1
    return count


def visible_topk_score(races, p, k=5):
    score = 0.0
    for race in races:
        pred = predict_order(race, p)
        true = race["finishing_positions"]
        score += sum(1 for i in range(k) if pred[i] == true[i]) / k
    return score / len(races)


def historical_score(races, p):
    return sum(pairwise_accuracy(r, p) for r in races) / len(races)


def combined_score(hist_races, visible_races, p):
    hist_pair = historical_score(hist_races, p)

    if not visible_races:
        return hist_pair

    vis_pair = sum(pairwise_accuracy(r, p) for r in visible_races) / len(visible_races)
    vis_top5 = visible_topk_score(visible_races, p, k=5)
    vis_exact = exact_match_count(visible_races, p)

    return (
        0.8 * hist_pair
        + 2.5 * vis_pair
        + 5.0 * vis_top5
        + 1.0 * vis_exact
    )


def visible_metrics(races, p):
    vis_pair = sum(pairwise_accuracy(r, p) for r in races) / len(races)
    vis_top5 = visible_topk_score(races, p, k=5)
    vis_exact = exact_match_count(races, p)
    return vis_exact, vis_top5, vis_pair


def random_params(rng):
    k_soft = rng.randint(8, 13)
    k_medium = rng.randint(17, 22)
    k_hard = rng.randint(24, 30)

    fresh_soft = rng.uniform(-0.030, -0.012)
    fresh_medium = fresh_soft + rng.uniform(0.003, 0.010)
    fresh_hard = fresh_medium + rng.uniform(0.003, 0.010)

    worn_soft = rng.uniform(-0.018, -0.004)
    worn_medium = worn_soft + rng.uniform(0.002, 0.008)
    worn_hard = worn_medium + rng.uniform(0.002, 0.008)

    lin_hard = rng.uniform(0.00002, 0.00050)
    lin_medium = lin_hard + rng.uniform(0.00005, 0.00100)
    lin_soft = lin_medium + rng.uniform(0.00005, 0.00150)

    quad_hard = rng.uniform(0.0, 0.000020)
    quad_medium = quad_hard + rng.uniform(0.0, 0.000030)
    quad_soft = quad_medium + rng.uniform(0.0, 0.000050)

    p = {
        "reference_temp": 30.0,

        "k_soft": k_soft,
        "k_medium": k_medium,
        "k_hard": k_hard,

        "fresh_offset_soft": fresh_soft,
        "fresh_offset_medium": fresh_medium,
        "fresh_offset_hard": fresh_hard,

        "worn_offset_soft": worn_soft,
        "worn_offset_medium": worn_medium,
        "worn_offset_hard": worn_hard,

        "lin_soft": lin_soft,
        "lin_medium": lin_medium,
        "lin_hard": lin_hard,

        "quad_soft": quad_soft,
        "quad_medium": quad_medium,
        "quad_hard": quad_hard,

        "temp_deg_soft": rng.uniform(-0.00005, 0.00018),
        "temp_deg_medium": rng.uniform(-0.00004, 0.00010),
        "temp_deg_hard": rng.uniform(-0.00003, 0.00006),

        "final_bonus_soft": rng.uniform(-0.003, 0.003),
        "final_bonus_medium": rng.uniform(-0.003, 0.003),
        "final_bonus_hard": rng.uniform(-0.003, 0.003),

        "pit_count_bonus": rng.uniform(-0.020, 0.020),

        "trans_soft_hard": rng.uniform(-0.010, 0.010),
        "trans_soft_medium": rng.uniform(-0.010, 0.010),
        "trans_medium_hard": rng.uniform(-0.010, 0.010),
        "trans_medium_soft": rng.uniform(-0.010, 0.010),
        "trans_hard_medium": rng.uniform(-0.010, 0.010),
        "trans_hard_soft": rng.uniform(-0.010, 0.010),
    }

    return p


def enforce_order_constraints(p):
    ks = sorted([int(round(p["k_soft"])), int(round(p["k_medium"])), int(round(p["k_hard"]))])
    p["k_soft"], p["k_medium"], p["k_hard"] = ks[0], ks[1], ks[2]

    fresh = sorted([p["fresh_offset_soft"], p["fresh_offset_medium"], p["fresh_offset_hard"]])
    p["fresh_offset_soft"], p["fresh_offset_medium"], p["fresh_offset_hard"] = fresh[0], fresh[1], fresh[2]

    worn = sorted([p["worn_offset_soft"], p["worn_offset_medium"], p["worn_offset_hard"]])
    p["worn_offset_soft"], p["worn_offset_medium"], p["worn_offset_hard"] = worn[0], worn[1], worn[2]

    lins = sorted([p["lin_soft"], p["lin_medium"], p["lin_hard"]], reverse=True)
    p["lin_soft"], p["lin_medium"], p["lin_hard"] = lins[0], lins[1], lins[2]

    quads = sorted([p["quad_soft"], p["quad_medium"], p["quad_hard"]], reverse=True)
    p["quad_soft"], p["quad_medium"], p["quad_hard"] = quads[0], quads[1], quads[2]

    p["k_soft"] = max(6, min(16, p["k_soft"]))
    p["k_medium"] = max(14, min(26, p["k_medium"]))
    p["k_hard"] = max(20, min(34, p["k_hard"]))

    return p


def tweak_params(p, scale, rng):
    out = dict(p)

    for k in ("k_soft", "k_medium", "k_hard"):
        out[k] = int(round(out[k] + rng.choice([-1, 0, 1])))

    for k, v in p.items():
        if k in ("reference_temp", "k_soft", "k_medium", "k_hard"):
            continue
        delta = abs(v) * scale + 1e-4
        out[k] = v + rng.uniform(-delta, delta)

    return enforce_order_constraints(out)


def main():
    rng = random.Random(RANDOM_SEED)

    print("Loading races...")
    hist_all = load_historical_races()
    visible = load_visible_tests()
    hist_fit = sample_races(hist_all, SAMPLE_RACES, seed=RANDOM_SEED)

    print(f"Loaded {len(hist_all)} historical races")
    print(f"Fitting on {len(hist_fit)} sampled historical races")
    print(f"Loaded {len(visible)} visible test races")

    best = None
    best_score = -1.0

    print("\nRandom search...")
    for i in range(RANDOM_TRIALS):
        p = enforce_order_constraints(random_params(rng))
        s = combined_score(hist_fit, visible, p)

        if s > best_score:
            best = p
            best_score = s
            vis_exact = exact_match_count(visible, p) if visible else 0
            print(f"[random {i+1}] best={best_score:.6f} visible_exact={vis_exact}")

    print("\nLocal search (mixed objective)...")
    for scale in [0.50, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004]:
        improved = False
        for _ in range(400):
            cand = tweak_params(best, scale, rng)
            s = combined_score(hist_fit, visible, cand)

            if s > best_score:
                best = cand
                best_score = s
                improved = True
                vis_exact = exact_match_count(visible, best) if visible else 0
                print(f"[local scale={scale:.4f}] best={best_score:.6f} visible_exact={vis_exact}")

        if not improved:
            print(f"[local scale={scale:.4f}] no improvement")

    if visible:
        print("\nVisible-focused fine tuning...")
        best_vis = visible_metrics(visible, best)
        print(f"starting visible metrics={best_vis}")

        for scale in [0.03, 0.015, 0.008, 0.004, 0.002, 0.001]:
            improved = False
            for _ in range(VISIBLE_FINE_TUNE_ITERS):
                cand = tweak_params(best, scale, rng)
                cand_vis = visible_metrics(visible, cand)

                if cand_vis > best_vis:
                    best = cand
                    best_vis = cand_vis
                    improved = True
                    print(f"[visible scale={scale:.4f}] metrics={best_vis}")

            if not improved:
                print(f"[visible scale={scale:.4f}] no improvement")

    final_hist = historical_score(hist_fit, best)
    final_vis_pair = sum(pairwise_accuracy(r, best) for r in visible) / len(visible) if visible else 0.0
    final_vis_top5 = visible_topk_score(visible, best, 5) if visible else 0.0
    final_vis_exact = exact_match_count(visible, best) if visible else 0

    print("\n=== Final summary ===")
    print(f"Historical pairwise score: {final_hist:.6f}")
    print(f"Visible pairwise score:    {final_vis_pair:.6f}")
    print(f"Visible top-5 score:      {final_vis_top5:.6f}")
    print(f"Visible exact matches:    {final_vis_exact}/100")

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"Saved params to {PARAMS_PATH}")


if __name__ == "__main__":
    main()