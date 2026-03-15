import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "solution" / "params.json"


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


def transition_key(a, b):
    return f"trans_{a.lower()}_{b.lower()}"


def load_params():
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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


def simulate_race(config, strategies, p):
    rows = []

    for pos_key, strat in strategies.items():
        driver = strat["driver_id"]
        grid = int(pos_key.replace("pos", ""))
        t = strategy_time(config, strat, p)
        rows.append((t, grid, driver))

    rows.sort(key=lambda x: (x[0], x[1]))
    return [d for _, _, d in rows]


def main():
    data = json.load(sys.stdin)
    p = load_params()

    result = {
        "race_id": data["race_id"],
        "finishing_positions": simulate_race(
            data["race_config"],
            data["strategies"],
            p
        )
    }

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()