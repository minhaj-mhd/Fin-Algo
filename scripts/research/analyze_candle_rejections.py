import os
import json
import numpy as np
import scipy.stats as stats
import subprocess

def run_analysis():
    jsonl_path = "data/research/candle_rejections.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}")
        return

    trades = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))

    if not trades:
        print("Error: No trade records found in the JSONL file.")
        return

    print(f"Loaded {len(trades)} terminal candle outcomes.")

    # 1. Group by reject_reason × side
    # We define group keys: reject_reason. If None, it is "CONFIRMED" or "FILLED"
    groups = {}
    for t in trades:
        reason = t.get("reject_reason") or "CONFIRMED_ENTRY"
        side = t.get("side", "LONG")
        key = (reason, side)
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    # Compute bootstrap CI helper
    def bootstrap_ci(data, num_resamples=2000):
        if len(data) == 0:
            return 0.0, 0.0
        rng = np.random.default_rng(42)
        resamples = []
        for _ in range(num_resamples):
            sample = rng.choice(data, size=len(data), replace=True)
            resamples.append(sample.mean())
        return np.percentile(resamples, 2.5), np.percentile(resamples, 97.5)

    report_lines = []
    report_lines.append("---")
    report_lines.append("title: \"Candle Rejection Performance\"")
    report_lines.append("type: research")
    report_lines.append("status: wip")
    report_lines.append("updated: " + np.datetime64('now').astype(str))
    report_lines.append("---")
    report_lines.append("# ⚠️ UNVERIFIED / Exploratory: Candle Rejection & Veto Performance")
    report_lines.append("\n> [!WARNING]")
    report_lines.append("> This report contains exploratory analysis from scripts/research/analyze_candle_rejections.py.")
    report_lines.append("> Per AGENTS.md, these research scripts hold no verdict authority. Grade metrics are for research only.\n")

    report_lines.append("## 📊 Rejection / Veto Performance Summary")
    report_lines.append("| Reject Reason | Side | N | Mean Net (bps) | Median Net (bps) | SL-Hit Rate | t-stat | 95% Bootstrap CI | Guard Value | Significant? |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for (reason, side), group_trades in sorted(groups.items()):
        net_pnls = [t["net_pnl_bps"] for t in group_trades]
        n = len(group_trades)
        mean_net = np.mean(net_pnls)
        median_net = np.median(net_pnls)
        
        # Calculate SL Hit rate: peak_adverse_pct <= -stop_loss_pct
        sl_hits = []
        for t in group_trades:
            adverse = t.get("peak_adverse_pct") or 0.0
            sl = t.get("stop_loss_pct") or 0.5
            sl_hits.append(1 if adverse <= -sl else 0)
        sl_hit_rate = np.mean(sl_hits) * 100.0 if sl_hits else 0.0
        
        # t-test vs 0
        if n > 1 and np.std(net_pnls) > 0:
            t_stat, p_val = stats.ttest_1samp(net_pnls, 0.0)
        else:
            t_stat, p_val = 0.0, 1.0
            
        ci_lower, ci_upper = bootstrap_ci(net_pnls)
        sig = "Yes" if (ci_lower > 0 or ci_upper < 0) and n > 1 else "No"
        
        # Guard value: -mean(net) for rejections, meaning we avoided this net loss/gain
        is_rejection = reason not in ("CONFIRMED_ENTRY", "None")
        guard_val = -mean_net if is_rejection else 0.0
        guard_val_str = f"{guard_val:+.2f} bps" if is_rejection else "N/A"
        
        report_lines.append(
            f"| {reason} | {side} | {n} | {mean_net:+.2f} | {median_net:+.2f} | {sl_hit_rate:.1f}% | {t_stat:+.2f} | [{ci_lower:+.2f}, {ci_upper:+.2f}] | {guard_val_str} | {sig} |"
        )

    # 2. Fade vs Market (paired delta on failed-confirmation trades)
    # Failed confirmation trades went to PENDING_LIMIT
    failed_conf_trades = []
    for t in trades:
        is_failed_conf = False
        if t.get("entry_mode") in ("fade_limit_filled", "limit_expired_market_fill"):
            is_failed_conf = True
        elif t.get("status") == "CANCELLED_EXPIRED" and t.get("reject_reason") == "LIMIT_EXPIRED":
            is_failed_conf = True
        if is_failed_conf:
            failed_conf_trades.append(t)

    report_lines.append("\n## ⚔️ Strategy Comparison: Fade vs Market (Limit Retracement)")
    report_lines.append("Comparing retracement limit strategy (A) vs immediate market entry (B) on signals that failed immediate look-back confirmation.")
    
    if failed_conf_trades:
        fade_nets = []
        market_nets = []
        paired_deltas = []
        
        for t in failed_conf_trades:
            # Fade Strategy net return
            if t.get("status") == "CANCELLED_EXPIRED":
                fade_net = 0.0  # limit expired, no trade
            else:
                fade_net = t["net_pnl_bps"]  # filled, actual trade net
            
            # Market Strategy net return (counterfactual from market_entry_px to exit_price)
            exit_px = t["exit_price"]
            market_px = t["market_entry_px"]
            side = t["side"]
            
            if side == "LONG":
                market_gross = (exit_px - market_px) / market_px
            else:
                market_gross = (market_px - exit_px) / market_px
            market_net = market_gross * 10000.0 - 10.0
            
            fade_nets.append(fade_net)
            market_nets.append(market_net)
            paired_deltas.append(fade_net - market_net)
            
        n_fc = len(paired_deltas)
        mean_fade = np.mean(fade_nets)
        mean_market = np.mean(market_nets)
        mean_delta = np.mean(paired_deltas)
        median_delta = np.median(paired_deltas)
        
        if n_fc > 1 and np.std(paired_deltas) > 0:
            t_fc, p_fc = stats.ttest_1samp(paired_deltas, 0.0)
        else:
            t_fc, p_fc = 0.0, 1.0
            
        ci_l_fc, ci_u_fc = bootstrap_ci(paired_deltas)
        sig_fc = "Yes" if (ci_l_fc > 0 or ci_u_fc < 0) and n_fc > 1 else "No"
        
        report_lines.append(f"- **Number of retracement signals**: {n_fc}")
        report_lines.append(f"- **Fade Strategy Mean Net**: {mean_fade:.2f} bps")
        report_lines.append(f"- **Market Strategy Mean Net**: {mean_market:.2f} bps")
        report_lines.append(f"- **Mean Paired Difference (A - B)**: {mean_delta:+.2f} bps (Median: {median_delta:+.2f} bps)")
        report_lines.append(f"- **t-statistic**: {t_fc:+.2f}")
        report_lines.append(f"- **95% Bootstrap CI of Difference**: [{ci_l_fc:+.2f}, {ci_u_fc:+.2f}]")
        report_lines.append(f"- **Is Fade significantly better than Market?**: {sig_fc}")
    else:
        report_lines.append("*No failed-confirmation trades available for comparison.*")

    # 3. Confirmation Value
    report_lines.append("\n## 🔍 Confirmation Value Analysis")
    report_lines.append("Does immediate candle direction confirmation select higher-performing entries than the baseline average of all signals?")
    
    confirmed_trades = [t for t in trades if t.get("entry_mode") == "immediate"]
    if confirmed_trades:
        conf_nets = [t["net_pnl_bps"] for t in confirmed_trades]
        all_nets = []
        for t in trades:
            if t.get("entry_mode") == "immediate":
                all_nets.append(t["net_pnl_bps"])
            else:
                # Vetoed / cancelled are counterfactual from market_entry_px to exit_price
                entry_px = t["market_entry_px"]
                exit_px = t["exit_price"]
                side = t["side"]
                if side == "LONG":
                    gross = (exit_px - entry_px) / entry_px
                else:
                    gross = (entry_px - exit_px) / entry_px
                net = gross * 10000.0 - 10.0
                all_nets.append(net)
                
        mean_conf = np.mean(conf_nets)
        mean_all = np.mean(all_nets)
        diff_conf = mean_conf - mean_all
        
        report_lines.append(f"- **Confirmed Entry Mean Net**: {mean_conf:.2f} bps (N={len(confirmed_trades)})")
        report_lines.append(f"- **All Signals Baseline Mean Net**: {mean_all:.2f} bps (N={len(trades)})")
        report_lines.append(f"- **Confirmation Edge (Confirmed - All)**: {diff_conf:+.2f} bps")
    else:
        report_lines.append("*No confirmed trades available for analysis.*")

    # Write report
    report_dir = "finalgo-memory-layer/finalgo/04 — Research"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "Candle Rejection Performance.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
        
    print(f"Report written successfully to {report_path}")

    # Rebuild index
    try:
        print("Rebuilding Obsidian index...")
        subprocess.run(["python", "scripts/memory/build_index.py"], check=True)
        print("Obsidian index rebuilt successfully.")
    except Exception as e:
        print(f"Error rebuilding Obsidian index: {e}")

if __name__ == "__main__":
    run_analysis()
