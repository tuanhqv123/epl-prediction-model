import pandas as pd
import numpy as np

print("TẠI SAO XG KHÔNG CẢI THIỆN MODEL?")
print("=" * 60)

# Load data
xg_df = pd.read_csv("understat_data/our_match_xg.csv")

print("""
1. XG LÀ GÌ?
-----------
xG (Expected Goals) = xác suất ghi bàn của mỗi cú sút
- Tính từ: vị trí (X, Y), loại sút (chân/đầu), tình huống (penalty/corner/open play)
- xG của trận = tổng xG của tất cả cú sút

2. VẤN ĐỀ VỚI XG:
-----------------
""")

# Check xG vs Goals correlation
xg_df['goal_diff'] = xg_df['h_goals'] - xg_df['a_goals']
xg_df['xg_diff'] = xg_df['h_xG'] - xg_df['a_xG']

corr = xg_df['xg_diff'].corr(xg_df['goal_diff'])
print(f"Correlation xG_diff vs Goal_diff: {corr:.4f}")
print(f"=> xG chỉ giải thích {corr**2*100:.1f}% variance của goals")

# Check cases where xG and Goals disagree
xg_df['xg_winner'] = np.where(xg_df['h_xG'] > xg_df['a_xG'], 'H',
                     np.where(xg_df['h_xG'] < xg_df['a_xG'], 'A', 'D'))
xg_df['actual_winner'] = np.where(xg_df['h_goals'] > xg_df['a_goals'], 'H',
                         np.where(xg_df['h_goals'] < xg_df['a_goals'], 'A', 'D'))

agree = (xg_df['xg_winner'] == xg_df['actual_winner']).mean()
print(f"\nxG winner = Actual winner: {agree*100:.1f}%")
print(f"=> {(1-agree)*100:.1f}% trận xG dự đoán sai winner!")

print("""
3. TẠI SAO GOALS TỐT HƠN XG CHO PREDICTION?
------------------------------------------
""")

# The key insight: we're predicting FUTURE matches, not explaining past matches
print("""
- xG đo lường "cơ hội tạo ra" (chances created)
- Goals đo lường "khả năng ghi bàn thực tế" (actual finishing ability)

Khi predict trận TƯƠNG LAI:
- Team có xG cao nhưng không ghi bàn = có thể họ thiếu finisher tốt
- Team có xG thấp nhưng ghi nhiều bàn = có thể họ có finisher xuất sắc

=> Goals phản ánh TỔNG HỢP của:
   1. Khả năng tạo cơ hội (xG captures this)
   2. Khả năng dứt điểm (xG KHÔNG capture this)
   3. May mắn (random variance)

=> Dùng Goals để update ELO = capture cả 3 yếu tố
=> Dùng xG để update ELO = chỉ capture yếu tố 1
""")

# Example: teams that overperform/underperform xG
print("\n4. VÍ DỤ CỤ THỂ:")
print("-" * 60)

# Calculate xG over/underperformance per team
team_perf = []
for team in xg_df['h_team'].unique():
    home = xg_df[xg_df['h_team'] == team]
    away = xg_df[xg_df['a_team'] == team]
    
    total_goals = home['h_goals'].sum() + away['a_goals'].sum()
    total_xg = home['h_xG'].sum() + away['a_xG'].sum()
    matches = len(home) + len(away)
    
    if matches > 100:  # Only teams with enough data
        team_perf.append({
            'team': team,
            'matches': matches,
            'goals': total_goals,
            'xG': total_xg,
            'diff': total_goals - total_xg,
            'diff_per_match': (total_goals - total_xg) / matches
        })

perf_df = pd.DataFrame(team_perf).sort_values('diff_per_match', ascending=False)
print("\nTeams that OVERPERFORM xG (score more than expected):")
print(perf_df.head(5)[['team', 'matches', 'goals', 'xG', 'diff_per_match']].to_string(index=False))

print("\nTeams that UNDERPERFORM xG (score less than expected):")
print(perf_df.tail(5)[['team', 'matches', 'goals', 'xG', 'diff_per_match']].to_string(index=False))

print("""
5. KẾT LUẬN:
-----------
- xG là metric tốt để PHÂN TÍCH trận đấu đã qua
- Nhưng để PREDICT trận tương lai, Goals vẫn tốt hơn
- Vì Goals = xG + Finishing ability + Luck
- Model đã có SOT-based proxy xG trong expected_diff feature
- Thêm xG không thêm thông tin mới đáng kể

=> GIỮA NGUYÊN MODEL BASELINE (Loss=0.9362, Acc=56.7%)
""")
