# Statcast 컬럼 명세

- 전체 1,424,426구 / 컬럼 111개
- 기간 2024-03-20 ~ 2025-09-28
- 2024 711,898구 / 2025 712,528구

## 규칙

- 🚫 = **타깃 투구의 결과**. 라벨 생성 후 피처에서 반드시 제거.
  단 과거 8구 시퀀스에서는 사용 가능(이미 알려진 사실).
- `post_*` / `delta_*` 접두사는 정의상 전부 금지.

| 컬럼 | 타입 | 결측% | 고유값 | 예시 |
|---|---|---|---|---|
| `pitch_type` | string | 0.4 | 18 | FF, SL, CH |
| `game_date` | datetime64[ns] | 0.0 | 369 | 2024-03-31 00:00:0, 2024-03-30 00:00:0, 2024-03-28 00:00:0 |
| `release_speed` | float64 | 0.4 | 725 | 94.5, 94.9, 89.2 |
| `release_pos_x` | float64 | 0.4 | 959 | -1.74, -1.66, -1.78 |
| `release_pos_z` | float64 | 0.4 | 668 | 5.62, 5.5, 5.66 |
| `player_name` | string | 0.0 | 1,111 | Gray, Jon, Wicks, Jordan, Ureña, José |
| `batter` | Int64 | 0.0 | 788 | 664023, 673548, 641355 |
| `pitcher` | Int64 | 0.0 | 1,113 | 592351, 696136, 570632 |
| `events` 🚫 | string | 74.3 | 22 | field_out, single, walk |
| `description` 🚫 | string | 0.0 | 15 | called_strike, ball, foul |
| `zone` | Int64 | 0.4 | 13 | 4, 14, 13 |
| `des` 🚫 | string | 74.3 | 239,889 | Ian Happ grounds o, Seiya Suzuki singl, Cody Bellinger wal |
| `game_type` | string | 0.0 | 1 | R |
| `stand` | string | 0.0 | 2 | L, R |
| `p_throws` | string | 0.0 | 2 | R, L |
| `home_team` | string | 0.0 | 30 | TEX, TB, SEA |
| `away_team` | string | 0.0 | 30 | CHC, TOR, BOS |
| `type` 🚫 | string | 0.0 | 3 | S, B, X |
| `hit_location` 🚫 | Int64 | 77.6 | 9 | 4, 8, 2 |
| `bb_type` 🚫 | string | 82.5 | 4 | ground_ball, fly_ball, line_drive |
| `balls` | Int64 | 0.0 | 4 | 0, 1, 2 |
| `strikes` | Int64 | 0.0 | 3 | 0, 1, 2 |
| `game_year` | Int64 | 0.0 | 2 | 2024, 2025 |
| `pfx_x` | float64 | 0.4 | 469 | -1.17, -1.22, -0.08 |
| `pfx_z` | float64 | 0.4 | 437 | 1.1, 1.2, 0.67 |
| `plate_x` | float64 | 0.4 | 1,419,133 | -0.336693225966118, 0.9804933552340458, -0.960921873643344 |
| `plate_z` | float64 | 0.4 | 1,419,133 | 2.3508348310624747, 2.1102258932885203, 1.4221971765939598 |
| `on_3b` | Int64 | 90.6 | 740 | 665804, 668800, 665120 |
| `on_2b` | Int64 | 81.1 | 764 | 673548, 668800, 694497 |
| `on_1b` | Int64 | 69.6 | 771 | 673548, 641355, 694671 |
| `outs_when_up` | Int64 | 0.0 | 3 | 0, 1, 2 |
| `inning` | Int64 | 0.0 | 14 | 1, 2, 3 |
| `inning_topbot` | string | 0.0 | 2 | Top, Bot |
| `hc_x` 🚫 | float64 | 82.5 | 20,937 | 153.34, 126.69, 51.26 |
| `hc_y` 🚫 | float64 | 82.5 | 18,537 | 159.8, 79.25, 53.82 |
| `vx0` | float64 | 0.4 | 1,419,282 | 6.370888413573001, 9.738564711880985, 2.2307563068809064 |
| `vy0` | float64 | 0.4 | 1,419,281 | -137.4173346114854, -137.7203846021830, -129.9035955651535 |
| `vz0` | float64 | 0.4 | 1,419,282 | -5.323252661411089, -5.901267137485416, -5.905849194663354 |
| `ax` | float64 | 0.4 | 1,419,282 | -16.27258183430261, -17.77111548899392, -1.412994726617688 |
| `ay` | float64 | 0.4 | 1,419,281 | 31.504853565130137, 32.00987404435315, 26.30875974036476 |
| `az` | float64 | 0.4 | 1,419,281 | -17.20822533348994, -15.72103493391954, -23.53169189355748 |
| `sz_top` | float64 | 0.4 | 674,619 | 3.13696141103885, 3.18836985071336, 3.22369401541201 |
| `sz_bot` | float64 | 0.4 | 664,012 | 1.33752652220159, 1.5563033180467, 1.46925717282905 |
| `hit_distance_sc` 🚫 | Int64 | 66.8 | 483 | 63, 98, 2 |
| `launch_speed` 🚫 | float64 | 66.8 | 1,151 | 96.6, 112.8, 43.2 |
| `launch_angle` 🚫 | Int64 | 66.7 | 181 | 3, 4, -47 |
| `effective_speed` | float64 | 0.5 | 766 | 95.7, 95.5, 90.6 |
| `release_spin_rate` | Int64 | 0.7 | 3,400 | 1851, 2231, 2306 |
| `release_extension` | float64 | 0.4 | 62 | 7.1, 6.9, 7.0 |
| `game_pk` | Int64 | 0.0 | 4,859 | 745035, 745037, 745039 |
| `fielder_2` | Int64 | 0.0 | 131 | 668800, 665804, 543228 |
| `fielder_3` | Int64 | 0.0 | 247 | 665120, 683737, 677649 |
| `fielder_4` | Int64 | 0.0 | 243 | 543760, 663538, 664040 |
| `fielder_5` | Int64 | 0.0 | 237 | 673962, 666624, 663611 |
| `fielder_6` | Int64 | 0.0 | 161 | 677649, 621020, 608369 |
| `fielder_7` | Int64 | 0.0 | 323 | 694671, 643565, 694497 |
| `fielder_8` | Int64 | 0.0 | 231 | 665750, 641355, 608671 |
| `fielder_9` | Int64 | 0.0 | 304 | 694497, 673548, 666969 |
| `release_pos_y` | float64 | 0.4 | 568 | 53.41, 53.59, 53.5 |
| `estimated_ba_using_speedangle` 🚫 | float64 | 82.8 | 925 | 0.495, 0.671, 0.819 |
| `estimated_woba_using_speedangle` 🚫 | float64 | 74.6 | 1,751 | 0.445, 0.636, 0.689131 |
| `woba_value` 🚫 | float64 | 74.3 | 8 | 0.0, 0.9, 0.7 |
| `woba_denom` 🚫 | Int64 | 74.4 | 2 | 1, 0 |
| `babip_value` 🚫 | Int64 | 74.3 | 2 | 0, 1 |
| `iso_value` 🚫 | Int64 | 74.3 | 4 | 0, 3, 1 |
| `launch_speed_angle` 🚫 | Int64 | 82.6 | 6 | 4, 6, 3 |
| `at_bat_number` | Int64 | 0.0 | 114 | 1, 2, 3 |
| `pitch_number` | Int64 | 0.0 | 17 | 1, 2, 3 |
| `pitch_name` | string | 0.4 | 18 | 4-Seam Fastball, Slider, Changeup |
| `home_score` | Int64 | 0.0 | 22 | 0, 2, 3 |
| `away_score` | Int64 | 0.0 | 23 | 0, 3, 5 |
| `bat_score` | Int64 | 0.0 | 24 | 0, 3, 2 |
| `fld_score` | Int64 | 0.0 | 24 | 0, 3, 2 |
| `post_away_score` 🚫 | Int64 | 0.0 | 23 | 0, 3, 5 |
| `post_home_score` 🚫 | Int64 | 0.0 | 22 | 0, 2, 3 |
| `post_bat_score` 🚫 | Int64 | 0.0 | 24 | 0, 3, 2 |
| `post_fld_score` 🚫 | Int64 | 0.0 | 24 | 0, 3, 2 |
| `if_fielding_alignment` | string | 0.6 | 3 | Infield shade, Standard, Strategic |
| `of_fielding_alignment` | string | 0.6 | 2 | Standard, Strategic |
| `spin_axis` | Int64 | 0.7 | 361 | 217, 214, 197 |
| `delta_home_win_exp` 🚫 | float64 | 0.0 | 1,054 | 0.003, -0.002, -0.005 |
| `delta_run_exp` 🚫 | float64 | 0.3 | 1,731 | -0.037, 0.025, 0.053 |
| `bat_speed` | float64 | 54.5 | 875 | 69.6, 76.9, 73.4 |
| `swing_length` | float64 | 54.5 | 121 | 6.7, 7.1, 7.0 |
| `miss_distance` | float64 | 89.4 | 143,692 | 6.840103704, 10.243695096, 1.560077196 |
| `estimated_slg_using_speedangle` 🚫 | float64 | 82.8 | 1,825 | 0.543, 0.777, 3.021 |
| `delta_pitcher_run_exp` 🚫 | float64 | 0.3 | 1,731 | 0.037, -0.025, -0.053 |
| `hyper_speed` | float64 | 66.6 | 322 | 96.6, 112.8, 88.0 |
| `home_score_diff` | Int64 | 0.0 | 42 | 0, -3, -1 |
| `bat_score_diff` | Int64 | 0.0 | 45 | 0, 3, -3 |
| `home_win_exp` 🚫 | float64 | 0.0 | 1,000 | 0.5, 0.503, 0.501 |
| `bat_win_exp` 🚫 | float64 | 0.0 | 1,240 | 0.5, 0.497, 0.499 |
| `age_pit_legacy` | Int64 | 0.0 | 25 | 32, 24, 33 |
| `age_bat_legacy` | Int64 | 0.0 | 22 | 29, 28, 25 |
| `age_pit` | Int64 | 0.0 | 25 | 33, 25, 27 |
| `age_bat` | Int64 | 0.0 | 22 | 30, 29, 25 |
| `n_thruorder_pitcher` | Int64 | 0.0 | 5 | 1, 2, 3 |
| `n_priorpa_thisgame_player_at_bat` | Int64 | 0.0 | 8 | 0, 1, 2 |
| `pitcher_days_since_prev_game` | Int64 | 4.9 | 141 | 3, 2, 1 |
| `batter_days_since_prev_game` | Int64 | 1.2 | 124 | 1, 3, 2 |
| `pitcher_days_until_next_game` | Int64 | 3.9 | 142 | 6, 3, 2 |
| `batter_days_until_next_game` | Int64 | 0.9 | 124 | 1, 2, 8 |
| `api_break_z_with_gravity` | float64 | 0.4 | 1,341 | 1.46, 1.35, 2.18 |
| `api_break_x_arm` | float64 | 0.4 | 471 | 1.17, 1.22, 0.08 |
| `api_break_x_batter_in` | float64 | 0.4 | 471 | -1.17, -1.22, -0.08 |
| `arm_angle` | float64 | 0.7 | 1,278 | 35.3, 31.1, 33.2 |
| `attack_angle` | float64 | 54.5 | 647,411 | 4.647823536530293, 0.131444860421381, -4.334967082159135 |
| `attack_direction` | float64 | 54.5 | 647,411 | -2.118820422500102, 8.705511020240689, 11.914787929056567 |
| `swing_path_tilt` | float64 | 54.5 | 647,406 | 42.862885082236325, 45.54593758334903, 37.61996434966567 |
| `intercept_ball_minus_batter_pos_x_inches` | float64 | 54.6 | 646,653 | 35.45340773472175, 28.955543444580183, 35.92286519877072 |
| `intercept_ball_minus_batter_pos_y_inches` | float64 | 54.6 | 646,653 | 29.07852239853185, 23.06357617786504, 16.648986784422938 |