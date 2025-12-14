import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. 모델 구조 정의
SEQ_LEN = 8 

class AdvancedBaseballTransformer(nn.Module):
    def __init__(self, embedding_dims, num_numeric, d_model, num_classes, nhead=8, num_layers=4):
        super().__init__()
        
        # Embeddings
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for col, (num_unique, dim) in embedding_dims.items():
            self.embeddings[col] = nn.Embedding(num_unique, dim)
            total_embed_dim += dim
            
        self.input_proj = nn.Linear(total_embed_dim + num_numeric, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Context Projection
        self.context_proj = nn.Linear(6, 32) 
        
        # Output Head
        self.fc_head = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_cat, x_num, x_ctx):
        embedded_list = []
        keys = ['batter', 'pitcher', 'pitch_type', 'stand', 'p_throws']
        for i, key in enumerate(keys):
            embedded_list.append(self.embeddings[key](x_cat[:, :, i]))
            
        x_emb = torch.cat(embedded_list, dim=2)
        x_combined = torch.cat([x_emb, x_num], dim=2)
        
        x = self.input_proj(x_combined) + self.pos_encoder
        x = self.transformer(x)
        
        # Transformer 
        history_vec = x[:, -1, :]
        
        current_vec = F.relu(self.context_proj(x_ctx))
        
        # 결합 
        final_vec = torch.cat([history_vec, current_vec], dim=1)
        
        return self.fc_head(final_vec)

# 2. 설정 및 리소스 로드
st.set_page_config(page_title="NextPitch AI Advisor", page_icon="⚾", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F1F3F5; color: #212529; font-family: 'Suit', sans-serif; }
    .rec-card {
        background-color: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 15px;
        border-left: 6px solid #228BE6; transition: transform 0.2s;
    }
    .rec-card:hover { transform: translateY(-2px); }
    div.stButton > button {
        background-color: #228BE6; color: white; font-weight: bold; font-size: 1.2rem;
        border: none; padding: 1rem; border-radius: 10px; width: 100%;
        box-shadow: 0 4px 6px rgba(34, 139, 230, 0.3);
    }
    div.stButton > button:hover { background-color: #1971C2; }
    h1, h2, h3 { color: #343A40 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        encoders = joblib.load('models/paper_encoders.pkl')
        embed_config = joblib.load('models/embed_config.pkl')
        num_cols = joblib.load('models/num_cols.pkl')
        
        num_numeric = len(num_cols)
        num_classes = len(encoders['outcome'].classes_)
        
        model = AdvancedBaseballTransformer(embed_config, num_numeric, d_model=256, num_classes=num_classes, nhead=8, num_layers=4)
        model.load_state_dict(torch.load('models/paper_model.pth', map_location='cpu'))
        model.eval()
        
        #  이름 매핑 파일 로드
        if os.path.exists('models/player_mapping.pkl'):
            # pkl 파일 로드
            raw_map = joblib.load('models/player_mapping.pkl')
            # 키를 문자열로 변환하여 저장 (매칭 오류 방지)
            player_map = {str(k): v for k, v in raw_map.items()}
        else:
            player_map = {}
            
        return model, encoders, num_cols, player_map
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
        return None, None, None, None

model, encoders, num_cols, player_map = load_resources()
if not model: st.stop()

# 3. 로직
COMMON_PITCHES = {
    'FF': '4-Seam Fastball', 'SL': 'Slider', 'CU': 'Curveball', 'CH': 'Changeup',
    'SI': 'Sinker', 'FC': 'Cutter', 'ST': 'Sweeper', 'FS': 'Splitter'
}

def get_zone_coords(zone_num):
    xs = [-0.6, 0.0, 0.6]
    zs = [3.2, 2.5, 1.8]
    row = (zone_num - 1) // 3
    col = (zone_num - 1) % 3
    return xs[col], zs[row]

def get_zone_desc(zone_num, stand):
    row = (zone_num - 1) // 3
    col = (zone_num - 1) % 3
    h_desc = ["높은", "가운데", "낮은"][row]
    if col == 1: s_desc = "가운데"
    elif col == 0: s_desc = "몸쪽" if stand == 'R' else "바깥쪽"
    else: s_desc = "바깥쪽" if stand == 'R' else "몸쪽"
    return f"{s_desc} {h_desc}"

def get_pitch_specs(pitch_type, p_throws):
    specs = {
        'FF': (94.5, 2300, -0.6, 1.4), 'SL': (84.5, 2450, 0.4, 0.2), 
        'CU': (78.0, 2600, 0.7, -0.8), 'CH': (85.5, 1750, -0.7, 0.5), 
        'SI': (93.0, 2150, -0.8, 0.7), 'FC': (90.0, 2400, 0.2, 0.5),
        'ST': (83.0, 2700, 0.6, 0.1), 'FS': (87.0, 1400, -0.5, 0.2)
    }
    val = specs.get(pitch_type, (90.0, 2200, 0.0, 0.0))
    speed, spin, pfx_x, pfx_z = val
    if p_throws == 'L': pfx_x = -pfx_x
    return speed, spin, pfx_x, pfx_z

def recommend_strategy(batter_id, pitcher_id, stand, p_throws, b, s, o, on_1b, on_2b, on_3b):
    scenarios = []
    for pt_code in COMMON_PITCHES.keys():
        if pt_code in encoders['pitch_type'].classes_:
            for zone in range(1, 10):
                scenarios.append({'pitch': pt_code, 'zone': zone})
    
    # ID 인코딩
    try: b_idx = encoders['batter'].transform([batter_id])[0]
    except: b_idx = 0 
    try: p_idx = encoders['pitcher'].transform([pitcher_id])[0]
    except: p_idx = 0
    stand_idx = encoders['stand'].transform([stand])[0]
    p_throws_idx = encoders['p_throws'].transform([p_throws])[0]
    
    cat_batch, num_batch, ctx_batch = [], [], []
    
    for sc in scenarios:
        pt_idx = encoders['pitch_type'].transform([sc['pitch']])[0]
        speed, spin, pfx_x, pfx_z = get_pitch_specs(sc['pitch'], p_throws)
        px, pz = get_zone_coords(sc['zone'])
        
        cat_batch.append([b_idx, p_idx, pt_idx, stand_idx, p_throws_idx])
        num_batch.append([b, s, o, 1 if on_1b else 0, 1 if on_2b else 0, 1 if on_3b else 0, speed, spin, pfx_x, pfx_z, px, pz])
        ctx_batch.append([b, s, o, 1 if on_1b else 0, 1 if on_2b else 0, 1 if on_3b else 0])
        
    num_batch_scaled = encoders['scaler'].transform(num_batch)
    X_cat = torch.tensor(cat_batch, dtype=torch.long).unsqueeze(1).repeat(1, SEQ_LEN, 1)
    X_num = torch.tensor(num_batch_scaled, dtype=torch.float32).unsqueeze(1).repeat(1, SEQ_LEN, 1)
    X_ctx = torch.tensor(ctx_batch, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(X_cat, X_num, X_ctx)
        temperature = 5.0 
        probs = torch.softmax(logits / temperature, dim=1).numpy()
        
    outcomes = encoders['outcome'].classes_
    idx_good = [i for i, x in enumerate(outcomes) if x in ['strike', 'strikeout', 'field_out', 'swinging_strike', 'foul']]
    idx_bad = [i for i, x in enumerate(outcomes) if x in ['single', 'double', 'home_run', 'ball', 'walk', 'hit_by_pitch']]
    
    results = []
    for i, p_dist in enumerate(probs):
        p_success = np.sum(p_dist[idx_good])
        p_fail = np.sum(p_dist[idx_bad])
        
        penalty = 0.05 if scenarios[i]['pitch'] == 'SI' else 0
        score = p_success - (p_fail * 1.5) - penalty
        results.append({'pitch': scenarios[i]['pitch'], 'zone': scenarios[i]['zone'], 'prob_success': p_success, 'prob_fail': p_fail, 'score': score})
        
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:3]

# 4. UI 구성
st.title("⚾ NextPitch AI: Smart Catcher")
st.markdown("##### 타자의 약점과 상황을 분석하여, 가장 확률 높은 공을 추천합니다.")

raw_batter_ids = encoders['batter'].classes_
valid_batter_options = []

for bid in raw_batter_ids:
    bid_str = str(bid)
    if bid_str in player_map:
        display_name = f"{player_map[bid_str]}" # (ID 제거)
        valid_batter_options.append((display_name, bid_str))
    elif not bid_str.isdigit():
        valid_batter_options.append((bid_str, bid_str))

valid_batter_options.sort(key=lambda x: x[0])

bat_names = [opt[0] for opt in valid_batter_options]
bat_ids = [opt[1] for opt in valid_batter_options]

raw_pitcher_ids = encoders['pitcher'].classes_
valid_pitcher_options = []
for pid in raw_pitcher_ids:
    pid_str = str(pid)
    if pid_str in player_map:
        display = f"{player_map[pid_str]}" # (ID 제거)
        valid_pitcher_options.append((display, pid_str))
    elif not pid_str.isdigit():
        valid_pitcher_options.append((pid_str, pid_str))

valid_pitcher_options.sort(key=lambda x: x[0])
pit_names = [opt[0] for opt in valid_pitcher_options]
pit_ids = [opt[1] for opt in valid_pitcher_options]

with st.container():
    c1, c2, c3 = st.columns([2, 1, 1])
    
    def_idx = 0
    for i, name in enumerate(bat_names):
        if "Shohei" in name:
            def_idx = i
            break
            
    if not bat_names:
        st.error("데이터 로드 실패: 선수 목록이 비어있습니다. player_mapping.pkl을 확인하세요.")
        st.stop()

    sel_bat_name = c1.selectbox("상대 타자 (Batter)", bat_names, index=def_idx)
    selected_batter_id = bat_ids[bat_names.index(sel_bat_name)]
    
    stand = c2.selectbox("타자 타석", ['L', 'R'])
    p_throws = c3.selectbox("투수 손", ['R', 'L'])
    
    c4, c5, c6, c7 = st.columns(4)
    b = c4.number_input("Ball", 0, 3, 0)
    s = c5.number_input("Strike", 0, 2, 0)
    o = c6.number_input("Out", 0, 2, 0)
    runners = c7.multiselect("주자", ["1루", "2루", "3루"])
    on_1b, on_2b, on_3b = "1루" in runners, "2루" in runners, "3루" in runners

    with st.expander("투수 설정 (선택 사항)"):
        if pit_names:
            sel_pit_name = st.selectbox("투수 (Pitcher)", pit_names)
            selected_pitcher_id = pit_ids[pit_names.index(sel_pit_name)]
        else:
            st.write("투수 정보 없음")
            selected_pitcher_id = "0"
        st.caption("※ 투수를 선택하지 않아도 AI가 평균적인 구위를 가정하여 추천합니다.")

st.divider()

if st.button("📢 최적의 투구 전략 추천받기", type="primary", use_container_width=True):
    with st.spinner("AI가 72가지 시나리오를 시뮬레이션 중..."):
        recs = recommend_strategy(selected_batter_id, selected_pitcher_id, stand, p_throws, b, s, o, on_1b, on_2b, on_3b)
        
        st.success(f"**{sel_bat_name}** 선수를 잡아낼 Top 3 전략입니다!")
        
        col_list, col_viz = st.columns([1.3, 1])
        with col_list:
            for rank, r in enumerate(recs):
                p_name = COMMON_PITCHES.get(r['pitch'], r['pitch'])
                z_desc = get_zone_desc(r['zone'], stand)
                st.markdown(f"""
                <div class="rec-card">
                    <h3 style="margin:0; color:#228BE6;">#{rank+1} {p_name}</h3>
                    <p style="font-size:1.1rem; font-weight:bold; margin:5px 0;">🎯 {z_desc} (Zone {r['zone']})</p>
                    <div style="background-color:#E9ECEF; border-radius:5px; height:10px; width:100%; margin-top:10px;">
                        <!-- [수정] float() 변환으로 에러 해결 -->
                        <div style="background-color:#40C057; width:{float(r['prob_success'])*100}%; height:100%; border-radius:5px;"></div>
                    </div>
                    <p style="font-size:0.9rem; color:#868E96; margin-top:5px;">
                        성공 확률: <b>{float(r['prob_success'])*100:.1f}%</b> (피안타 위험: {float(r['prob_fail'])*100:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col_viz:
            fig = go.Figure()
            fig.add_shape(type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5, line=dict(color="Black", width=3))
            for x in [-0.27, 0.27]: fig.add_shape(type="line", x0=x, y0=1.5, x1=x, y1=3.5, line=dict(color="Gray", width=1, dash="dot"))
            for z in [2.16, 2.83]: fig.add_shape(type="line", x0=-0.83, y0=z, x1=0.83, y1=z, line=dict(color="Gray", width=1, dash="dot"))
            
            colors = ['#FFD700', '#C0C0C0', '#CD7F32']
            for i, r in enumerate(recs):
                px, pz = get_zone_coords(r['zone'])
                fig.add_trace(go.Scatter(
                    x=[px], y=[pz], mode='markers+text',
                    marker=dict(size=50, color=colors[i], opacity=0.9, line=dict(width=2, color='white')),
                    text=str(i+1), textfont=dict(size=24, color='black', family="Arial Black"),
                    name=f"#{i+1} {r['pitch']}",
                    hovertemplate=f"<b>#{i+1} {r['pitch']}</b><br>Success: {float(r['prob_success'])*100:.1f}%"
                ))
            fig.add_shape(type="path", path="M -0.83 0 L 0.83 0 L 0.83 -0.5 L 0 -1.0 L -0.83 -0.5 Z", fillcolor="#DEE2E6", line_width=0)
            fig.update_layout(width=400, height=500, xaxis=dict(range=[-1.5, 1.5], visible=False), yaxis=dict(range=[0, 4.5], visible=False), margin=dict(l=10, r=10, t=30, b=10), title="포수 시점 (Catcher View)", showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

