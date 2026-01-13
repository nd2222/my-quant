import yfinance as yf
import pandas as pd
import numpy as np
import sys
import requests
import io
import time
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from datetime import datetime

# =========================================================
# [시스템 설정] 한글 깨짐 방지 및 시각화 초기화
# =========================================================
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
plt.rcParams['axes.unicode_minus'] = False 

try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except:
    print(">>> [경고] 맑은 고딕 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

# ================= [기태님의 자산 관리 설정] =================
CAPITAL_KRW = 23000000   # IEX 매수 잔고 반영
RISK_RATIO = 0.01        # 계좌당 리스크 1%

# [보유 종목 관리] 매수한 종목은 여기에 추가하면 자동 관리됨
MY_POSITIONS = [
    {'ticker': 'GOOGL', 'price': 201.935, 'qty': 69, 'entry_date': '2025-08-13'},
    {'ticker': 'IEX', 'price': 186.77, 'qty': 35, 'entry_date': '2026-01-13'}
]

MACRO_ASSETS = {
    '^GSPC': 'S&P 500', '^IXIC': '나스닥 종합', '^SOX': '필라델피아 반도체',
    'GLD': '금(Gold)', 'SLV': '은(Silver)', 'USO': '원유(Crude)',
    'UUP': '달러인덱스', '^TNX': '미 10년 금리'
}

def get_realtime_rate():
    try: return yf.Ticker("KRW=X").history(period="1d")['Close'].iloc[-1]
    except: return 1468.0

def get_indices_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        sp_df = pd.read_html(io.StringIO(requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers).text), flavor='lxml')[0]
        sp_sectors = dict(zip(sp_df['Symbol'].str.replace('.', '-'), sp_df['GICS Sector']))
        nq_list = sorted(pd.read_html(io.StringIO(requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=headers).text), flavor='lxml')[4]['Ticker'].tolist())
        sox_list = sorted(['AMD', 'ADI', 'ASML', 'AMAT', 'AVGO', 'ARM', 'GFS', 'INTC', 'KLAC', 'LRCX', 'MRVL', 'MCHP', 'MU', 'MPWR', 'NVDA', 'NXPI', 'ON', 'QCOM', 'RMBS', 'STM', 'SWKS', 'TSM', 'TER', 'TXN', 'UMC', 'WOLF', 'LSCC', 'ENTG', 'QRVO'])
        return list(sp_sectors.keys()), nq_list, sox_list, sp_sectors
    except: return [], [], [], {}

class UltimateGiTaeSystem:
    def __init__(self, capital):
        self.capital = capital
        self.usd_krw = get_realtime_rate()
        self.risk_money = capital * RISK_RATIO

    # [핵심 1] 기술적 지표 정밀 계산 (복구됨)
    def calculate_indicators(self, df):
        if df is None or len(df) < 200: return None
        df = df.copy()
        
        # ATR (변동성)
        df['tr'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['atr'] = df['tr'].ewm(span=20, adjust=False).mean()
        df['atr_ma50'] = df['atr'].rolling(50).mean()
        
        # 이동평균선
        df['ma20'] = df['Close'].rolling(20).mean()
        df['ma200'] = df['Close'].rolling(200).mean()
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        
        # 10일 저점 (Trailing Stop 익절 라인)
        df['exit_l'] = df['Low'].rolling(10).min()
        
        # ADX (추세 강도)
        p_dm = df['High'].diff()
        m_dm = df['Low'].diff()
        p_dm = p_dm.where((p_dm > m_dm) & (p_dm > 0), 0.0)
        m_dm = -m_dm.where((m_dm > p_dm) & (m_dm > 0), 0.0)
        tr_s = df['tr'].ewm(span=14, adjust=False).mean()
        p_di = 100 * (p_dm.ewm(span=14).mean() / tr_s)
        m_di = 100 * (m_dm.ewm(span=14).mean() / tr_s)
        df['adx'] = (100 * abs(p_di - m_di) / (p_di + m_di)).ewm(span=14).mean()
        
        return df

    # [핵심 2] 슈퍼리드 전략 점수 계산 엔진 (완전 복구됨)
    def calculate_super_lead_score(self, curr, df, spy_perf):
        score = 0
        
        # 1. 추세 점수 (기본 60점)
        if curr['Close'] > curr['ma200']: score += 30
        h55 = df['High'].rolling(55).max().iloc[-2] # 55일 신고가
        if curr['Close'] > h55: score += 30
        
        # 2. 모멘텀 점수 (ADX, 거래량)
        score += min(20, (curr['adx'] / 45) * 20)
        vol_r = curr['Volume'] / curr['vol_ma20'] if curr['vol_ma20'] > 0 else 1
        score += min(20, (vol_r / 2.0) * 20)
        
        # 3. 변동성 축소 (Squeeze) 보너스
        squeeze = 1.2 if curr['atr'] < curr['atr_ma50'] else 0.9
        
        # 4. 시장 대비 초과 수익 (Alpha) 보너스
        perf_3m = (curr['Close'] / df['Close'].iloc[-63]) - 1 if len(df) > 63 else 0
        alpha = 1.25 if perf_3m > spy_perf else 1.0
        
        final_score = score * squeeze * alpha
        
        # 5. 과열 패널티 (이격도 과다 시 제외)
        if curr['Close'] > curr['ma20'] * 1.08: return 0.0
        
        return round(final_score, 2)

    def save_position_chart(self, ticker, df, buy_price):
        """보유 종목 추적 차트 (한글 폰트 적용)"""
        os.makedirs("Charts", exist_ok=True)
        plt.figure(figsize=(12, 6))
        plot_data = df.tail(60)
        
        plt.plot(plot_data.index, plot_data['Close'], label='현재가', color='white', linewidth=2)
        plt.axhline(y=buy_price, color='gold', linestyle='--', label=f'매수가 (${buy_price})')
        plt.plot(plot_data.index, plot_data['Close'] - (2 * plot_data['atr']), color='red', alpha=0.5, label='손절선 (2ATR)')
        plt.step(plot_data.index, plot_data['exit_l'], color='cyan', where='post', label='익절선 (10일 최저)')
        
        plt.title(f"{ticker} 수익 관리 차트", color='white', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.2, linestyle='--')
        
        # 다크 모드 스타일링
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().set_facecolor('#121212')
        plt.tick_params(colors='white')
        for spine in plt.gca().spines.values(): spine.set_color('#555')
            
        plt.savefig(f"Charts/{ticker}_tracking.png", dpi=100, bbox_inches='tight')
        plt.close()

    def generate_html_report(self, top_3, excluded, my_status):
        today_str = datetime.now().strftime("%Y%m%d")
        full_now = datetime.now().strftime("%Y-%m-%d %H:%M")
        os.makedirs("Reports", exist_ok=True)
        
        def make_table(data_list, is_pos=False):
            if not data_list: return "<p style='text-align:center; color:#777;'>데이터 없음</p>"
            rows = ""
            if is_pos:
                for r in data_list:
                    color = "#ff4757" if r['profit'] < 0 else "#2ecc71"
                    rows += f"<tr><td><b>{r['ticker']}</b></td><td>${r['buy']:.2f}</td><td>${r['curr']:.2f}</td><td style='color:{color}; font-weight:bold;'>{r['profit']:+.2f}%</td><td>${r['stop']:.2f}</td><td>${r['exit']:.2f}</td><td>{r['status']}</td></tr>"
                cols = "<th>종목</th><th>매수가</th><th>현재가</th><th>수익률</th><th>손절가(Risk)</th><th>익절가(Trail)</th><th>상태</th>"
            else:
                for r in data_list:
                    unit = int(self.risk_money / (r['atr'] * 2 * self.usd_krw))
                    rows += f"<tr><td><b>{r['ticker']}</b></td><td>{r['score']}</td><td>${r['close']:.2f}</td><td>{unit}주</td><td>{r['max_corr']:.2f}</td><td>{r['perf_3m']:.1%}</td></tr>"
                cols = "<th>종목</th><th>전략점수</th><th>현재가</th><th>추천수량</th><th>상관성</th><th>3개월수익</th>"
            return f"<table><tr>{cols}</tr>{rows}</table>"

        html = f"""
        <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
        <style>
        body{{background:#121212;color:#e0e0e0;font-family:'Malgun Gothic', sans-serif;padding:20px;line-height:1.6;}}
        .container{{max-width:1200px;margin:auto;}}
        .card{{background:#1e1e1e;border-radius:12px;padding:25px;margin-bottom:30px;border:1px solid #333;box-shadow:0 4px 15px rgba(0,0,0,0.5);}}
        h1,h2{{color:#f1c40f;border-bottom:2px solid #333;padding-bottom:10px;}} 
        table{{width:100%;border-collapse:collapse;margin-top:15px;font-size:15px;}} 
        th{{background:#2c3e50;color:#f1c40f;padding:12px;text-align:left;border:1px solid #444;}}
        td{{border:1px solid #444;padding:12px;}}
        .chart-container{{display:flex;flex-wrap:wrap;justify-content:space-between;}}
        .chart-box{{width:49%;margin-bottom:20px;text-align:center;}}
        .chart-img{{width:100%;border-radius:8px;border:2px solid #444;transition:transform 0.2s;}}
        .chart-img:hover{{transform:scale(1.02);border-color:#f1c40f;}}
        </style></head><body><div class="container">
        <h1>📊 기태님 자산 관리 대시보드 ({full_now})</h1>
        
        <div class="card"><h2>✅ [MY] 보유 종목 현황</h2>{make_table(my_status, is_pos=True)}</div>
        
        <div class="card"><h2>📈 [MY] 수익 관리 차트 (익절선 추적)</h2>
            <div class="chart-container">
            {''.join([f'<div class="chart-box"><img src="../Charts/{p["ticker"]}_tracking.png" class="chart-img"></div>' for p in MY_POSITIONS])}
            </div>
        </div>
        
        <div class="card"><h2>🥇 오늘의 추천 TOP 3 (기보유 제외)</h2>{make_table(top_3.to_dict('records'))}</div>
        
        <div class="card"><h2>⚠️ 분석 제외 (보유중/고상관성)</h2>{make_table(excluded.head(10).to_dict('records'))}</div>
        </div></body></html>
        """
        for path in [f"Reports/Report_{today_str}.html", "index.html"]:
            with open(path, "w", encoding="utf-8") as f: f.write(html)

    def run(self):
        sp_list, nq_list, sox_list, sp_sectors = get_indices_data()
        my_tickers = [p['ticker'].strip().upper() for p in MY_POSITIONS]
        all_tickers = sorted(list(set(sp_list + nq_list + sox_list + list(MACRO_ASSETS.keys()) + my_tickers)))
        
        print(f"\n>>> [시스템] 총 {len(all_tickers)}개 자산 정밀 분석 시작...")
        # 데이터 다운로드 (진행상황 표시 안함)
        data = yf.download(all_tickers, period="2y", auto_adjust=True, group_by='ticker', progress=False)
        
        # SPY 수익률 계산 (시장 대비 비교용)
        spy_perf = (data['^GSPC']['Close'].iloc[-1] / data['^GSPC']['Close'].iloc[-63]) - 1
        
        # 1. 내 종목 분석 및 차트 생성
        my_status = []
        holdings_data = {}
        for p in MY_POSITIONS:
            t = p['ticker']
            if t not in data.columns.levels[0]: continue
            
            df = self.calculate_indicators(data[t].dropna())
            holdings_data[t] = df['Close'] # 상관성 계산용 데이터 저장
            
            curr = df['Close'].iloc[-1]
            stop = curr - (2 * df['atr'].iloc[-1])
            exit_l = df['exit_l'].iloc[-1]
            
            # 상태 판단
            status = "⚠️ 매도신호" if curr < exit_l else ("⚠️ 손절위험" if curr < stop else "보유(Keep)")
            
            my_status.append({
                'ticker': t, 
                'buy': p['price'], 
                'curr': curr, 
                'profit': (curr/p['price']-1)*100, 
                'stop': stop, 
                'exit': exit_l, 
                'status': status
            })
            self.save_position_chart(t, df, p['price'])
            print(f">>> [보유] {t} 분석 완료 ({status})")

        # 2. 신규 추천 종목 발굴
        all_signals = []
        print("\n>>> [탐색] 신규 종목 스캐닝 중...")
        
        for t in all_tickers:
            # 내 종목이나 지수는 추천 대상 아님
            if t in my_tickers or t in MACRO_ASSETS: continue
            if t not in data.columns.levels[0]: continue
            
            df = self.calculate_indicators(data[t].dropna())
            if df is None: continue
            
            # 여기가 핵심: 슈퍼리드 전략 점수 계산
            score = self.calculate_super_lead_score(df.iloc[-1], df, spy_perf)
            
            if score >= 75.0:
                # 보유 종목들과의 최대 상관성 계산
                max_corr = 0
                if holdings_data:
                    max_corr = max([df['Close'].corr(h_close) for h_close in holdings_data.values()])
                
                s = {
                    'ticker': t, 
                    'close': df.iloc[-1]['Close'], 
                    'atr': df.iloc[-1]['atr'], 
                    'score': score, 
                    'max_corr': max_corr, 
                    'perf_3m': (df.iloc[-1]['Close']/df['Close'].iloc[-63]-1),
                    'sector': sp_sectors.get(t, "기타")
                }
                all_signals.append(s)

        # 3. 결과 정리 및 리포트 작성
        df_all = pd.DataFrame(all_signals).drop_duplicates('ticker')
        
        top_3 = pd.DataFrame()
        excluded = pd.DataFrame()
        
        if not df_all.empty:
            # 상관성 0.5 미만인 것만 통과
            passed = df_all[df_all['max_corr'] < 0.5].sort_values('score', ascending=False)
            # 섹터별 1위만 뽑아서 Top 3 선정
            top_3 = passed.groupby('sector').head(1).sort_values('score', ascending=False).head(3)
            # 탈락한 종목들 (상관성 높거나 점수 낮음)
            excluded = df_all[~df_all.index.isin(top_3.index)].sort_values('score', ascending=False)
            
            # 자동 매수 타겟 저장 (1위 종목)
            if not top_3.empty:
                best = top_3.iloc[0]
                unit = int(self.risk_money / (best['atr'] * 2 * self.usd_krw))
                with open("target.txt", "w", encoding="utf-8") as f:
                    f.write(f"{best['ticker']},{unit}")
                    
        self.generate_html_report(top_3, excluded, my_status)
        print(f">>> [완료] 리포트 생성 및 차트 저장 끝.")

        # 4. 깃허브 업로드
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Report Update: {datetime.now().strftime('%Y%m%d')}"], check=True)
            subprocess.run(["git", "push"], check=True)
            print(">>> [시스템] GitHub 동기화 완료.")
        except: pass

if __name__ == "__main__":
    UltimateGiTaeSystem(CAPITAL_KRW).run()