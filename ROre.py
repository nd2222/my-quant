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
from datetime import datetime, timedelta

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
    pass

# ================= [기태님의 자산 관리 설정] =================
CAPITAL_KRW = 23000000   
RISK_RATIO = 0.01        

# [보유 종목] 
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

    def calculate_indicators(self, df):
        if df is None or len(df) < 200: return None
        df = df.copy()
        df['tr'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['atr'] = df['tr'].ewm(span=20, adjust=False).mean()
        df['atr_ma50'] = df['atr'].rolling(50).mean()
        df['ma20'] = df['Close'].rolling(20).mean()
        df['ma200'] = df['Close'].rolling(200).mean()
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        df['exit_l'] = df['Low'].rolling(10).min()
        
        p_dm = df['High'].diff()
        m_dm = df['Low'].diff()
        p_dm = p_dm.where((p_dm > m_dm) & (p_dm > 0), 0.0)
        m_dm = -m_dm.where((m_dm > p_dm) & (m_dm > 0), 0.0)
        tr_s = df['tr'].ewm(span=14, adjust=False).mean()
        p_di = 100 * (p_dm.ewm(span=14).mean() / tr_s)
        m_di = 100 * (m_dm.ewm(span=14).mean() / tr_s)
        df['adx'] = (100 * abs(p_di - m_di) / (p_di + m_di)).ewm(span=14).mean()
        return df

    def calculate_super_lead_score(self, curr, df, spy_perf):
        score = 0
        if curr['Close'] > curr['ma200']: score += 30
        h55 = df['High'].rolling(55).max().iloc[-2]
        if curr['Close'] > h55: score += 30
        score += min(20, (curr['adx'] / 45) * 20)
        vol_r = curr['Volume'] / curr['vol_ma20'] if curr['vol_ma20'] > 0 else 1
        score += min(20, (vol_r / 2.0) * 20)
        squeeze = 1.2 if curr['atr'] < curr['atr_ma50'] else 0.9
        perf_3m = (curr['Close'] / df['Close'].iloc[-63]) - 1 if len(df) > 63 else 0
        alpha = 1.25 if perf_3m > spy_perf else 1.0
        final_score = score * squeeze * alpha
        if curr['Close'] > curr['ma20'] * 1.08: return 0.0
        return round(final_score, 2)

    def save_position_chart(self, ticker, df, buy_price, entry_date):
        os.makedirs("Charts", exist_ok=True)
        
        # [수정] 매수일 - 10일부터 데이터 자르기
        try:
            start_dt = datetime.strptime(entry_date, "%Y-%m-%d") - timedelta(days=10)
            plot_data = df.loc[start_dt:]
        except:
            plot_data = df.tail(60) # 날짜 에러나면 그냥 최근 60일

        plt.figure(figsize=(14, 7)) # [수정] 그래프 크기 확대
        
        plt.plot(plot_data.index, plot_data['Close'], label='현재가', color='white', linewidth=2.5)
        plt.axhline(y=buy_price, color='gold', linestyle='--', linewidth=2, label=f'매수가 (${buy_price})')
        
        # 손절선, 익절선 그리기
        stop_line = plot_data['Close'] - (2 * plot_data['atr'])
        plt.plot(plot_data.index, stop_line, color='#ff4757', alpha=0.6, linewidth=1.5, label='손절선 (2ATR)')
        plt.step(plot_data.index, plot_data['exit_l'], color='#00d2d3', where='post', linewidth=2, label='익절선 (10일 최저)')
        
        # 매수 지점 표시
        try:
            plt.scatter(pd.to_datetime(entry_date), buy_price, color='gold', s=150, zorder=5, edgecolors='white', label='매수 체결')
        except: pass

        plt.title(f"{ticker} 수익 추적 차트 (매수일: {entry_date})", color='white', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper left', fontsize=12, facecolor='#2d3436', edgecolor='white')
        plt.grid(True, alpha=0.15, linestyle='--')
        
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().set_facecolor('#121212')
        plt.tick_params(colors='white', labelsize=11)
        for spine in plt.gca().spines.values(): spine.set_color('#555')
            
        plt.savefig(f"Charts/{ticker}_tracking.png", dpi=100, bbox_inches='tight')
        plt.close()

    def generate_html_report(self, macro_data, indices_results, gold_list, top_3, excluded, my_status):
        today_str = datetime.now().strftime("%Y%m%d")
        full_now = datetime.now().strftime("%Y-%m-%d %H:%M")
        os.makedirs("Reports", exist_ok=True)
        
        # [수정] 테이블 생성 함수: 내 종목 강조 기능 추가
        def make_table(data_list, is_pos=False):
            if not data_list: return "<p style='text-align:center; color:#777;'>데이터 없음</p>"
            rows = ""
            if is_pos:
                for r in data_list:
                    color = "#ff4757" if r['profit'] < 0 else "#2ecc71"
                    rows += f"<tr><td><b>{r['ticker']}</b></td><td>${r['buy']:.2f}</td><td>${r['curr']:.2f}</td><td style='color:{color}; font-weight:bold;'>{r['profit']:+.2f}%</td><td style='color:gold;'>{r['buy_score']}점</td><td style='color:white;'>{r['curr_score']}점</td><td>{r['status']}</td></tr>"
                cols = "<th>종목</th><th>매수가</th><th>현재가</th><th>수익률</th><th>매수당시 점수</th><th>현재 점수</th><th>상태</th>"
            else:
                for r in data_list:
                    # [핵심] 내 종목이면 강조 표시!
                    is_mine = r.get('is_mine', False)
                    row_style = "background-color: #2c3e50; border: 2px solid gold;" if is_mine else ""
                    ticker_disp = f"🏆 {r['ticker']} (보유중)" if is_mine else r['ticker']
                    
                    unit = int(self.risk_money / (r['atr'] * 2 * self.usd_krw))
                    rows += f"<tr style='{row_style}'><td><b>{ticker_disp}</b></td><td>{r['score']}</td><td>${r['close']:.2f}</td><td>{unit}주</td><td>{r['max_corr']:.2f}</td><td>{r['perf_3m']:.1%}</td></tr>"
                cols = "<th>종목</th><th>점수</th><th>현재가</th><th>수량</th><th>상관성</th><th>3M수익</th>"
            return f"<table><tr>{cols}</tr>{rows}</table>"

        # [수정] HTML 스타일: 이미지 클릭 시 확대(Lightbox) 기능 추가
        html = f"""
        <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
        <style>
        body{{background:#121212;color:#e0e0e0;font-family:'Malgun Gothic', sans-serif;padding:20px;line-height:1.6;}}
        .container{{max-width:1200px;margin:auto;}}
        .card{{background:#1e1e1e;border-radius:12px;padding:25px;margin-bottom:30px;border:1px solid #333;box-shadow:0 4px 15px rgba(0,0,0,0.5);}}
        h1,h2{{color:#f1c40f;border-bottom:2px solid #333;padding-bottom:10px;}} 
        table{{width:100%;border-collapse:collapse;margin-top:15px;font-size:14px;}} 
        th{{background:#2c3e50;color:#f1c40f;padding:12px;text-align:left;border:1px solid #444;}}
        td{{border:1px solid #444;padding:12px;}}
        
        /* 차트 스타일 & 클릭 확대 효과 */
        .chart-container{{display:flex;flex-wrap:wrap;justify-content:space-between;}}
        .chart-box{{width:48%;margin-bottom:20px;text-align:center;cursor:pointer;}}
        .chart-img{{width:100%;border-radius:8px;border:2px solid #444;transition:transform 0.2s;}}
        .chart-img:hover{{transform:scale(1.03);border-color:#f1c40f;}}
        
        /* 팝업(Lightbox) 스타일 */
        .lightbox {{display:none;position:fixed;z-index:999;padding-top:50px;left:0;top:0;width:100%;height:100%;background-color:rgba(0,0,0,0.9);}}
        .lightbox-content {{margin:auto;display:block;width:90%;max-width:1200px;}}
        .close {{position:absolute;top:15px;right:35px;color:#f1c40f;font-size:40px;font-weight:bold;cursor:pointer;}}
        
        .bull {{color: #ff4757;}} .bear {{color: #2e86de;}}
        </style>
        <script>
        function openModal(src) {{
            document.getElementById('myModal').style.display = "block";
            document.getElementById("img01").src = src;
        }}
        function closeModal() {{ document.getElementById('myModal').style.display = "none"; }}
        </script>
        </head><body><div class="container">
        
        <div id="myModal" class="lightbox">
          <span class="close" onclick="closeModal()">&times;</span>
          <img class="lightbox-content" id="img01">
        </div>

        <h1>📊 기태님 자산 관리 대시보드 ({full_now})</h1>
        
        <div class="card"><h2>[0] 글로벌 시장 요약</h2>
            <table><tr><th>항목</th><th>현재가</th><th>변동</th><th>상태</th></tr>
            {"".join(f"<tr><td>{n}</td><td>{v['curr']:.2f}</td><td class='{'bull' if v['pct']>0 else 'bear'}'>{v['pct']:+.2f}%</td><td>{v['status']}</td></tr>" for n, v in macro_data.items())}
            </table></div>

        <div class="card"><h2>✅ [MY] 보유 종목 점수 분석 (매수 vs 현재)</h2>{make_table(my_status, is_pos=True)}</div>
        
        <div class="card"><h2>📈 [MY] 수익 관리 차트 (클릭하여 확대)</h2>
            <div class="chart-container">
            {''.join([f'<div class="chart-box" onclick="openModal(\'../Charts/{p["ticker"]}_tracking.png\')"><img src="../Charts/{p["ticker"]}_tracking.png" class="chart-img"></div>' for p in MY_POSITIONS])}
            </div></div>
        
        <div class="card"><h2>🥇 최종 추천 TOP 3 (안전 분산)</h2>{make_table(top_3.to_dict('records'))}</div>
        <div class="card"><h2>🌟 초엄격 '슈퍼리드' 골든 리스트 (130점 이상)</h2>{make_table(gold_list)}</div>

        <div class="card"><h2>[2-1] 반도체(SOX) (총 {indices_results['2-1. 반도체(SOX)']['total']}개 중 {len(indices_results['2-1. 반도체(SOX)']['items'])}개 포착)</h2>{make_table(indices_results['2-1. 반도체(SOX)']['items'])}</div>
        <div class="card"><h2>[2-2] 나스닥100 (총 {indices_results['2-2. 나스닥100']['total']}개 중 {len(indices_results['2-2. 나스닥100']['items'])}개 포착)</h2>{make_table(indices_results['2-2. 나스닥100']['items'])}</div>
        <div class="card"><h2>[2-3] S&P 500 (총 {indices_results['2-3. S&P 500']['total']}개 중 {len(indices_results['2-3. S&P 500']['items'])}개 포착)</h2>{make_table(indices_results['2-3. S&P 500']['items'])}</div>

        <div class="card"><h2>⚠️ 중복 위험 종목 (Excluded)</h2>{make_table(excluded.head(10).to_dict('records'))}</div>
        </div></body></html>
        """
        for path in [f"Reports/Report_{today_str}.html", "index.html"]:
            with open(path, "w", encoding="utf-8") as f: f.write(html)

    def run(self):
        sp_list, nq_list, sox_list, sp_sectors = get_indices_data()
        my_tickers = [p['ticker'].strip().upper() for p in MY_POSITIONS]
        all_tickers = sorted(list(set(sp_list + nq_list + sox_list + list(MACRO_ASSETS.keys()) + my_tickers)))
        
        print(f"\n>>> [시스템] 총 {len(all_tickers)}개 자산 정밀 분석 시작...")
        data = yf.download(all_tickers, period="2y", auto_adjust=True, group_by='ticker', progress=False)
        spy_perf = (data['^GSPC']['Close'].iloc[-1] / data['^GSPC']['Close'].iloc[-63]) - 1
        
        # 0. 매크로
        macro_results = {}
        for ticker, name in MACRO_ASSETS.items():
            if ticker in data.columns.levels[0]:
                d = data[ticker].dropna()
                curr, prev = d['Close'].iloc[-1], d['Close'].iloc[-2]
                status = "강세 ☀️" if curr > d['Close'].rolling(200).mean().iloc[-1] else "약세 ⛈️"
                macro_results[name] = {'curr': curr, 'pct': (curr/prev-1)*100, 'status': status}

        # 1. 내 종목 분석
        my_status = []
        holdings_data = {}
        for p in MY_POSITIONS:
            t = p['ticker']
            if t not in data.columns.levels[0]: continue
            
            df_curr = self.calculate_indicators(data[t].dropna())
            curr_score = self.calculate_super_lead_score(df_curr.iloc[-1], df_curr, spy_perf)
            holdings_data[t] = df_curr['Close']
            
            # 매수 시점 점수 역산출
            try:
                df_buy_hist = data[t].loc[:p['entry_date']]
                spy_buy_hist = data['^GSPC'].loc[:p['entry_date']]
                spy_perf_buy = (spy_buy_hist['Close'].iloc[-1] / spy_buy_hist['Close'].iloc[-63]) - 1 if len(spy_buy_hist) > 63 else 0
                df_buy = self.calculate_indicators(df_buy_hist)
                buy_score = self.calculate_super_lead_score(df_buy.iloc[-1], df_buy, spy_perf_buy)
            except:
                buy_score = 0
                
            curr = df_curr['Close'].iloc[-1]
            stop = curr - (2 * df_curr['atr'].iloc[-1])
            exit_l = df_curr['exit_l'].iloc[-1]
            status = "⚠️ 매도신호" if curr < exit_l else ("⚠️ 손절위험" if curr < stop else "보유(Keep)")
            
            my_status.append({'ticker': t, 'buy': p['price'], 'curr': curr, 'profit': (curr/p['price']-1)*100, 'stop': stop, 'exit': exit_l, 'status': status, 'buy_score': buy_score, 'curr_score': curr_score})
            # [수정] 차트에 매수일 정보 전달
            self.save_position_chart(t, df_curr, p['price'], p['entry_date'])
            print(f">>> [보유] {t}: 현재 {curr_score}점")

        # 2. 인덱스별 전수조사 (내 종목 포함하여 강조 표시)
        indices_to_scan = [("2-1. 반도체(SOX)", sox_list), ("2-2. 나스닥100", nq_list), ("2-3. S&P 500", sp_list)]
        web_indices_results = {} 
        all_signals = []
        
        print("\n>>> [탐색] 인덱스별 전수조사 시작...")
        
        for idx_name, t_list in indices_to_scan:
            print(f"    - {idx_name} 스캔 중...")
            found_items = []
            
            for t in t_list:
                # [수정] 내 종목도 분석 대상에 포함 (단, 추천에선 제외하기 위해 플래그 표시)
                if t in MACRO_ASSETS: continue
                if t not in data.columns.levels[0]: continue
                
                df = self.calculate_indicators(data[t].dropna())
                if df is None: continue
                score = self.calculate_super_lead_score(df.iloc[-1], df, spy_perf)
                
                if score >= 75.0:
                    max_corr = 0
                    if holdings_data:
                        max_corr = max([df['Close'].corr(h_close) for h_close in holdings_data.values()])
                    
                    is_mine = t in my_tickers # 내 종목 여부 확인
                    
                    s = {'ticker': t, 'close': df.iloc[-1]['Close'], 'atr': df.iloc[-1]['atr'], 'score': score, 
                         'max_corr': max_corr, 'perf_3m': (df.iloc[-1]['Close']/df['Close'].iloc[-63]-1), 
                         'sector': sp_sectors.get(t, "기타"), 'is_mine': is_mine}
                    
                    found_items.append(s)
                    # 내 종목이 아니면 추천 후보로 등록
                    if not is_mine:
                        all_signals.append(s)
            
            web_indices_results[idx_name] = {'items': found_items, 'total': len(t_list)}

        # 3. 결과 정리
        df_all = pd.DataFrame(all_signals).drop_duplicates('ticker')
        top_3 = pd.DataFrame()
        excluded = pd.DataFrame()
        gold_list = []
        
        if not df_all.empty:
            gold_list = df_all[df_all['score'] >= 130].sort_values('score', ascending=False).to_dict('records')
            passed = df_all[df_all['max_corr'] < 0.5].sort_values('score', ascending=False)
            top_3 = passed.groupby('sector').head(1).sort_values('score', ascending=False).head(3)
            excluded = df_all[~df_all.index.isin(top_3.index)].sort_values('score', ascending=False)
            
            if not top_3.empty:
                best = top_3.iloc[0]
                unit = int(self.risk_money / (best['atr'] * 2 * self.usd_krw))
                with open("target.txt", "w", encoding="utf-8") as f:
                    f.write(f"{best['ticker']},{unit}")
                    
        self.generate_html_report(macro_results, web_indices_results, gold_list, top_3, excluded, my_status)
        print(f">>> [완료] 리포트 생성 끝.")

        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Report Update: {datetime.now().strftime('%Y%m%d')}"], check=True)
            subprocess.run(["git", "push"], check=True)
            print(">>> [시스템] GitHub 동기화 완료.")
        except: pass

if __name__ == "__main__":
    UltimateGiTaeSystem(CAPITAL_KRW).run()