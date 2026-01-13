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
from datetime import datetime

# 한글 및 시각화 설정 (창 꺼짐 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
plt.rcParams['axes.unicode_minus'] = False 

# ================= [기태님의 자산 관리 설정] =================
CAPITAL_KRW = 23000000   # IEX 매수 후 잔고 반영
RISK_RATIO = 0.01        # 계좌당 리스크 1%

# [보유 종목] 매수 후 여기에 추가하면 자동으로 분석에서 제외 및 추적 시작
MY_POSITIONS = [
    {'ticker': 'GOOGL', 'price': 201.935, 'qty': 69, 'entry_date': '2025-08-13'},
    {'ticker': 'IEX', 'price': 186.77, 'qty': 35, 'entry_date': '2026-01-13'}
]

MACRO_ASSETS = {
    '^GSPC': 'S&P 500', '^IXIC': '나스닥 종합', '^SOX': '필라델피아 반도체',
    'GLD': '금(Gold)', 'SLV': '은(Silver)', 'USO': '원유(Crude)',
    'UUP': '달러인덱스', '^TNX': '미 10년 금리'
}

SECTOR_MAP = {
    'Information Technology': '정보기술(IT)', 'Health Care': '헬스케어', 
    'Financials': '금융', 'Consumer Discretionary': '임의소비재', 
    'Communication Services': '커뮤니케이션', 'Industrials': '산업재', 
    'Consumer Staples': '필수소비재', 'Energy': '에너지', 
    'Utilities': '유틸리티', 'Real Estate': '부동산', 'Materials': '소재',
    'Technology': '기술주(SOX)'
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
        if df is None or len(df) < 20: return None
        df = df.copy()
        df['tr'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['atr'] = df['tr'].ewm(span=20, adjust=False).mean()
        df['ma200'] = df['Close'].rolling(200).mean()
        df['exit_l'] = df['Low'].rolling(10).min() # 익절가: 10일 저점
        p_dm, m_dm = df['High'].diff(), df['Low'].diff()
        tr_s = df['tr'].ewm(span=14, adjust=False).mean()
        df['adx'] = (100 * (p_dm.ewm(span=14).mean()/tr_s - abs(m_dm).ewm(span=14).mean()/tr_s).abs() / 
                     (p_dm.ewm(span=14).mean()/tr_s + abs(m_dm).ewm(span=14).mean()/tr_s)).ewm(span=14).mean()
        return df

    def save_position_chart(self, ticker, df, buy_price):
        """보유 종목 전용 추적 차트 생성"""
        os.makedirs("Charts", exist_ok=True)
        plt.figure(figsize=(12, 6))
        plot_data = df.tail(60)
        
        plt.plot(plot_data.index, plot_data['Close'], label='현재가', color='white', linewidth=2)
        plt.axhline(y=buy_price, color='gold', linestyle='--', label=f'매수가 (${buy_price})')
        plt.plot(plot_data.index, plot_data['Close'] - (2 * plot_data['atr']), color='red', alpha=0.5, label='손절선 (ATR)')
        plt.step(plot_data.index, plot_data['exit_l'], color='cyan', where='post', label='익절선 (10D Low)')
        
        plt.title(f"{ticker} 포지션 관리 차트", color='white', fontsize=14)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.1)
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().set_facecolor('#121212')
        plt.tick_params(colors='white')
        plt.savefig(f"Charts/{ticker}_tracking.png")
        plt.close()

    def generate_html_report(self, top_3, excluded, my_status):
        today_str = datetime.now().strftime("%Y%m%d")
        os.makedirs("Reports", exist_ok=True)
        
        def make_table(data_list, is_pos=False):
            if not data_list: return "<p>해당 없음</p>"
            rows = ""
            if is_pos:
                for r in data_list:
                    color = "#ff4757" if r['profit'] < 0 else "#2ecc71"
                    rows += f"<tr><td><b>{r['ticker']}</b></td><td>${r['buy']:.2f}</td><td>${r['curr']:.2f}</td><td style='color:{color}'>{r['profit']:+.2f}%</td><td>${r['stop']:.2f}</td><td>${r['exit']:.2f}</td><td>{r['status']}</td></tr>"
                cols = "<th>티커</th><th>매수가</th><th>현재가</th><th>수익률</th><th>손절가</th><th>익절가(10D)</th><th>상태</th>"
            else:
                for r in data_list:
                    unit = int(self.risk_money / (r['atr'] * 2 * self.usd_krw))
                    rows += f"<tr><td>{r['ticker']}</td><td>{r['score']}</td><td>${r['close']:.2f}</td><td>{unit}주</td><td>{r['max_corr']:.2f}</td><td>{r['perf_3m']:.1%}</td></tr>"
                cols = "<th>티커</th><th>점수</th><th>현재가</th><th>수량</th><th>상관성</th><th>3M수익</th>"
            return f"<table><tr>{cols}</tr>{rows}</table>"

        html = f"""
        <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
        <style>
        body{{background:#121212;color:#eee;font-family:sans-serif;padding:20px;line-height:1.6;}}
        .card{{background:#1e1e1e;border-radius:12px;padding:25px;margin-bottom:30px;border:1px solid #333;box-shadow: 0 4px 15px rgba(0,0,0,0.5);}}
        h1,h2{{color:gold; border-bottom: 1px solid #444; padding-bottom:10px;}} 
        table{{width:100%;border-collapse:collapse;margin-top:15px;}} th,td{{border:1px solid #444;padding:12px;text-align:left;}} th{{background:#333; color:gold;}}
        .chart-img{{width:48%; margin:1%; border-radius:10px; border:2px solid #444; transition: transform 0.3s;}}
        .chart-img:hover {{transform: scale(1.02);}}
        </style></head><body>
        <h1>📊 기태님 개인 자산 관리 대시보드 ({datetime.now().strftime("%Y-%m-%d %H:%M")})</h1>
        <div class="card"><h2>✅ [계좌 현황] 내 보유 종목</h2>{make_table(my_status, is_pos=True)}</div>
        <div class="card"><h2>📈 [관리 차트] 수익 추적 및 익절 타이밍</h2>
            {''.join([f'<div style="display:inline-block; width:100%; text-align:center;"><img src="../Charts/{p["ticker"]}_tracking.png" class="chart-img" style="width:90%;"></div>' for p in MY_POSITIONS])}
        </div>
        <div class="card"><h2>🥇 [신규 추천] 오늘의 TOP 3</h2>{make_table(top_3.to_dict('records'))}</div>
        <div class="card"><h2>⚠️ [분석 제외] 기존 보유 및 상관성 높은 종목</h2>{make_table(excluded.head(5).to_dict('records'))}</div>
        </body></html>
        """
        for path in [f"Reports/Report_{today_str}.html", "index.html"]:
            with open(path, "w", encoding="utf-8") as f: f.write(html)

    def run(self):
        sp_list, nq_list, sox_list, sp_sectors = get_indices_data()
        my_tickers = [p['ticker'].strip().upper() for p in MY_POSITIONS]
        all_tickers = sorted(list(set(sp_list + nq_list + sox_list + list(MACRO_ASSETS.keys()) + my_tickers)))
        
        print(f"\n>>> [시스템 가동] {len(all_tickers)}개 자산 분석 및 추적 차트 생성 중...")
        data = yf.download(all_tickers, period="2y", auto_adjust=True, group_by='ticker', progress=False)
        spy_perf = (data['^GSPC']['Close'].iloc[-1] / data['^GSPC']['Close'].iloc[-63]) - 1
        
        # 1. 내 종목 상태 분석 및 차트 저장
        my_status = []
        holdings_data = {}
        for p in MY_POSITIONS:
            t = p['ticker']
            df = self.calculate_indicators(data[t].dropna())
            holdings_data[t] = df['Close']
            curr = df['Close'].iloc[-1]
            stop = curr - (2 * df['atr'].iloc[-1])
            exit_l = df['exit_l'].iloc[-1]
            status = "보유(Keep)" if curr > stop and curr > exit_l else "⚠️ 매도검토"
            my_status.append({'ticker':t, 'buy':p['price'], 'curr':curr, 'profit':(curr/p['price']-1)*100, 'stop':stop, 'exit':exit_l, 'status':status})
            self.save_position_chart(t, df, p['price'])

        # 2. 신규 종목 스캔 (보유 중인 종목은 철저히 필터링)
        all_signals = []
        for t in all_tickers:
            if t in my_tickers or t in MACRO_ASSETS: continue
            df = self.calculate_indicators(data[t].dropna())
            if df is None: continue
            
            # 슈퍼리드 스코어 계산
            score = 0
            curr = df.iloc[-1]
            if curr['Close'] > curr['ma200']: score += 60
            if curr['adx'] > 25: score += 40
            
            if score >= 75.0:
                max_corr = max([df['Close'].corr(h_close) for h_close in holdings_data.values()])
                all_signals.append({'ticker': t, 'close': curr['Close'], 'atr': curr['atr'], 
                                   'score': score, 'max_corr': max_corr, 'perf_3m': (curr['Close']/df['Close'].iloc[-63]-1),
                                   'sector': sp_sectors.get(t, "기타")})

        # 3. 데이터 통합 및 파일 출력
        df_all = pd.DataFrame(all_signals).drop_duplicates('ticker')
        if not df_all.empty:
            passed = df_all[df_all['max_corr'] < 0.5].sort_values('score', ascending=False)
            top_3 = passed.groupby('sector').head(1).sort_values('score', ascending=False).head(3)
            excluded = df_all[df_all['max_corr'] >= 0.5].sort_values('score', ascending=False)
            self.generate_html_report(top_3, excluded, my_status)
            
            if not top_3.empty:
                with open("target.txt", "w", encoding="utf-8") as f:
                    best = top_3.iloc[0]
                    f.write(f"{best['ticker']},{int(self.risk_money / (best['atr'] * 2 * self.usd_krw))}")

        # 4. GitHub 자동 동기화
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Report Update: {datetime.now().strftime('%Y%m%d')}"], check=True)
            subprocess.run(["git", "push"], check=True)
            print(">>> [알림] 깃허브 업로드 및 리포트 갱신 완료.")
        except: pass

if __name__ == "__main__":
    UltimateGiTaeSystem(CAPITAL_KRW).run()