import yfinance as yf
import pandas as pd
import numpy as np
import sys
import requests
import io
import time
import os
import subprocess
from datetime import datetime

# 윈도우 한글 출력 보정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# ================= [설정] =================
CAPITAL_KRW = 34000000   
RISK_RATIO = 0.01        
MY_POSITIONS = [{'ticker': 'GOOGL', 'price': 201.935, 'qty': 69, 'entry_date': '2025-08-13'}]
MACRO_ASSETS = {'^GSPC': 'S&P 500', '^IXIC': '나스닥 종합', '^SOX': '필라델피아 반도체', 'GLD': '금', 'SLV': '은', 'USO': '원유', 'UUP': '달러', '^TNX': '미10년금리'}
SECTOR_MAP = {'Information Technology': 'IT', 'Health Care': '헬스케어', 'Financials': '금융', 'Consumer Discretionary': '임의소비재', 'Communication Services': '커뮤니케이션', 'Industrials': '산업재', 'Consumer Staples': '필수소비재', 'Energy': '에너지', 'Utilities': '유틸리티', 'Real Estate': '부동산', 'Materials': '소재', 'Technology': '기술주'}

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
        p_dm, m_dm = df['High'].diff(), df['Low'].diff()
        tr_s = df['tr'].ewm(span=14, adjust=False).mean()
        df['adx'] = (100 * (p_dm.ewm(span=14).mean()/tr_s - abs(m_dm).ewm(span=14).mean()/tr_s).abs() / 
                     (p_dm.ewm(span=14).mean()/tr_s + abs(m_dm).ewm(span=14).mean()/tr_s)).ewm(span=14).mean()
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
        final = score * squeeze * alpha
        if curr['Close'] > curr['ma20'] * 1.08: return 0.0
        return round(final, 2)

    def generate_html_report(self, top_3, excluded):
        today_str = datetime.now().strftime("%Y%m%d")
        os.makedirs("Reports", exist_ok=True)
        filename = f"Reports/Report_{today_str}.html"
        
        def make_table(data_list):
            if not data_list: return "<p>없음</p>"
            rows = ""
            for r in data_list:
                unit = int(self.risk_money / (r['atr'] * 2 * self.usd_krw))
                rows += f"<tr><td>{r['ticker']}</td><td>{r['score']}</td><td>${r['close']:.2f}</td><td>{unit}주</td><td>{r['max_corr']:.2f}</td></tr>"
            return f"<table><tr><th>티커</th><th>점수</th><th>현재가</th><th>수량</th><th>상관성</th></tr>{rows}</table>"

        html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>기태 퀀트</title><style>body{{background:#121212;color:#eee;font-family:sans-serif;}} table{{width:100%;border-collapse:collapse;}} th,td{{border:1px solid #444;padding:8px;}} th{{background:#333;color:gold;}}</style></head><body><h1>📊 기태 리포트 ({today_str})</h1><h2>🥇 Top 3</h2>{make_table(top_3.to_dict('records'))}</body></html>"""
        for path in ["index.html", filename]:
            with open(path, "w", encoding="utf-8") as f: f.write(html)
        
        # 깃허브 업로드
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Auto: {today_str}"], check=True)
            subprocess.run(["git", "push"], check=True)
        except: pass

    def run(self):
        sp_list, nq_list, sox_list, sp_sectors = get_indices_data()
        my_tickers = [p['ticker'] for p in MY_POSITIONS]
        all_tickers = sorted(list(set(sp_list + nq_list + sox_list + list(MACRO_ASSETS.keys()) + my_tickers)))
        
        print(f"\n>>> [분석 엔진] 총 {len(all_tickers)}개 자산 분석 시작...")
        data = yf.download(all_tickers, period="2y", auto_adjust=True, group_by='ticker', progress=False)
        spy_perf = (data['^GSPC']['Close'].iloc[-1] / data['^GSPC']['Close'].iloc[-63]) - 1
        holdings_data = {t: data[t]['Close'].dropna() for t in my_tickers}
        
        all_signals = []
        for t in all_tickers:
            if t in my_tickers or t not in data.columns.levels[0]: continue
            df = self.calculate_indicators(data[t].dropna())
            if df is None: continue
            score = self.calculate_super_lead_score(df.iloc[-1], df, spy_perf)
            if score >= 75.0:
                max_corr = max([df['Close'].corr(h_close) for h_close in holdings_data.values()])
                s = {'ticker': t, 'label': 'S', 'close': df.iloc[-1]['Close'], 'atr': df.iloc[-1]['atr'], 
                     'adx': df.iloc[-1]['adx'], 'exit_l': df['Low'].rolling(10).min().iloc[-1], 'score': score, 
                     'perf_3m': 0, 'sector': sp_sectors.get(t, "기타"), 'max_corr': max_corr, 
                     'stop': df.iloc[-1]['Close']-(2*df.iloc[-1]['atr'])}
                all_signals.append(s)

        df_all = pd.DataFrame(all_signals).drop_duplicates('ticker')
        if not df_all.empty:
            passed = df_all[df_all['max_corr'] < 0.5].sort_values('score', ascending=False)
            top_3 = passed.groupby('sector').head(1).sort_values('score', ascending=False).head(3)
            excluded = df_all[df_all['max_corr'] >= 0.5]
            
            self.generate_html_report(top_3, excluded)

            # ================= [핵심] 1위 종목 파일 저장 =================
            if not top_3.empty:
                best_stock = top_3.iloc[0]
                qty = int(self.risk_money / (best_stock['atr'] * 2 * self.usd_krw))
                
                # target.txt 파일에 "티커,수량" 형식으로 저장
                with open("target.txt", "w", encoding="utf-8") as f:
                    f.write(f"{best_stock['ticker']},{qty}")
                
                print(f"\n>>> 📝 [매수 신호 저장] {best_stock['ticker']} {qty}주 -> target.txt에 기록 완료!")
            else:
                # 살 게 없으면 파일을 비움
                with open("target.txt", "w", encoding="utf-8") as f: f.write("")
                print("\n>>> 💤 살만한 종목이 없어 매수 대상을 비웠습니다.")

if __name__ == "__main__":
    UltimateGiTaeSystem(CAPITAL_KRW).run()