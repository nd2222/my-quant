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

# ================= [기태님의 3,400만원 철통 자산 관리 설정] =================
CAPITAL_KRW = 34000000   
RISK_RATIO = 0.01        

MY_POSITIONS = [
    {'ticker': 'GOOGL', 'price': 201.935, 'qty': 69, 'entry_date': '2025-08-13'}
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
    try:
        rate = yf.Ticker("KRW=X").history(period="1d")['Close'].iloc[-1]
        return rate
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

    def print_detailed_row(self, s, prefix="  >"):
        unit = int(self.risk_money / (s['atr'] * 2 * self.usd_krw))
        sec_kr = SECTOR_MAP.get(s['sector'], s['sector'])
        rr_ratio = abs((s['close'] - s['exit_l']) / (s['close'] - s['stop'])) if abs(s['close'] - s['stop']) > 0 else 0
        print(f"{prefix} {s['ticker']:<5} ({sec_kr}): {s['label']} 돌파 [점수 {s['score']:.1f}]")
        print(f"      (수량 {unit:>3}주 | 가격 ${s['close']:<7.2f} | 3M수익 {s['perf_3m']:.1%})")
        print(f"      (손절 ${s['stop']:.2f} | 익절 ${s['exit_l']:.2f} | 손익비 {rr_ratio:.1f} | 상관성 {s['max_corr']:.2f})")
        print("")

    def generate_html_report(self, macro_data, indices_results, gold_list, top_3, excluded):
        """[방대한 데이터 통합] 웹 리포트 생성"""
        today_str = datetime.now().strftime("%Y%m%d")
        full_now = datetime.now().strftime("%Y-%m-%d %H:%M")
        os.makedirs("Reports", exist_ok=True)
        filename = f"Reports/Report_{today_str}.html"

        def make_table(data_list, highlight=False):
            if not data_list: return "<p>포착된 종목 없음</p>"
            rows = ""
            for r in data_list:
                rr = abs((r['close']-r['exit_l'])/(r['close']-r['stop'])) if abs(r['close']-r['stop'])>0 else 0
                unit = int(self.risk_money / (r['atr'] * 2 * self.usd_krw))
                rows += f"<tr class='{'rank-1' if highlight else ''}'><td>{r['ticker']}</td><td>{SECTOR_MAP.get(r['sector'], r['sector'])}</td><td>{r['score']}</td><td>${r['close']:.2f}</td><td>{unit}주</td><td>{rr:.1f}</td><td>{r['max_corr']:.2f}</td><td>{r['perf_3m']:.1%}</td></tr>"
            return f"<table><tr><th>티커</th><th>섹터</th><th>점수</th><th>현재가</th><th>수량</th><th>손익비</th><th>상관성</th><th>3M수익</th></tr>{rows}</table>"

        html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>기태 퀀트 리포트_{today_str}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .card {{ background: #1e1e1e; border-radius: 12px; padding: 20px; margin-bottom: 25px; border: 1px solid #333; }}
                h1 {{ color: #f1c40f; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #f1c40f; border-left: 5px solid #f1c40f; padding-left: 15px; margin-bottom: 15px; font-size: 1.4em; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
                th, td {{ border: 1px solid #333; padding: 12px; text-align: left; }}
                th {{ background: #2c2c2c; color: #f1c40f; }}
                .rank-1 {{ background: rgba(241, 196, 15, 0.1); }}
                .bull {{ color: #ff4757; }} .bear {{ color: #2e86de; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 기태 님 슈퍼리드 전수조사 리포트 ({full_now})</h1>
                <div class="card">
                    <h2>[0] 글로벌 시장 요약</h2>
                    <table><tr><th>항목</th><th>현재가</th><th>변동</th><th>상태</th></tr>
                    {"".join(f"<tr><td>{n}</td><td>{v['curr']:.2f}</td><td class='{'bull' if v['pct']>0 else 'bear'}'>{v['pct']:+.2f}%</td><td>{v['status']}</td></tr>" for n, v in macro_data.items())}
                    </table>
                </div>
                <div class="card"><h2>[1] 최종 추천 TOP 3 (안전 분산)</h2>{make_table(top_3.to_dict('records'), True)}</div>
                <div class="card"><h2>[2] 초엄격 '슈퍼리드' 골든 리스트</h2>{make_table(gold_list)}</div>
                <div class="card"><h2>[3-1] 반도체(SOX) 전수조사</h2>{make_table(indices_results['2-1. 반도체(SOX)'])}</div>
                <div class="card"><h2>[3-2] 나스닥100 전수조사</h2>{make_table(indices_results['2-2. 나스닥100'])}</div>
                <div class="card"><h2>[3-3] S&P 500 전수조사</h2>{make_table(indices_results['2-3. S&P 500'])}</div>
                <div class="card"><h2>[4] 중복 위험 종목 (High Correlation)</h2>{make_table(excluded.head(10).to_dict('records'))}</div>
            </div>
        </body>
        </html>
        """
        for path in ["index.html", filename]:
            with open(path, "w", encoding="utf-8") as f: f.write(html)
        print(f">>> [시스템] 날짜별 웹 리포트 생성 완료 ({filename})")

    def auto_git_push(self):
        try:
            print(">>> [시스템] GitHub 업로드 중 (파일이 많아 시간이 소요될 수 있습니다)...")
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Full Report Update: {datetime.now().strftime('%Y%m%d')}"], check=True)
            subprocess.run(["git", "push"], check=True)
            print(">>> [알림] 업로드 성공! https://nd2222.github.io/my-quant/")
        except Exception as e: print(f">>> [오류] 업로드 실패: {e}")

    def run(self):
        sp_list, nq_list, sox_list, sp_sectors = get_indices_data()
        my_tickers = [p['ticker'] for p in MY_POSITIONS]
        all_tickers = sorted(list(set(sp_list + nq_list + sox_list + list(MACRO_ASSETS.keys()) + my_tickers)))
        
        print(f"\n>>> [전략 엔진] 총 {len(all_tickers)}개 자산 정밀 분석 시작...")
        data = yf.download(all_tickers, period="2y", auto_adjust=True, group_by='ticker', progress=False)
        spy_perf = (data['^GSPC']['Close'].iloc[-1] / data['^GSPC']['Close'].iloc[-63]) - 1
        holdings_data = {t: data[t]['Close'].dropna() for t in my_tickers}

        macro_results = {}
        print("\n" + "="*95 + "\n [0] 글로벌 거시 지표 요약\n" + "-"*95)
        for ticker, name in MACRO_ASSETS.items():
            if ticker in data.columns.levels[0]:
                d = data[ticker].dropna()
                curr, prev = d['Close'].iloc[-1], d['Close'].iloc[-2]
                status = "강세 ☀️" if curr > d['Close'].rolling(200).mean().iloc[-1] else "약세 ⛈️"
                macro_results[name] = {'curr': curr, 'pct': (curr/prev-1)*100, 'status': status}
                print(f" ● {name:<15}: {curr:>10.2f} ({macro_results[name]['pct']:>+5.2f}%) | {status}")

        all_signals = []
        indices_to_scan = [("2-1. 반도체(SOX)", sox_list), ("2-2. 나스닥100", nq_list), ("2-3. S&P 500", sp_list)]
        web_indices_results = {name: [] for name, _ in indices_to_scan}
        
        for idx_name, t_list in indices_to_scan:
            print("\n" + "="*95 + f"\n [{idx_name}] 전수 조사 결과\n" + "-"*95)
            for i, t in enumerate(t_list, 1):
                sys.stdout.write(f"\r  ▶ {idx_name} 분석 진행률: {i}/{len(t_list)} ({t:<5})")
                sys.stdout.flush()
                
                if t in my_tickers or t not in data.columns.levels[0]: continue
                df = self.calculate_indicators(data[t].dropna())
                if df is None: continue
                score = self.calculate_super_lead_score(df.iloc[-1], df, spy_perf)
                
                if score >= 75.0:
                    max_corr = max([df['Close'].corr(h_close) for h_close in holdings_data.values()])
                    s = {'ticker': t, 'label': 'S2' if df.iloc[-1]['Close'] > df['High'].rolling(55).max().iloc[-2] else 'S1',
                         'close': df.iloc[-1]['Close'], 'atr': df.iloc[-1]['atr'], 'adx': df.iloc[-1]['adx'], 
                         'exit_l': df['Low'].rolling(10).min().iloc[-1], 'score': score, 
                         'perf_3m': (df.iloc[-1]['Close']/df['Close'].iloc[-63]-1), 
                         'sector': sp_sectors.get(t, "Technology" if t in sox_list else "기타"), 
                         'max_corr': max_corr, 'stop': df.iloc[-1]['Close']-(2*df.iloc[-1]['atr'])}
                    all_signals.append(s)
                    web_indices_results[idx_name].append(s)
                    print("\n")
                    self.print_detailed_row(s)
            print(f"\n  >>> {idx_name}: 총 {len(web_indices_results[idx_name])}개 포착.")

        df_all = pd.DataFrame(all_signals).drop_duplicates('ticker')
        if not df_all.empty:
            gold_list = df_all[df_all['score'] >= 130].sort_values('score', ascending=False).to_dict('records')
            passed = df_all[df_all['max_corr'] < 0.5].sort_values('score', ascending=False)
            excluded = df_all[df_all['max_corr'] >= 0.5].sort_values('score', ascending=False)
            top_3 = passed.groupby('sector').head(1).sort_values('score', ascending=False).head(3)
            
            print("\n" + "="*95 + "\n [4] 최종 추천 및 자동 업데이트\n" + "-"*95)
            for i, r in enumerate(top_3.to_dict('records'), 1): self.print_detailed_row(r, prefix=f"  🥇 {i}위")
            
            self.generate_html_report(macro_results, web_indices_results, gold_list, top_3, excluded)
            self.auto_git_push()

        input("\n[알림] 모든 작업 완료. 엔터를 누르면 종료됩니다.")

if __name__ == "__main__":
    UltimateGiTaeSystem(CAPITAL_KRW).run()