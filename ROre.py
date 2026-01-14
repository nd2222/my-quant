import yfinance as yf
import pandas as pd
import numpy as np
import sys
import requests
import io
import time
import os
import json
import webbrowser
import matplotlib.pyplot as plt
import subprocess
import traceback
from matplotlib import font_manager, rc
from datetime import datetime, timedelta
from tqdm import tqdm

# ================= [시스템 설정] =================
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)
plt.rcParams['axes.unicode_minus'] = False 

try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except: pass

# ================= [자산 및 설정] =================
CAPITAL_KRW = 23000000 
RISK_RATIO = 0.01 
AV_API_KEY = "I4VLTU5MYZY7RZL9"

DATA_DIR = r"C:\Quant\Data"
FIN_FILE = os.path.join(DATA_DIR, "financials.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("Charts", exist_ok=True)
os.makedirs("Reports", exist_ok=True)

MY_POSITIONS = [
    {'ticker': 'GOOGL', 'price': 201.935, 'qty': 69, 'entry_date': '2025-08-13'},
    {'ticker': 'IEX', 'price': 186.77, 'qty': 35, 'entry_date': '2026-01-13'}
]

MACRO_ASSETS = {
    '^GSPC': 'S&P 500', '^IXIC': '나스닥 종합', '^SOX': '필라델피아 반도체',
    'GLD': '금(Gold)', 'SLV': '은(Silver)', 'USO': '원유(Crude)',
    'UUP': '달러인덱스', '^TNX': '미 10년 금리'
}

class ExpertQuantSystem:
    def __init__(self, capital):
        self.capital = capital
        self.usd_krw = 1450.0 
        self.risk_money = capital * RISK_RATIO
        self.market_regime = "neutral"
        
        if os.path.exists(FIN_FILE):
            try:
                with open(FIN_FILE, 'r') as f:
                    data = json.load(f)
                    sample = next(iter(data.get('stocks', {})))
                    if 'sector' not in data['stocks'][sample]:
                        f.close()
                        os.remove(FIN_FILE)
            except: pass
            
        self.financials = self.load_financials()
        
        self.sector_risk_profile = {
            '필수소비재': 'defensive', '헬스케어': 'defensive', '유틸리티': 'defensive',
            '통신': 'defensive', '에너지': 'cyclical', '소재': 'cyclical', 
            '산업재': 'cyclical', '금융': 'cyclical', '임의소비재': 'cyclical',
            '기술': 'growth', '부동산': 'cyclical'
        }

    def load_financials(self):
        if os.path.exists(FIN_FILE):
            try:
                with open(FIN_FILE, 'r') as f:
                    data = json.load(f)
                    if (datetime.now() - datetime.strptime(data['update_date'], "%Y-%m-%d")).days > 3:
                        return {}
                    return data.get('stocks', {})
            except: pass
        return {}

    def detect_market_regime(self, spy_df):
        if spy_df is None or len(spy_df) < 60: return "neutral"
        curr = spy_df['Close'].iloc[-1]
        ma20 = spy_df['Close'].rolling(20).mean().iloc[-1]
        ma200 = spy_df['Close'].rolling(200).mean().iloc[-1]
        perf_1m = (curr / spy_df['Close'].iloc[-21]) - 1
        
        if curr < ma20 and perf_1m < -0.02: return "downtrend"
        elif curr > ma200 and perf_1m > 0.03: return "uptrend"
        else: return "neutral"

    def fetch_yf_info(self, ticker):
        if ticker in self.financials and self.financials[ticker].get('roe') != 0:
            return self.financials[ticker]

        stock = yf.Ticker(ticker)
        roe, sector = 0, "Unknown"
        try:
            info = stock.info
            roe = info.get('returnOnEquity', 0)
            sector = info.get('sector', 'Unknown')
        except: pass

        if roe == 0 or roe is None:
            try:
                fin = stock.financials
                bal = stock.balance_sheet
                net_income = fin.loc['Net Income'].iloc[0]
                equity = bal.loc['Stockholders Equity'].iloc[0]
                if equity > 0: roe = net_income / equity
            except: pass
            
        if roe is None: roe = 0
        roe = max(min(roe, 1.0), -1.0)
        
        sector_map = {'Technology': '기술', 'Financial Services': '금융', 'Healthcare': '헬스케어', 'Consumer Cyclical': '임의소비재', 'Industrials': '산업재', 'Energy': '에너지', 'Consumer Defensive': '필수소비재', 'Basic Materials': '소재', 'Real Estate': '부동산', 'Communication Services': '통신', 'Utilities': '유틸리티'}
        sector = sector_map.get(sector, sector)
        return {'roe': roe, 'sector': sector}

    def validate_data(self, df):
        if df is None or df.empty or len(df) < 150: return False
        if (datetime.now() - df.index[-1]).days > 5: return False 
        return True

    def calculate_indicators(self, df):
        df = df.copy()
        df['tr'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['atr'] = df['tr'].ewm(span=20).mean()
        df['atr_ma50'] = df['atr'].rolling(50).mean()
        df['ma20'] = df['Close'].rolling(20).mean()
        df['ma200'] = df['Close'].rolling(200).mean()
        df['exit_l'] = df['Low'].rolling(10).min()
        df['stop_loss'] = df['Close'] - (2 * df['atr'])
        
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = -df['Low'].diff().clip(upper=0)
        dx = 100 * abs(plus_dm.ewm(alpha=1/14).mean() - minus_dm.ewm(alpha=1/14).mean()) / (plus_dm.ewm(alpha=1/14).mean() + minus_dm.ewm(alpha=1/14).mean())
        df['adx'] = dx.ewm(alpha=1/14).mean()
        return df

    def calculate_adaptive_score(self, curr, df, spy_perf, roe, sector, max_corr=0):
        score = 0
        trend_score = 0
        if curr['Close'] > curr['ma200']:
            dist = min(1.0, (curr['Close'] - curr['ma200']) / curr['ma200'] * 5)
            trend_score += 10 + (10 * dist)
            high_60 = df['High'].rolling(60).max().iloc[-2]
            if high_60 > 0:
                prox = max(0, min(1.0, (curr['Close'] - high_60 * 0.85) / (high_60 * 0.15)))
                trend_score += 15 * prox
        
        mom_score = 0
        perf_3m = (curr['Close'] / df['Close'].iloc[-63]) - 1 if len(df) >= 63 else 0
        alpha = perf_3m - spy_perf
        if alpha > 0: mom_score += min(25, max(0, (alpha * 100) + 5))
        adx = curr['adx']
        mom_score += min(5, max(0, (adx - 20) / 10 * 5))
        
        qual_score = 0
        if roe > 0:
            if roe < 0.10: qual_score = roe * 150
            elif roe < 0.20: qual_score = 10 + (roe - 0.10) * 50 
            else: qual_score = 15
        vol_ratio = df['Volume'][-20:].mean() / max(df['Volume'][-60:-20].mean(), 1)
        qual_score += min(10, max(0, 5 + (vol_ratio - 1) * 2.5))
        
        risk_score = 0
        if curr['atr'] < curr['atr_ma50']: risk_score += 5
        low_10 = df['exit_l'].iloc[-1]
        if curr['ma20'] > low_10 and curr['Close'] > low_10:
            recov = min(1.0, (curr['Close'] - low_10) / (curr['ma20'] - low_10))
            risk_score += 5 * recov
            
        base_score = trend_score + mom_score + qual_score + risk_score
        sector_risk = self.sector_risk_profile.get(sector, 'cyclical')
        if sector_risk == 'defensive' and self.market_regime == 'downtrend': base_score += 5.0
        elif sector_risk == 'growth' and self.market_regime == 'uptrend': base_score += 5.0
            
        if max_corr > 0.6:
            if max_corr < 0.8: base_score -= (max_corr - 0.6) * 15
            else: base_score -= ((max_corr - 0.8) * 40) + 3.0
            
        return round(max(0, min(100, base_score)), 1)

    def save_white_chart(self, ticker, df, buy_price, entry_date, status, score, roe_val, sector):
        os.makedirs("Charts", exist_ok=True)
        try: start_dt = datetime.strptime(entry_date, "%Y-%m-%d") - timedelta(days=30)
        except: start_dt = df.index[-90]
        plot_data = df.loc[start_dt:]
        
        plt.figure(figsize=(14, 7), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('#fdfdfd')
        plt.plot(plot_data.index, plot_data['Close'], color='#2c3e50', linewidth=2, label='현재가')
        if buy_price > 0:
            plt.axhline(y=buy_price, color='#f1c40f', linestyle='--', linewidth=2, label=f'매수가 (${buy_price:.2f})')
        plt.step(plot_data.index, plot_data['exit_l'], color='#3498db', where='post', linewidth=1.5, label='익절 기준 (10일 저점)')
        plt.plot(plot_data.index, plot_data['stop_loss'], color='#e74c3c', linestyle=':', linewidth=1.5, label='손절 기준 (2ATR)')
        
        plt.title(f"[{ticker}] {sector} | 점수: {score}점 | ROE: {roe_val}", fontsize=16, fontweight='bold', color='#34495e')
        plt.legend(loc='upper left', shadow=True)
        plt.grid(True, alpha=0.2)
        plt.savefig(f"Charts/{ticker}_tracking.png", dpi=100, bbox_inches='tight')
        plt.close()

    def generate_html(self, macro_data, my_status, top3, gold_list, scan_results, excluded):
        full_now = datetime.now().strftime("%Y-%m-%d %H:%M")
        regime_msg = "하락장 방어 모드" if self.market_regime == "downtrend" else "추세 추종 모드"
        
        html = f"""
        <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
        <style>
            body{{background:#f4f7f6;color:#333;font-family:'Apple SD Gothic Neo',sans-serif;padding:30px;}}
            .container{{max-width:1280px;margin:auto;}}
            .card{{background:white;border-radius:12px;padding:25px;margin-bottom:25px;box-shadow:0 5px 15px rgba(0,0,0,0.05);}}
            h1{{color:#2c3e50;font-size:28px;margin-bottom:10px;border-bottom:3px solid #3498db;padding-bottom:15px;}}
            .badge{{background:#e74c3c;color:white;padding:5px 10px;border-radius:5px;font-size:14px;vertical-align:middle;}}
            h2{{color:#34495e;font-size:22px;margin-bottom:15px;border-left:5px solid #3498db;padding-left:15px;}}
            table{{width:100%;border-collapse:collapse;margin-top:15px;font-size:15px;}}
            th{{background:#f8f9fa;padding:12px;text-align:left;border-bottom:2px solid #dfe6e9;color:#636e72;}}
            td{{padding:12px;border-bottom:1px solid #eee;}}
            .profit{{color:#27ae60;font-weight:bold;}} .loss{{color:#e74c3c;font-weight:bold;}}
            .mine{{background-color:#e8f8f5;font-weight:500;}}
            .score-high{{color:#e67e22;font-weight:bold;}}
            .chart-container{{display:grid;grid-template-columns:repeat(auto-fit, minmax(45%, 1fr));gap:20px;}}
            .chart-img{{width:100%;border-radius:8px;border:1px solid #eee;box-shadow:0 2px 5px rgba(0,0,0,0.1);}}
        </style></head><body><div class="container">
        <h1>📊 기태님 자산 관리 대시보드 <span class="badge">{self.market_regime.upper()}</span></h1>
        <p>시스템 모드: <b>{regime_msg}</b> | 생성 시간: {full_now}</p>

        <div class="card"><h2>[0] 글로벌 시장 요약</h2>
            <table><tr><th>지수</th><th>현재가</th><th>변동률</th><th>추세</th></tr>
            {"".join(f"<tr><td>{n}</td><td>{v['curr']:.2f}</td><td class='{'profit' if v['pct']>0 else 'loss'}'>{v['pct']:+.2f}%</td><td>{v['status']}</td></tr>" for n, v in macro_data.items())}
            </table></div>

        <div class="card"><h2>✅ [MY] 보유 종목 진단</h2>
            <table><tr><th>종목</th><th>섹터</th><th>수익률</th><th>현재점수</th><th>ROE</th><th>상태</th></tr>
            {"".join(f"<tr><td><b>{r['ticker']}</b></td><td>{r['sector']}</td><td class='{'loss' if r['profit']<0 else 'profit'}'>{r['profit']:+.2f}%</td><td><span class='score-high'>{r['curr_score']}점</span></td><td>{r['roe']}</td><td>{r['status']}</td></tr>" for r in my_status)}
            </table></div>

        <div class="card"><h2>📈 [MY] 수익 관리 차트</h2>
            <div class="chart-container">
            {"".join(f'<a href="Charts/{p["ticker"]}_tracking.png" target="_blank"><img src="Charts/{p["ticker"]}_tracking.png" class="chart-img"></a>' for p in MY_POSITIONS)}
            </div>
        </div>

        <div class="card"><h2>🥇 최종 추천 TOP 3 (스마트 분산 + 매매 가이드)</h2>
            <table><tr><th>종목</th><th>섹터</th><th>점수</th><th>현재가</th><th>수량</th><th>손절가(Cut)</th><th>익절가(Target)</th><th>상관성</th></tr>
            {"".join(f"<tr><td><b>{r['ticker']}</b></td><td>{r['sector']}</td><td><span class='score-high'>{r['score']}점</span></td><td>${r['close']:.2f}</td><td><b>{r['qty']}주</b></td><td class='loss'>${r['stop']:.2f}</td><td class='profit'>${r['target']:.2f}</td><td>{r['max_corr']:.2f}</td></tr>" for r in top3)}
            </table></div>

        <div class="card"><h2>🌟 슈퍼리드 골든 리스트 (85점 이상)</h2>
            <table><tr><th>종목</th><th>섹터</th><th>점수</th><th>ROE</th><th>수량</th><th>손절가</th><th>익절가</th><th>상관성</th></tr>
            {"".join(f"<tr><td><b>{r['ticker']}</b></td><td>{r['sector']}</td><td><span class='score-high'>{r['score']}점</span></td><td>{r['roe']}</td><td>{r['qty']}주</td><td>${r['stop']:.2f}</td><td>${r['target']:.2f}</td><td>{r['max_corr']:.2f}</td></tr>" for r in gold_list)}
            </table></div>

        {"".join(f"<div class='card'><h2>[{name}] (총 {res['total']}개 중 {len(res['items'])}개 포착)</h2>" + 
            "<table><tr><th>종목</th><th>섹터</th><th>점수</th><th>현재가</th><th>수량</th><th>손절가</th><th>익절가</th><th>3M수익</th></tr>" + 
            "".join(f"<tr class='{'mine' if r.get('is_mine') else ''}'><td><b>{'🏆 ' + r['ticker'] if r.get('is_mine') else r['ticker']}</b></td><td>{r['sector']}</td><td>{r['score']}점</td><td>${r['close']:.2f}</td><td>{r['qty']}주</td><td>${r['stop']:.2f}</td><td>${r['target']:.2f}</td><td>{r['perf_3m']:.1%}</td></tr>" for r in res['items']) + 
            "</table></div>" for name, res in scan_results.items())}

        <div class="card"><h2>⚠️ 중복 위험 종목 (Excluded)</h2>
            <table><tr><th>종목</th><th>섹터</th><th>점수</th><th>상관성</th><th>비고</th></tr>
            {"".join(f"<tr><td><b>{r['ticker']}</b></td><td>{r['sector']}</td><td>{r['score']}점</td><td>{r['max_corr']:.2f}</td><td>보유 종목과 유사</td></tr>" for r in excluded)}
            </table></div>
        </div></body></html>"""
        with open("index.html", "w", encoding="utf-8") as f: f.write(html)
        print("\n>>> [완료] 리포트 생성 완료. 브라우저를 엽니다.")
        webbrowser.open('file://' + os.path.realpath("index.html"))

    def upload_to_github(self):
        print("\n>>> [Git] GitHub 업로드 시도 중...")
        try:
            subprocess.run(["git", "pull"], check=False, capture_output=True)
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Report: {datetime.now().strftime('%Y-%m-%d %H:%M')}"], check=False, capture_output=True)
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print(f"   -> [성공] GitHub 업로드 완료!")
        except Exception as e:
            print(f"\n[!!!] GitHub 업로드 실패: {e}")

    def run(self):
        print(">>> [1/6] 시장 데이터 확보...")
        macro_results = {}
        try:
            macro_data = yf.download(list(MACRO_ASSETS.keys()), period="6mo", group_by='ticker', progress=False)
            for t, n in MACRO_ASSETS.items():
                if t in macro_data.columns.levels[0]:
                    d = macro_data[t].dropna()
                    curr, prev = d['Close'].iloc[-1], d['Close'].iloc[-2]
                    status = "강세 ☀️" if curr > d['Close'].rolling(120).mean().iloc[-1] else "약세 ⛈️"
                    macro_results[n] = {'curr': curr, 'pct': (curr/prev-1)*100, 'status': status}
            if '^GSPC' in macro_data.columns.levels[0]:
                self.market_regime = self.detect_market_regime(macro_data['^GSPC'])
                print(f"   -> 시장 상태: {self.market_regime.upper()}")
        except: self.market_regime = "neutral"
        spy_perf = (macro_results.get('S&P 500', {}).get('curr', 4000) / 4000) - 1

        print(">>> [2/6] 유니버스 구성...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            sp_l = pd.read_html(io.StringIO(requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers).text))[0]['Symbol'].str.replace('.', '-').tolist()
            nq_l = pd.read_html(io.StringIO(requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=headers).text))[4]['Ticker'].tolist()
        except: 
            sp_l, nq_l = ['AAPL','MSFT','GOOG','AMZN','NVDA','JPM','PG','JNJ'], ['AAPL','MSFT','GOOGL','NVDA','PEP','COST']
        sox_l = ['AMD','ADI','ASML','AMAT','AVGO','INTC','KLAC','LRCX','MRVL','MU','NVDA','NXPI','ON','QCOM','STM','SWKS','TSM','TER','TXN']
        all_t = sorted(list(set(sp_l + nq_l + sox_l + [p['ticker'] for p in MY_POSITIONS])))

        print(f">>>> [3/6] 총 {len(all_t)}개 종목 다운로드...")
        data = yf.download(all_t, period="2y", group_by='ticker', progress=True, threads=False)

        print(">>> [4/6] 보유 종목 분석...")
        my_status, holdings_data = [], {}
        for p in tqdm(MY_POSITIONS):
            t = p['ticker']
            if t in data and not data[t].empty:
                df = data[t].dropna()
                holdings_data[t] = df['Close']
                df = self.calculate_indicators(df)
                info = self.fetch_yf_info(t)
                self.financials[t] = info
                score = self.calculate_adaptive_score(df.iloc[-1], df, spy_perf, info['roe'], info['sector'])
                df_buy = self.calculate_indicators(data[t].loc[:p['entry_date']])
                buy_score = self.calculate_adaptive_score(df_buy.iloc[-1], df_buy, spy_perf, 0, info['sector']) if df_buy is not None else 0
                my_status.append({'ticker':t, 'date':p['entry_date'], 'buy':p['price'], 'curr':df['Close'].iloc[-1], 
                                  'profit':(df['Close'].iloc[-1]/p['price']-1)*100, 
                                  'buy_score':buy_score, 'curr_score':score, 'roe':f"{info['roe']*100:.1f}%", 'sector':info['sector'], 'status':"보유(Keep)"})
                self.save_white_chart(t, df, p['price'], p['entry_date'], "보유(Keep)", score, f"{info['roe']*100:.1f}%", info['sector'])

        print(">>> [5/6] 전수 조사...")
        scans = [("2-1. 반도체(SOX)", sox_l), ("2-2. 나스닥100", nq_l), ("2-3. S&P 500", sp_l)]
        scan_results, all_candidates = {}, []
        seen_tickers = set()

        for name, t_list in scans:
            found = []
            for t in tqdm(t_list, desc=name):
                if t not in data or t in MACRO_ASSETS: continue
                df = data[t].dropna()
                if not self.validate_data(df): continue
                df = self.calculate_indicators(df)
                if df is None: continue
                
                info = self.fetch_yf_info(t)
                self.financials[t] = info
                score = self.calculate_adaptive_score(df.iloc[-1], df, spy_perf, info['roe'], info['sector'])
                
                cutoff = 35 if self.market_regime == "downtrend" else 65
                if score >= cutoff:
                    t_data = df['Close'][-60:]
                    max_corr = 0
                    if holdings_data:
                        corrs = [t_data.corr(h[-60:]) for h in holdings_data.values() if len(h) >= 60]
                        if corrs: max_corr = max(corrs)
                    
                    # [핵심] 매매 가이드 계산
                    atr = df['atr'].iloc[-1]
                    close = df['Close'].iloc[-1]
                    stop_price = close - (2 * atr)
                    target_price = close + (4 * atr) # 손익비 1:2
                    risk_per_share = (close - stop_price) * self.usd_krw
                    qty = int(self.risk_money / risk_per_share) if risk_per_share > 0 else 0
                    
                    item = {
                        'ticker':t, 'score':score, 'roe':f"{info['roe']*100:.1f}%", 'sector':info['sector'],
                        'close':close, 'perf_3m':(close/df['Close'].iloc[-63]-1), 'max_corr':max_corr, 
                        'qty': qty, 'stop': stop_price, 'target': target_price,
                        'is_mine': t in holdings_data
                    }
                    found.append(item)
                    if not item['is_mine'] and t not in seen_tickers:
                        all_candidates.append(item)
                        seen_tickers.add(t)
            found.sort(key=lambda x: x['score'], reverse=True)
            scan_results[name] = {'items': found, 'total': len(t_list)}

        with open(FIN_FILE, 'w') as f: json.dump({'update_date': datetime.now().strftime("%Y-%m-%d"), 'stocks': self.financials}, f)

        # 추천 로직 (스마트 필터)
        df_all = pd.DataFrame(all_candidates)
        if not df_all.empty:
            top3 = []
            candidates = df_all[df_all['max_corr'] < 0.6].sort_values('score', ascending=False).to_dict('records')
            picked_sectors = []
            for c in candidates:
                if len(top3) >= 3: break
                if c['sector'] not in picked_sectors:
                    top3.append(c); picked_sectors.append(c['sector'])
            
            if len(top3) < 3:
                others = df_all[(df_all['max_corr'] < 0.75) & (~df_all['ticker'].isin([x['ticker'] for x in top3]))].sort_values('score', ascending=False).to_dict('records')
                top3.extend(others[:3-len(top3)])
            
            if len(top3) < 3: # 0.85로 최종 확장
                others = df_all[(df_all['max_corr'] < 0.85) & (~df_all['ticker'].isin([x['ticker'] for x in top3]))].sort_values('score', ascending=False).to_dict('records')
                top3.extend(others[:3-len(top3)])

            gold_list = df_all[df_all['score'] >= 85].sort_values('score', ascending=False).to_dict('records')
            excluded = df_all[(df_all['max_corr'] >= 0.85)].sort_values('score', ascending=False).head(15).to_dict('records')
        else: top3, gold_list, excluded = [], [], []

        self.generate_html(macro_results, my_status, top3, gold_list, scan_results, excluded)
        self.upload_to_github()

if __name__ == "__main__":
    try:
        ExpertQuantSystem(CAPITAL_KRW).run()
    except Exception as e:
        print(f"\n[오류] {e}")
        traceback.print_exc()
    finally:
        input("\n[안내] 엔터를 누르면 종료합니다...")