import yfinance as yf
import pandas as pd
import numpy as np
import sys
import requests
import io
import time

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
        df['ma50'] = df['Close'].rolling(50).mean()
        df['ma200'] = df['Close'].rolling(200).mean()
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        p_dm, m_dm = df['High'].diff(), df['Low'].diff()
        tr_s = df['tr'].ewm(span=14, adjust=False).mean()
        df['adx'] = (100 * (p_dm.ewm(span=14).mean()/tr_s - abs(m_dm).ewm(span=14).mean()/tr_s).abs() / 
                     (p_dm.ewm(span=14).mean()/tr_s + abs(m_dm).ewm(span=14).mean()/tr_s)).ewm(span=14).mean()
        return df

    def calculate_super_lead_score(self, curr, df, spy_perf):
        """[수정 완료] 함수 이름 일치 및 로직 강화"""
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
        if curr['Close'] > curr['ma20'] * 1.08: return 0.0 # 과열 필터
        return round(final, 2)

    def print_detailed_row(self, s, prefix="  >"):
        unit = int(self.risk_money / (s['atr'] * 2 * self.usd_krw))
        sec_kr = SECTOR_MAP.get(s['sector'], s['sector'])
        rr_ratio = abs((s['close'] - s['exit_l']) / (s['close'] - s['stop'])) if abs(s['close'] - s['stop']) > 0 else 0
        
        print(f"{prefix} {s['ticker']:<5} ({sec_kr}): {s['label']} 돌파 [점수 {s['score']:.1f}]")
        print(f"      (수량 {unit:>3}주 | 가격 ${s['close']:<7.2f} | 3M수익 {s['perf_3m']:.1%})")
        print(f"      (손절 ${s['stop']:.2f} | 익절 ${s['exit_l']:.2f} | 손익비 {rr_ratio:.1f} | 상관성 {s['max_corr']:.2f})")
        print("")

    def run(self):
        sp_list, nq_list, sox_list, sp_sectors = get_indices_data()
        my_tickers = [p['ticker'] for p in MY_POSITIONS]
        all_tickers = sorted(list(set(sp_list + nq_list + sox_list + list(MACRO_ASSETS.keys()) + my_tickers)))
        
        print(f"\n>>> 실시간 환율 적용: 1달러 = {self.usd_krw:.2f}원")
        print(f">>> [전략 엔진] 총 {len(all_tickers)}개 자산 정밀 분석 시작...")
        
        data = yf.download(all_tickers, period="2y", auto_adjust=True, group_by='ticker', progress=False)
        spy_perf = (data['^GSPC']['Close'].iloc[-1] / data['^GSPC']['Close'].iloc[-63]) - 1 if not data['^GSPC'].empty else 0
        holdings_data = {t: data[t]['Close'].dropna() for t in my_tickers}

        # [0] 시장 요약
        print("\n" + "="*95 + "\n [0] 글로벌 거시 지표 및 시장 상태 요약\n" + "-"*95)
        for ticker, name in MACRO_ASSETS.items():
            if ticker in data.columns.levels[0]:
                d = data[ticker].dropna()
                if not d.empty:
                    curr, prev = d['Close'].iloc[-1], d['Close'].iloc[-2]
                    status = "강세 ☀️" if curr > d['Close'].rolling(200).mean().iloc[-1] else "약세 ⛈️"
                    print(f" ● {name:<15}: {curr:>10.2f} ({ (curr/prev-1)*100 :>+5.2f}%) | {status}")

        # [1] 보유 종목
        print("\n" + "="*95 + "\n [1] 현재 보유 종목 정밀 진단\n" + "-"*95)
        for pos in MY_POSITIONS:
            t = pos['ticker']
            df = self.calculate_indicators(data[t].dropna())
            if df is not None:
                curr = df.iloc[-1]
                entry_atr = df.loc[df.index <= pos['entry_date']]['atr'].iloc[-1] if not df.loc[df.index <= pos['entry_date']].empty else df['atr'].iloc[-1]
                stop_p, exit_l = pos['price'] - (2 * entry_atr), df['Low'].rolling(10).min().iloc[-1]
                print(f" ● {t:<5} | 수익 {(curr['Close']/pos['price']-1)*100:>5.1f}% | 현재가 ${curr['Close']:.2f} | 손절가 ${stop_p:.2f} | 익절가 ${exit_l:.2f}")

        # [2] 지수별 분석
        all_signals = []
        indices_to_scan = [("2-1. 반도체(SOX)", sox_list), ("2-2. 나스닥100", nq_list), ("2-3. S&P 500", sp_list)]
        
        for idx_name, t_list in indices_to_scan:
            print("\n" + "="*95 + f"\n [{idx_name}] 전수 조사 결과 (총 {len(t_list)}개 분석)")
            print("-" * 95)
            curr_found = 0
            for i, t in enumerate(t_list, 1):
                sys.stdout.write(f"\r  ▶ 분석 진행률: {i}/{len(t_list)} ({t:<5})")
                sys.stdout.flush()
                
                if t in my_tickers or t not in data.columns.levels[0]: continue
                df = self.calculate_indicators(data[t].dropna())
                if df is None: continue
                
                score = self.calculate_super_lead_score(df.iloc[-1], df, spy_perf)
                
                if score >= 75.0:
                    correlations = [df['Close'].corr(h_close) for h_close in holdings_data.values()]
                    max_corr = max(correlations) if correlations else 0.0
                    s = {
                        'ticker': t, 'label': 'S2' if df.iloc[-1]['Close'] > df['High'].rolling(55).max().iloc[-2] else 'S1',
                        'close': df.iloc[-1]['Close'], 'atr': df.iloc[-1]['atr'], 'adx': df.iloc[-1]['adx'], 
                        'exit_l': df['Low'].rolling(10).min().iloc[-1], 'score': score, 
                        'perf_3m': (df.iloc[-1]['Close']/df['Close'].iloc[-63]-1) if len(df) > 63 else 0, 
                        'sector': sp_sectors.get(t, "Technology" if t in sox_list else "기타"), 
                        'max_corr': max_corr, 'stop': df.iloc[-1]['Close']-(2*df.iloc[-1]['atr']),
                        'close_series': df['Close']
                    }
                    all_signals.append(s)
                    print(f"\n") 
                    self.print_detailed_row(s)
                    curr_found += 1
            print(f"\n  >>> {idx_name}: 총 {curr_found}개 종목 최종 포착됨.")

        # [3] 골든 리스트 (점수 130 이상)
        unique_signals = {v['ticker']:v for v in all_signals}.values()
        perfect = [s for s in unique_signals if s['score'] >= 130]
        print("\n" + "="*95 + f"\n [3] 초엄격 '슈퍼리드' 골든 리스트 (총 {len(perfect)}개)")
        print("-" * 95)
        for s in sorted(perfect, key=lambda x: x['score'], reverse=True):
            self.print_detailed_row(s, prefix="  ★")

        # [4] 최종 추천 TOP 3
        print("\n" + "="*95 + "\n [4] 최종 추천 TOP 3 (상관관계 0.5 미만 & 분산 최적화)\n" + "-"*95)
        df_all = pd.DataFrame(unique_signals)
        if not df_all.empty:
            passed = df_all[df_all['max_corr'] < 0.5].sort_values(by='score', ascending=False)
            excluded = df_all[df_all['max_corr'] >= 0.5].sort_values(by='score', ascending=False)

            if not passed.empty:
                top_3 = passed.groupby('sector').head(1).sort_values(by='score', ascending=False).head(3)
                for i, r in enumerate(top_3.to_dict('records'), 1):
                    self.print_detailed_row(r, prefix=f"  🥇 {i}위")
            else: print("  적합한 분산 종목이 없습니다.")

            print("\n" + "="*95 + "\n [5] ★중복 위험★ 실력은 좋으나 포트폴리오와 동조화가 높은 종목\n" + "-"*95)
            for r in excluded.head(5).to_dict('records'):
                self.print_detailed_row(r, prefix="  ⚠️ [중복]")

        # [6] 전문가 리스크 관리 제언
        total_risk = len(MY_POSITIONS) * 1.0 
        print("\n" + "="*95 + "\n [6] 전문가 리스크 관리 제언\n" + "-"*95)
        print(f" ● 현재 포트폴리오 리스크 총량(Portfolio Heat): {total_risk:.1f}% (안전 범위)")
        print(f" ● 실시간 환율({self.usd_krw:.1f}원) 기반으로 계산된 추천 수량입니다. 환율 변동에 주의하세요.")
        print(f" ● 모든 추천 종목은 손절가를 반드시 시스템에 미리 입력해두어 '비자발적 장기투자'를 방지하십시오.")
        
        input("\n[알림] 기태님, 모든 분석이 완료되었습니다. 엔터를 누르면 종료됩니다.")

if __name__ == "__main__":
    UltimateGiTaeSystem(CAPITAL_KRW).run()