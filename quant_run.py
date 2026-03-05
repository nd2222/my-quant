import sys
import os
import io
import json
import time
import random
import warnings
import webbrowser
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import shutil
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from utils.momentum import calculate_momentum_13612

# ================= [시스템 설정] =================
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)
plt.rcParams['axes.unicode_minus'] = False 

# ================= [수정 #5 - 매직 넘버 Config 통합] =================
DEFAULT_CONFIG = {
    # 계좌 설정
    "account": {
        "total_krw": 76826207,          # 총 자산 (원) - 사용자 실제 계좌 기준
        "risk_ratio": 0.01,             # 개별 종목 리스크 (1%)
        "max_alloc_per_stock": 0.20,    # 종목별 최대 비중 (20%)
        "expert_cutoff": 70,            # Expert 전략 점수 커트오프
        "min_invest_per_stock": 10000000,  # 종목별 최소 투자금 (1,000 만원)
        "half_kelly_ratio": 0.08,       # 하프 켈리 비율 (승률 50% 가정시 8%)
        "min_cash_buffer": 0.05         # P0: 현금 최소 버퍼 (5%) - 전액투자 방지
    },
    # 데이터 수집 설정
    "data": {
        "chunk_size": 50,               # yfinance 일괄 다운로드 청크 크기
        "delay_between_chunks": 2.0,    # 청크 간 딜레이 (초, Rate Limit 대응)
        "download_timeout": 30,         # 다운로드 타임아웃 (초)
        "download_period": "1y",        # 기본 다운로드 기간
        "macro_period": "6mo",          # 매크로 데이터 기간
        "rs_period_days": 63,           # RS 계산 기간 (3 개월)
        "volume_avg_days": 21           # 거래량 평균 기간
    },
    # 환율 설정
    "exchange": {
        "default_rate": 1450.0,         # 기본 환율 (최후의 수단)
        "cache_max_age_hours": 24       # 환율 캐시 유효기간 (시간)
    },
    # 캐시/로그 설정
    "cache": {
        "log_max_bytes": 5242880,       # 로그 파일 최대 크기 (5MB)
        "log_backup_count": 3           # 로그 백업 개수
    },
    # 포지션
    "positions": [
        {'ticker': 'CAT', 'price': 729.55, 'qty': 10, 'entry_date': '2026-02-09'},
        {'ticker': 'IEX', 'price': 186.77, 'qty': 35, 'entry_date': '2026-01-13'},
        {'ticker': 'TSLA', 'price': 403.84, 'qty': 16, 'entry_date': '2024-12-31'},
        {'ticker': 'VTRS', 'price': 14.67, 'qty': 500, 'entry_date': '2026-02-09'},
        {'ticker': 'WDC', 'price': 225.01, 'qty': 11, 'entry_date': '2026-01-16'},
        {'ticker': 'XOM', 'price': 150.00, 'qty': 50, 'entry_date': '2026-02-09'},
        {'ticker': 'EOG', 'price': 120.76, 'qty': 108, 'entry_date': '2026-02-13'}
    ]
}

# ================= [상수 선언 - __main__ 에서 초기화] =================
# 아래 변수들은 __main__ 블록에서 config.json 로드 후 초기화됩니다
TOTAL_ACCOUNT_VALUE_KRW = DEFAULT_CONFIG['account']['total_krw']
RISK_RATIO = DEFAULT_CONFIG['account']['risk_ratio']
MAX_ALLOC_PER_STOCK = DEFAULT_CONFIG['account']['max_alloc_per_stock']
# ================= [경로 설정] =================
# P0-2: config.json dirs 섹션 적용 + 상대경로 기본
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # quant_run.py 가 있는 폴더
DATA_DIR = os.path.join(BASE_DIR, "Data")
REPORT_DIR = os.path.join(BASE_DIR, "Reports")
CHART_DIR = os.path.join(BASE_DIR, "Charts")
CACHE_DIR = os.path.join(DATA_DIR, "Cache")
GITHUB_DIR = os.path.join(BASE_DIR, "my-quant")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
log_file = os.path.join(BASE_DIR, "system.log")

for d in [DATA_DIR, REPORT_DIR, CHART_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= [로그 설정] =================
logger = logging.getLogger()
if logger.hasHandlers(): logger.handlers.clear()

# 1. 파일에 기록하는 핸들러 (system.log)
file_handler = RotatingFileHandler(log_file, maxBytes=5242880, backupCount=3, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 2. 터미널 화면에 실시간으로 보여주는 핸들러 (추가된 부분)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

logger.setLevel(logging.INFO)

# ================= [기본 상수 정의] =================
# P2-6: config 로드 전이므로 DEFAULT_CONFIG 에서 직접 읽음
EXPERT_CUTOFF = DEFAULT_CONFIG['account']['expert_cutoff']
MIN_INVEST_PER_STOCK = DEFAULT_CONFIG['account']['min_invest_per_stock']
CHUNK_SIZE = DEFAULT_CONFIG['data']['chunk_size']
DELAY_BETWEEN_CHUNKS = DEFAULT_CONFIG['data']['delay_between_chunks']
DOWNLOAD_TIMEOUT = DEFAULT_CONFIG['data']['download_timeout']
DOWNLOAD_PERIOD = DEFAULT_CONFIG['data']['download_period']
MACRO_PERIOD = DEFAULT_CONFIG['data']['macro_period']
RS_PERIOD_DAYS = DEFAULT_CONFIG['data']['rs_period_days']
VOLUME_AVG_DAYS = DEFAULT_CONFIG['data']['volume_avg_days']
DEFAULT_EXCHANGE_RATE = DEFAULT_CONFIG['exchange']['default_rate']
EXCHANGE_CACHE_MAX_AGE_HOURS = DEFAULT_CONFIG['exchange']['cache_max_age_hours']
LOG_MAX_BYTES = DEFAULT_CONFIG['cache']['log_max_bytes']
LOG_BACKUP_COUNT = DEFAULT_CONFIG['cache']['log_backup_count']
# P6: holdings.csv 에서 보유 종목 자동 로드
def load_positions_from_csv():
    """holdings.csv 에서 보유 종목 로드"""
    try:
        from holdings_loader import load_holdings_csv
        csv_path = os.path.join(BASE_DIR, 'holdings.csv')
        holdings = load_holdings_csv(csv_path)
        if holdings:
            logging.info(f"✅ holdings.csv 에서 {len(holdings)}개 종목 로드")
            # P1-8: config.json 에 runtime 값 저장 안함 (holdings.csv 만 사용)
            # DEFAULT_CONFIG['positions'] = holdings  # 주석처리
            return holdings
    except Exception as e:
        logging.warning(f"holdings.csv 로드 실패: {e}")
    
    # CSV 없거나 실패시 config.json 사용
    return DEFAULT_CONFIG.get('positions', [])

MY_POSITIONS = load_positions_from_csv()
MY_TICKERS = [p['ticker'] for p in MY_POSITIONS]
TICKERS_TO_CHART = MY_TICKERS

logging.info(f"📊 최종 보유종목: {len(MY_POSITIONS)}개 ({', '.join(MY_TICKERS)})")

# ================= [폰트 설정] =================
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except: pass

# ================= [시장 자산 정의] =================
MACRO_ASSETS = {
    '^GSPC': 'S&P 500', '^IXIC': '나스닥 종합', '^DJI': '다우 존스', '^RUT': '러셀 2000',
    '^SOX': '필라델피아 반도체', '^VIX': '공포지수 (VIX)', 'GLD': '금 (Gold)', 'SI=F': '은 (Silver)',
    'CL=F': 'WTI 원유', 'NG=F': '천연가스', 'HG=F': '구리 (Copper)', '^TNX': '미국채 10 년물', 
    'DX-Y.NYB': '달러 인덱스', 'BTC-USD': '비트코인'
}

# SOX_TICKERS - 완전 동적 수집 (하드코딩 제거)
SOX_TICKERS = []  # fetch_dynamic_tickers() 에서 자동 채움
SOX_FALLBACK = []  # ❌ 하드코딩 제거 - 수집 실패 시 빈 리스트
BLACKLIST = []
CUSTOM_ORDER = ['반도체 (SOX)', '나스닥 100', 'S&P500']

# [NEW] 섹터 한글 매핑
SECTOR_KOR = {
    'Technology': '기술 (IT)', 'Semiconductors': '반도체', 'Financial Services': '금융',
    'Energy': '에너지', 'Healthcare': '헬스케어', 'Consumer Cyclical': '경기소비재',
    'Industrials': '산업재', 'Consumer Defensive': '필수소비재', 'Utilities': '유틸리티',
    'Real Estate': '부동산', 'Basic Materials': '원자재', 'Communication Services': '통신',
    'Default': '기타'
}

# P0-2: Sector 가중치 수정 (debt 를 보너스 항목으로 전환, 양수 합계 100 점)
SECTOR_SCORING = {
    'Technology': { 'weights': {'growth': 30, 'roe': 15, 'rs': 25, 'momentum': 20, 'debt': 10}, 'thresh': {'growth': 20, 'roe': 15, 'per_max': 50, 'pbr_max': 15, 'debt_max': 100} },  # 양수 90 + debt 보너스 10 = 100
    'Semiconductors': { 'weights': {'growth': 35, 'roe': 15, 'rs': 25, 'momentum': 15, 'debt': 10}, 'thresh': {'growth': 25, 'roe': 20, 'per_max': 60, 'pbr_max': 20, 'debt_max': 80} },  # 양수 90 + debt 보너스 10 = 100
    'Financial Services': { 'weights': {'roe': 35, 'pbr': 20, 'div': 20, 'debt': 15, 'rs': 10}, 'thresh': {'roe': 12, 'pbr_max': 1.5, 'div_min': 3.0, 'debt_max': 300} },  # 양수 85 + debt 보너스 15 = 100
    'Energy': { 'weights': {'div': 30, 'roe': 20, 'debt': 20, 'pbr': 10, 'rs': 20}, 'thresh': {'roe': 10, 'div_min': 4.0, 'per_max': 12, 'pbr_max': 2.5, 'debt_max': 150} },  # 양수 80 + debt 보너스 20 = 100
    'Healthcare': { 'weights': {'roe': 28, 'growth': 23, 'div': 17, 'rs': 17, 'debt': 15}, 'thresh': {'roe': 15, 'growth': 10, 'per_max': 30, 'debt_max': 120} },  # 양수 85 + debt 보너스 15 = 100
    'Consumer Cyclical': { 'weights': {'roe': 28, 'growth': 28, 'rs': 22, 'momentum': 12, 'debt': 10}, 'thresh': {'roe': 18, 'growth': 12, 'per_max': 40, 'pbr_max': 10, 'debt_max': 120} },  # 양수 90 + debt 보너스 10 = 100
    'Default': { 'weights': {'roe': 28, 'growth': 17, 'pbr': 17, 'rs': 23, 'debt': 15}, 'thresh': {'roe': 15, 'growth': 10, 'per_max': 25, 'pbr_max': 4, 'debt_max': 150} }  # 양수 85 + debt 보너스 15 = 100
}

# 3 순위: 섹터 정규화 딕셔너리 (Yahoo Finance → SECTOR_SCORING 키 매핑)
SECTOR_NORMALIZE = {
    'Financials': 'Financial Services',
    'Consumer Discretionary': 'Consumer Cyclical',
    'Consumer Defensive': 'Consumer Staples',
    'Communication Services': 'Communication Services',
    'Technology': 'Technology',
    'Semiconductors': 'Semiconductors',
    'Energy': 'Energy',
    'Healthcare': 'Healthcare',
    'Industrials': 'Industrials',
    'Basic Materials': 'Basic Materials',
    'Utilities': 'Utilities',
    'Real Estate': 'Real Estate',
    'Consumer Staples': 'Consumer Defensive',
}

SECTOR_LIMITS = {'Semiconductors': 0.30, 'Energy': 0.20, 'Technology': 0.30, 'Financial Services': 0.20, 'Healthcare': 0.20}

# V25.0: 섹터 ETF 매핑 (섹터 로테이션 감지용)
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Energy': 'XLE',
    'Healthcare': 'XLV',
    'Financial Services': 'XLF',
    'Industrials': 'XLI',
    'Consumer Cyclical': 'XLY',
    'Semiconductors': 'SOXX',
    'Consumer Defensive': 'XLP',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Basic Materials': 'XLB',
    'Communication Services': 'XLC'
}

class EarlyWarningSystem:
    """V25.0: 보유 종목 선제 교체 신호 감지 (4 가지 신호 → 복합 긴급도 점수)"""
    
    def __init__(self):
        self.checkpoints = [-1, -6, -11, -16]  # 거래일 기준 (현재, 1 주전, 2 주전, 3 주전)
    
    def _check_momentum_decay(self, ticker, df):
        """신호 1: 모멘텀 둔화 감지 (3 주 연속 13612W 모멘텀 하락)"""
        momentums = []
        for cp in self.checkpoints:
            if len(df) < abs(cp):
                momentums.append(0)
                continue
            prices = df['Close'][:cp]
            if len(prices) < 21:
                momentums.append(0)
            else:
                ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100
                ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100
                ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100
                ret_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) * 100 if len(prices) >= 252 else 0
                m = (12 * ret_1m + 4 * ret_3m + 2 * ret_6m + ret_12m) / 19
                momentums.append(m)
        
        # 3 주 연속 하락 여부
        decay_weeks = sum(1 for i in range(3) if momentums[i] < momentums[i+1])
        score = decay_weeks * 25  # 최대 75 점
        trend = [f"{m:+.1f}%" for m in momentums]
        
        return score, trend
    
    def _check_rs_reversal(self, ticker, my_rs_score, all_rs_map):
        """신호 2: RS 역전 감지 (내 종목보다 RS 높은 신규 후보 N 개 이상)"""
        stronger = [
            t for t, rs in all_rs_map.items()
            if rs > my_rs_score + 10 and t not in MY_TICKERS
        ]
        stronger.sort(key=lambda x: all_rs_map[x], reverse=True)
        
        # 공격적: 3 개 이상 출현시 교체 신호
        score = min(100, len(stronger) * 5)
        
        return score, stronger[:5]  # 상위 5 개 후보 반환
    
    def _check_sector_rotation(self, my_sector, sector_momentum_map):
        """신호 3: 섹터 로테이션 감지 (섹터별 모멘텀 순위)"""
        if not sector_momentum_map or my_sector not in sector_momentum_map:
            return 0, 0, 0
        
        all_sectors = sorted(sector_momentum_map, key=lambda x: sector_momentum_map[x], reverse=True)
        my_rank = all_sectors.index(my_sector) + 1
        rank_pct = my_rank / len(all_sectors)
        
        # 하위 30% 면 섹터 자금 이탈 신호
        score = int(rank_pct * 100)
        
        return score, my_rank, len(all_sectors)
    
    def calculate_urgency(self, ticker, holding, all_rs_map, sector_map, df):
        """신호 4: 복합 긴급도 스코어"""
        s1, momentum_trend = self._check_momentum_decay(ticker, df)
        s2, stronger_list = self._check_rs_reversal(ticker, holding.get('rs_score', 50), all_rs_map)
        s3, rank, total = self._check_sector_rotation(holding.get('sector', 'Default'), sector_map)
        
        # 가중 합산
        urgency = s1 * 0.35 + s2 * 0.40 + s3 * 0.25
        
        # 공격적 모드: 단일 신호 강하면 부스트
        if s1 >= 50 or s2 >= 60 or s3 >= 70:
            urgency = min(100, urgency * 1.3)
        
        return {
            'urgency': round(urgency),
            'momentum_decay': s1,
            'rs_reversal': s2,
            'sector_rotation': s3,
            'momentum_trend': momentum_trend,
            'replace_candidates': stronger_list
        }

class EnhancedDataDownloader:
    def __init__(self): 
        self.chunk_size = CHUNK_SIZE
        self.delay_between_chunks = DELAY_BETWEEN_CHUNKS
    def download_all(self, tickers, period=DOWNLOAD_PERIOD, use_lazy=False):
        """P2-4: 중복 컬럼 검증 추가"""
        logging.info(f"총 {len(tickers)}개 종목 데이터 수집 중 (최근 1 년)...")
        # P2-4: 중복 제거 (fetch_dynamic_tickers 에서 이미 set() 처리하지만 안전장치)
        tickers = list(dict.fromkeys(tickers))  # 순서 유지하며 중복 제거
        chunks = [tickers[i:i + self.chunk_size] for i in range(0, len(tickers), self.chunk_size)]
        
        if use_lazy:
            return self._download_lazy(chunks, period)
        
        results = []
        for i, chunk in enumerate(chunks):
            logging.info(f"청크 {i+1}/{len(chunks)} 다운로드 중 ({len(chunk)}개)...")
            try:
                data = yf.download(chunk, period=period, group_by='ticker', progress=False, threads=False, timeout=DOWNLOAD_TIMEOUT)
                if not data.empty: results.append(data)
                time.sleep(self.delay_between_chunks)
            except Exception as e: logging.error(f"Download Error: {e}")
        if results:
            try:
                concatenated = pd.concat(results, axis=1)
                # P2-4: 중복 컬럼 검증 및 제거
                if isinstance(concatenated.columns, pd.MultiIndex):
                    duplicates = concatenated.columns.duplicated()
                    if duplicates.any():
                        logging.warning(f"중복 티커 컬럼 감지 ({duplicates.sum()}개), 제거합니다.")
                        concatenated = concatenated.loc[:, ~duplicates]
                return concatenated
            except Exception as e:
                logging.error(f"Concat error: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _download_lazy(self, chunks, period):
        """Lazy loading - 메모리 효율적 다운로드"""
        import gc
        final_df = None
        for i, chunk in enumerate(chunks):
            logging.info(f"청크 {i+1}/{len(chunks)} 다운로드 중 ({len(chunk)}개)... [Lazy Mode]")
            try:
                data = yf.download(chunk, period=period, group_by='ticker', progress=False, threads=False, timeout=DOWNLOAD_TIMEOUT)
                if not data.empty:
                    if final_df is None:
                        final_df = data.copy()
                    else:
                        final_df = pd.concat([final_df, data], axis=1, copy=False)
                    del data
                    gc.collect()
                time.sleep(self.delay_between_chunks)
            except Exception as e: 
                logging.error(f"Download Error (chunk {i+1}): {e}")
        return final_df if final_df is not None else pd.DataFrame()

class SmartFundamentalsFetcher:
    def __init__(self, data_dir):
        self.financials_file = os.path.join(data_dir, "financials.json")
        self.failed_file = os.path.join(data_dir, "failed_tickers.json")
        self.financials = self.load_json(self.financials_file, {"stocks": {}})
        self.failed = self.load_json(self.failed_file, {"failed": {}})
    def load_json(self, filepath, default):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except: return default
    def save_json(self, filepath, data):
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
        except: pass
    def fetch_with_retry(self, ticker, max_retries=3, priority=False):
        if ticker in self.financials['stocks']:
            data = self.financials['stocks'][ticker]
            last_update = datetime.strptime(data.get('updated', '2000-01-01'), "%Y-%m-%d")
            limit_days = 2 if priority else 14
            if (datetime.now() - last_update).days < limit_days: return data
        for attempt in range(max_retries):
            try:
                wait = (attempt + 1) * 1.0 if priority else 0.3
                time.sleep(wait + random.uniform(0, 0.5))
                info = yf.Ticker(ticker).info
                if not info or len(info) < 5: raise ValueError("Empty Data")
                sector = info.get('sector', '기타')
                if ticker in SOX_TICKERS: sector = 'Semiconductors'
                data = {
                    'roe': info.get('returnOnEquity', 0), 'pbr': info.get('priceToBook', 0),
                    'per': info.get('trailingPE', 0), 'psr': info.get('priceToSalesTrailing12Months', 0),
                    'debt_ratio': info.get('debtToEquity', 0), 'growth': info.get('revenueGrowth', 0),
                    'div': info.get('dividendYield', 0), 'sector': sector,
                    'updated': datetime.now().strftime("%Y-%m-%d"), 'quality': 'good'
                }
                def clean(val): return round(val * 100, 2) if val else 0
                def clean_raw(val): return round(val, 2) if val else 0
                data['roe'] = clean(data['roe'])
                data['growth'] = clean(data['growth'])
                data['div'] = clean(data['div'])
                data['pbr'] = clean_raw(data['pbr'])
                data['per'] = clean_raw(data['per'])
                data['psr'] = clean_raw(data['psr'])
                data['debt_ratio'] = clean_raw(data['debt_ratio'])
                self.financials['stocks'][ticker] = data
                self.save_json(self.financials_file, self.financials)
                logging.info(f"✅ {ticker}: 수신 완료")
                return data
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.warning(f"⚠️ {ticker}: 수신 실패 - {str(e)[:50]}")
                    self.failed['failed'][ticker] = {'last_attempt': datetime.now().strftime("%Y-%m-%d"), 'error': str(e)[:50]}
                    self.save_json(self.failed_file, self.failed)
        return None
    def get_data(self, ticker): return self.financials['stocks'].get(ticker, None)

class MacroAnalyzer:
    """P0-5: as_of_date 지원 (백테스트용)"""
    def __init__(self, data_dir=None, as_of_date=None):
        self.indicators = {}
        self.market_score = 50 
        self.regime = "Unknown"
        self.cash_ratio = 0.0
        self.warnings = []
        self.canary_signals = {}  # V24.9: 카나리아 신호등
        self.data_dir = data_dir if data_dir else DATA_DIR
        self.as_of_date = as_of_date  # P0-5: 백테스트 기준일
    def run_deep_dive(self):
        logging.info("[2/7] 매크로 딥 다이브 분석 중...")
        try:
            # P0-5: as_of_date 지원 (백테스트용)
            from datetime import timedelta
            
            if self.as_of_date:
                end = self.as_of_date
                start_short = end - timedelta(days=10)
                start_long = end - timedelta(days=400)
            else:
                end = None
                start_short = None
                start_long = None
            
            # 1. 주요 지표 수집
            if self.as_of_date:
                vix = yf.Ticker("^VIX").history(start=start_short, end=end)['Close'].iloc[-1]
                tnx = yf.Ticker("^TNX").history(start=start_short, end=end)['Close'].iloc[-1]
                oil = yf.Ticker("CL=F").history(start=start_short, end=end)['Close'].iloc[-1]
                dxy = yf.Ticker("DX-Y.NYB").history(start=start_short, end=end)['Close'].iloc[-1]
                spy = yf.Ticker("^GSPC").history(start=start_long, end=end)['Close']
            else:
                vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
                tnx = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
                oil = yf.Ticker("CL=F").history(period="5d")['Close'].iloc[-1]
                dxy = yf.Ticker("DX-Y.NYB").history(period="5d")['Close'].iloc[-1]
                spy = yf.Ticker("^GSPC").history(period="1y")['Close']
            
            self.indicators['VIX'] = round(vix, 2)
            self.indicators['10Y_Yield'] = round(tnx, 2)
            self.indicators['Oil'] = round(oil, 2)
            self.indicators['DXY'] = round(dxy, 2)
            spy_curr = spy.iloc[-1]
            spy_ma200 = spy.rolling(200).mean().iloc[-1]
            self.indicators['SPY_Trend'] = "Bull" if spy_curr > spy_ma200 else "Bear"
            
            # 2. 시장 스코어 계산 (P14: 2024-2025 기준 VIX 동적 조정)
            # P14: 2024-2025 년 VIX 장기평균 15-17 반영 (구버전 17-20 → 신버전 20-25)
            score = 100
            if vix > 35: score -= 50  # 극단적 공포
            elif vix > 25: score -= 20  # 공포 (2024-2025 기준 25 초과)
            elif vix > 20: score -= 10  # 주의 (2024-2025 기준 20 초과)
            if tnx > 4.5: score -= 20; self.warnings.append("국채금리 급등")
            elif tnx > 4.3: score -= 10; self.warnings.append("국채금리 상승세")
            if oil > 90: score -= 15; self.warnings.append("유가 폭등")
            elif oil > 85: score -= 5; self.warnings.append("유가 상승세")
            if dxy > 105: score -= 10; self.warnings.append("강달러")
            if self.indicators['SPY_Trend'] == "Bear": score -= 30; self.warnings.append("S&P500 하락 추세")
            
            # P14: 동시 발생 패턴 가중치 (VIX 상승 + 금리 상승 = 추가 페널티)
            if vix > 25 and tnx > 4.3:
                score -= 10
                self.warnings.append("VIX+ 금리 동시 상승 (위험 신호)")
            
            self.market_score = max(0, score)
            
            # 3. 카나리아 신호등 분석 (V24.9 추가)
            self._analyze_canary()
            
            # 4. 카나리아 우선으로 레짐 판단
            self._determine_regime_with_canary()
            
            logging.info(f"마켓 스코어: {self.market_score}점 ({self.regime})")
            if self.warnings:
                for w in self.warnings: logging.warning(f"⚠️ {w}")
            
            # V24.9 (10 번): 카나리아 히스토리 저장
            self._save_canary_history()
        except Exception as e:
            logging.error(f"매크로 분석 실패: {e}")
            self.regime = "Unknown"
    
    def _analyze_canary(self):
        """
        V24.9: 카나리아 신호등 - 3 레이어 구조
        
        레이어 1: 최상단 — 즉각적인 결론 (공격/주의/방어)
        레이어 2: 중간 — 4 종목 상세 (1M/3M/12M, MA200, 13612W)
        레이어 3: 하단 — 히스토리 + 맥락
        """
        canary_tickers = {'SPY': '미국주식', 'VWO': '신흥국', 'VEA': '선진국', 'BND': '채권'}
        negative_count = 0
        
        for ticker, name in canary_tickers.items():
            try:
                # P0-5: as_of_date 지원
                if self.as_of_date:
                    df = yf.Ticker(ticker).history(start=self.as_of_date - timedelta(days=400), end=self.as_of_date)
                else:
                    df = yf.Ticker(ticker).history(period="1y")
                
                # 레이어 2: 상세 데이터 계산
                ret_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0
                ret_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100 if len(df) >= 63 else 0
                ret_12m = (df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else 0
                
                # MA200 대비 위치
                ma200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else df['Close'].iloc[-1]
                curr_price = df['Close'].iloc[-1]
                ma200_status = '↑' if curr_price > ma200 * 1.02 else ('↓' if curr_price < ma200 * 0.98 else '→')
                
                # 13612W 모멘텀 (Keller 공식) - P0-4: 통합 함수 사용
                momentum_13612 = calculate_momentum_13612(df['Close'])
                
                # 레이어 1: 신호 판단 (13612W 모멘텀 기준 - Keller 원본)
                is_negative = momentum_13612 < 0
                if is_negative: negative_count += 1
                
                self.canary_signals[ticker] = {
                    'name': name,
                    'ret_1m': round(ret_1m, 1),
                    'ret_3m': round(ret_3m, 1),
                    'ret_12m': round(ret_12m, 1),
                    'ma200_status': ma200_status,
                    'ma200': round(ma200, 2),
                    'momentum_13612': round(momentum_13612, 1),
                    'negative': is_negative,
                    'mode': '방어' if is_negative else '공격'
                }
                
            except Exception as e:
                logging.warning(f"카나리아 {ticker} 분석 실패: {e}")
                self.canary_signals[ticker] = {
                    'name': name, 'ret_1m': 0, 'ret_3m': 0, 'ret_12m': 0,
                    'ma200_status': '→', 'ma200': 0, 'momentum_13612': 0,
                    'negative': False, 'mode': '공격'
                }
        
        self.canary_negative_count = negative_count
        
        # 레이어 1: 모드 결정
        if negative_count == 0:
            self.canary_mode = '공격'
            self.canary_mode_color = '#2ecc71'  # 초록
            self.canary_mode_icon = '🟢'
        elif negative_count == 1:
            self.canary_mode = '주의'
            self.canary_mode_color = '#f1c40f'  # 노랑
            self.canary_mode_icon = '🟡'
        else:
            self.canary_mode = '방어'
            self.canary_mode_color = '#e74c3c'  # 빨강
            self.canary_mode_icon = '🔴'
        
        logging.info(f"🕊️ 카나리아: {self.canary_mode_icon} {self.canary_mode} 모드 (음수 {negative_count}/4 개)")
        
        # 레이어 3: 히스토리 기록은 run_deep_dive 에서 한번만 호출 (이중 호출 방지)
    
    def _save_canary_history(self):
        """레이어 3: 카나리아 히스토리 저장 (JSON)"""
        history_file = os.path.join(self.data_dir, "canary_history.json")  # 2 순위: 인스턴스 변수 사용
        
        # 기존 히스토리 로드
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = {'records': [], 'current_mode': '공격', 'consecutive_days': 0}
        
        # 현재 상태
        today = datetime.now().strftime('%Y-%m-%d')
        current_mode = self.canary_mode
        
        # 모드 변경 감지
        mode_changed = history.get('current_mode') != current_mode
        
        if mode_changed:
            # 새 기록 추가
            prev_mode = history.get('current_mode', '공격')
            history['records'].append({
                'date': today,
                'from': prev_mode,
                'to': current_mode,
                'negative_count': self.canary_negative_count,
                'reason': self._get_mode_change_reason(history.get('current_mode'))
            })
            history['consecutive_days'] = 0
            history['current_mode'] = current_mode
            logging.info(f"📅 카나리아 신호 변경: {prev_mode} → {current_mode}")
        else:
            history['consecutive_days'] += 1
        
        # 최근 50 개 기록만 유지
        history['records'] = history['records'][-50:]
        
        # 저장
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"카나리아 히스토리 저장 실패: {e}")
    
    def _get_mode_change_reason(self, prev_mode):
        """모드 변경 사유 반환"""
        neg = self.canary_negative_count
        signals = self.canary_signals
        
        if prev_mode == '공격' and neg >= 1:
            fallen = [t for t, d in signals.items() if d['negative']]
            return f"{', '.join(fallen)} 이탈"
        elif prev_mode == '주의' and neg >= 2:
            fallen = [t for t, d in signals.items() if d['negative']]
            return f"{', '.join(fallen)} 추가 이탈"
        elif prev_mode == '방어' and neg <= 1:
            recovered = [t for t, d in signals.items() if not d['negative'] and d['ret_3m'] > 0]
            return f"{', '.join(recovered)} 회복"
        return ''
    
    def _determine_regime_with_canary(self):
        """V24.9: 카나리아 우선으로 레짐 판단 (공격 모드 = 현금 0%)"""
        neg = getattr(self, 'canary_negative_count', 0)
        min_cash_buffer = DEFAULT_CONFIG['account'].get('min_cash_buffer', 0.05)  # 5% (방어용)
        
        if neg >= 3:
            self.regime = "태풍 (Panic/Crash)"
            self.cash_ratio = max(0.8, min_cash_buffer)
        elif neg == 2:
            self.regime = "흐림 (Caution)"
            self.cash_ratio = max(0.5, min_cash_buffer)
        elif neg == 1:
            self.regime = "맑음 (Bull)"
            self.cash_ratio = max(0.2, min_cash_buffer)
        else:
            # 카나리아 모두 양수면 시장 스코어 사용
            if self.market_score >= 80:
                self.regime = "쾌청 (Strong Bull)"
                self.cash_ratio = 0.0  # 공격 모드: 현금 0% (풀투자)
            elif self.market_score >= 60:
                self.regime = "맑음 (Bull)"
                self.cash_ratio = 0.0  # 공격 모드: 현금 0% (풀투자)
            elif self.market_score >= 40:
                self.regime = "흐림 (Caution)"
                self.cash_ratio = max(0.3, min_cash_buffer)  # 30% 현금
            elif self.market_score >= 20:
                self.regime = "비 (Correction)"
                self.cash_ratio = 0.5  # 50% 현금
            else:
                self.regime = "태풍 (Panic/Crash)"
                self.cash_ratio = 0.8  # 80% 현금

class RebalanceManager:
    def __init__(self, holdings_data, macro_analyzer, cash_balance, usd_krw, fetcher):
        self.holdings = holdings_data
        self.ma = macro_analyzer
        self.cash = cash_balance
        self.usd_krw = usd_krw
        self.fetcher = fetcher
        self.sell_candidates = []
        self.buy_candidates = []
        self.scenarios = {}
        self.min_invest_per_stock = MIN_INVEST_PER_STOCK
        # 5 순위: 매도 점수 트렌드 추적 (지속성 추가)
        self.score_history_file = os.path.join(macro_analyzer.data_dir, "sell_score_history.json")
        self.sell_score_history = self._load_score_history()
        # V25.0: 조기경보 시스템
        self.ews = EarlyWarningSystem()
    def _load_score_history(self):
        """매도 점수 히스토리 로드"""
        try:
            with open(self.score_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    def _save_score_history(self):
        """매도 점수 히스토리 저장"""
        try:
            with open(self.score_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.sell_score_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"매도 점수 히스토리 저장 실패: {e}")
    def calculate_sell_score(self, h):
        """5 순위: 매도 점수 계산 + 트렌드 추적 (진입일 체크 추가)"""
        score = 0
        reasons = []
        
        # ✅ 신규 보유 종목 (진입 1 일차) 는 매도 신호 제외
        entry_date = h.get('entry_date')
        if entry_date:
            try:
                from datetime import datetime
                entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
                days_held = (datetime.now() - entry_dt).days
                if days_held < 1:  # 진입 당일
                    logging.debug(f"✅ {h['ticker']} 진입 1 일차 - 매도 신호 제외 (보유일수: {days_held}일)")
                    return 0, []  # 매도 신호 없음
            except Exception as e:
                logging.debug(f"진입일 체크 실패: {e}")
        
        stop_loss_price = h.get('stop_loss', 0)
        if stop_loss_price > 0 and h['current_price'] < stop_loss_price: score += 100; reasons.append("손절가 이탈")
        if h['expert_score'] > 0 and h['expert_score'] < 40: score += 50; reasons.append(f"Expert {h['expert_score']}점 (미달)")
        if h['rs_score'] > 0 and h['rs_score'] < 40: score += 30; reasons.append(f"RS {h['rs_score']}점 (약세)")
        if h.get('trend_score', 0) < 0: score += 20; reasons.append("하락 추세 전환")
        
        # 트렌드 추적 (이전 점수와 비교)
        ticker = h['ticker']
        prev_score = self.sell_score_history.get(ticker, 0)
        if score > prev_score and score >= 50:
            reasons.append(f"점수 급등 ({prev_score}→{score})")
        self.sell_score_history[ticker] = score
        
        return score, reasons
    def get_current_sector_allocation(self, total_asset):
        """
        섹터별 자산 배분 비율 계산 (현금 포함)
        
        Returns:
            dict: {
                'Technology': 0.25,      # 주식 섹터
                'Financials': 0.15,
                '현금': 0.30,            # 현금 비중 (추가됨)
                ...
            }
        """
        allocation = {}
        stock_total = 0
        
        # 1. 주식 섹터별 평가금 계산
        for h in self.holdings:
            fund = self.fetcher.get_data(h['ticker'])
            sector = fund.get('sector', '기타') if fund else '기타'
            val = h['current_price'] * h['qty'] * self.usd_krw
            allocation[sector] = allocation.get(sector, 0) + val
            stock_total += val
        
        # 2. 현금 추가 (총자산 - 주식평가금)
        cash = max(0, total_asset - stock_total)
        allocation['현금'] = cash
        
        # 3. 비율로 변환
        for k in allocation:
            allocation[k] = allocation[k] / total_asset
        
        return allocation
    def generate_plan(self, expert_results, turtle_results, rs_results, total_asset_val):
        logging.info("[7/7] 포트폴리오 리밸런싱 계산 중...")
        total_sell_proceeds = 0
        current_alloc = self.get_current_sector_allocation(total_asset_val)
        
        # V25.0: 섹터 모멘텀 계산 (조기경보용)
        sector_momentum = {}
        for sector, etf in SECTOR_ETFS.items():
            try:
                etf_df = yf.download(etf, period='1y', progress=False)
                if len(etf_df) >= 21:
                    ret_1m = (etf_df['Close'].iloc[-1] / etf_df['Close'].iloc[-21] - 1) * 100
                    sector_momentum[sector] = ret_1m
            except:
                sector_momentum[sector] = 0
        
        for h in self.holdings:
            score, reasons = self.calculate_sell_score(h)
            
            # V25.0: 조기경보 긴급도 계산
            ticker = h['ticker']
            fund = self.fetcher.get_data(ticker)
            holding_data = {
                'rs_score': h.get('rs_score', 50),
                'sector': fund.get('sector', 'Default') if fund else 'Default'
            }
            try:
                df = yf.download(ticker, period='1y', progress=False)
                urgency_data = self.ews.calculate_urgency(ticker, holding_data, {t: r['rs_score'] for t, r in expert_results.items()}, sector_momentum, df)
            except:
                urgency_data = {'urgency': 0, 'momentum_decay': 0, 'rs_reversal': 0, 'sector_rotation': 0, 'momentum_trend': [], 'replace_candidates': []}
            
            # 긴급도 기반 액션 수정
            action = "KEEP"
            if urgency_data['urgency'] >= 70: action = "즉시 교체 검토 (조기경보)"
            elif urgency_data['urgency'] >= 40: action = "교체 관찰 (조기경보)"
            elif score >= 100: action = "즉시 매도 (손절)"
            elif score >= 50: action = "매도 권장 (펀더멘털)"
            elif score >= 30: action = "교체 검토 (모멘텀)"
            
            if action != "KEEP":
                est_value = h['current_price'] * h['qty'] * self.usd_krw
                self.sell_candidates.append({
                    'ticker': h['ticker'],
                    'action': action,
                    'reason': ", ".join(reasons),
                    'est_value': est_value,
                    'qty': h['qty'],
                    'score': score,
                    'urgency': urgency_data['urgency'],
                    'urgency_details': urgency_data
                })
                total_sell_proceeds += est_value
        tier1 = [] 
        t_map = {r['ticker']: r for r in turtle_results}
        e_map = {r['ticker']: r for r in expert_results}
        for t_ticker, t_data in t_map.items():
            if '매수' in t_data['signal'] and t_ticker in e_map:
                e_data = e_map[t_ticker]
                sector = e_data['sector']
                limit = SECTOR_LIMITS.get(sector, 1.0)
                curr_exposure = current_alloc.get(sector, 0)
                warning_msg = ""
                if curr_exposure >= limit: warning_msg = f" <span style='color:red;font-weight:bold;font-size:0.8em'>⚠️비중초과 ({int(curr_exposure*100)}%)</span>"
                if e_data['score'] >= 70: tier1.append({'ticker': t_ticker, 'tier': 'Tier 1 (교집합)', 'price': t_data['price'], 'reason': f"교집합 {warning_msg}", 'score': e_data['score']})
        tier1.sort(key=lambda x: x['score'], reverse=True)
        tier2 = [] 
        for e in expert_results:
            sector = e['sector']
            limit = SECTOR_LIMITS.get(sector, 1.0)
            curr_exposure = current_alloc.get(sector, 0)
            warning_msg = ""
            if curr_exposure >= limit: warning_msg = f" <span style='color:red;font-weight:bold;font-size:0.8em'>⚠️비중초과 ({int(curr_exposure*100)}%)</span>"
            if e['score'] >= 80 and e['ticker'] not in [x['ticker'] for x in tier1]: tier2.append({'ticker': e['ticker'], 'tier': 'Tier 2 (Expert)', 'price': e['close'], 'reason': f"우량주 {warning_msg}", 'score': e['score']})
        tier2.sort(key=lambda x: x['score'], reverse=True)
        total_cash = self.cash + total_sell_proceeds
        
        # 4 순위: 시나리오별 종목 선정 기준 분리
        required_cash_ratio = self.ma.cash_ratio  # 변수 정의 추가
        all_candidates = tier1 + tier2
        
        # 시나리오 A (보수적): 상위 20% 만 선정 (Tier 1 위주)
        max_a = max(1, int(len(all_candidates) * 0.2))
        buy_candidates_a = all_candidates[:max_a] if max_a > 0 else all_candidates[:1]
        
        # 시나리오 B (균형): 상위 50% 선정
        max_b = max(1, int(len(all_candidates) * 0.5))
        buy_candidates_b = all_candidates[:max_b]
        
        # 시나리오 C (공격): 상위 80% 선정
        max_c = max(1, int(len(all_candidates) * 0.8))
        buy_candidates_c = all_candidates[:max_c]
        
        safe_cash_ratios = {
            'A': max(required_cash_ratio, 0.8),
            'B': max(required_cash_ratio, 0.5),
            'C': required_cash_ratio
        }
        
        alloc_a = total_cash * (1.0 - safe_cash_ratios['A'])
        buy_a = self._allocate_smart_sniper(alloc_a, buy_candidates_a)
        alloc_b = total_cash * (1.0 - safe_cash_ratios['B'])
        buy_b = self._allocate_smart_sniper(alloc_b, buy_candidates_b)
        alloc_c = total_cash * (1.0 - safe_cash_ratios['C'])
        buy_c = self._allocate_smart_sniper(alloc_c, buy_candidates_c)
        
        self.scenarios = {
            'A': {'name': '보수적 (분산)', 'buy_list': buy_a, 'cash_reserve': total_cash - alloc_a, 'cash_ratio': safe_cash_ratios['A']},
            'B': {'name': '균형 (분산)', 'buy_list': buy_b, 'cash_reserve': total_cash - alloc_b, 'cash_ratio': safe_cash_ratios['B']},
            'C': {'name': '공격 (분산)', 'buy_list': buy_c, 'cash_reserve': total_cash - alloc_c, 'cash_ratio': safe_cash_ratios['C']}
        }
        # 5 순위: 매도 점수 히스토리 저장
        self._save_score_history()
        return self.sell_candidates, self.scenarios
    def _allocate_smart_sniper(self, budget, candidates):
        if not candidates or budget < MIN_INVEST_PER_STOCK: return []
        target_count = min(len(candidates), int(budget / self.min_invest_per_stock))
        if target_count < 1: target_count = 1 
        targets = candidates[:target_count] 
        if not targets: return []
        per_stock = budget / len(targets)
        plan = []
        for c in targets:
            qty = int(per_stock / (c['price'] * self.usd_krw))
            if qty > 0: plan.append({'ticker': c['ticker'], 'qty': qty, 'amount': qty * c['price'] * self.usd_krw})
        return plan

class IntegratedQuantSystem:
    """P0-5: as_of_date 지원 (백테스트용)"""
    def __init__(self, total_account_value, as_of_date=None):
        self.total_account_value = total_account_value
        self.as_of_date = as_of_date  # P0-5: 백테스트 기준일
        self.usd_krw = DEFAULT_EXCHANGE_RATE
        self.data_downloader = EnhancedDataDownloader()
        self.fetcher = SmartFundamentalsFetcher(DATA_DIR)
        self.macro_analyzer = MacroAnalyzer(data_dir=DATA_DIR, as_of_date=as_of_date)  # P0-5: as_of_date 전달
        self.rebalance_manager = None
        self.holdings_value_krw = 0
        self.cash_balance = 0
        self.benchmark_returns = {} 

    def get_exchange_rate(self):
        """환율 조회 + 캐싱 (API 실패시 최근 값 사용)"""
        cache_file = os.path.join(CACHE_DIR, "exchange_rate.json")
        
        # P0-2: USDKRW=X 사용 (정방향)
        try:
            df = yf.download("USDKRW=X", period="5d", progress=False)
            if not df.empty:
                rate = df['Close'].iloc[-1]
                if hasattr(rate, 'item'): rate = rate.item()
                self.usd_krw = round(rate, 2)
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump({'rate': self.usd_krw, 'ts': time.time()}, f)
                except: pass
                return
        except Exception as e:
            logging.warning(f"환율 조회 실패 (yfinance): {e}")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    age_hours = (time.time() - cached['ts']) / 3600
                    if age_hours < EXCHANGE_CACHE_MAX_AGE_HOURS:
                        self.usd_krw = cached['rate']
                        logging.info(f"환율 캐시 사용: {self.usd_krw}원 ({age_hours:.1f}시간 전)")
                        return
        except: pass
        
        logging.info(f"환율 데이터 없음, 기본값 사용: {DEFAULT_EXCHANGE_RATE} 원")
        self.usd_krw = DEFAULT_EXCHANGE_RATE

    def calculate_sector_score(self, ticker, fund_data, price_data):
        sector = fund_data.get('sector', 'Default')
        
        # 3 순위: 섹터 정규화 딕셔너리 사용
        if sector not in SECTOR_SCORING:
            if ticker in SOX_TICKERS:
                sector = 'Semiconductors'
            elif sector in SECTOR_NORMALIZE:
                sector = SECTOR_NORMALIZE[sector]
            else:
                sector = 'Default'
        
        config = SECTOR_SCORING.get(sector, SECTOR_SCORING['Default'])
        w = config['weights']
        t = config['thresh']
        score = 0
        roe = fund_data.get('roe', 0)
        if roe >= t.get('roe', 15): score += w.get('roe', 0)
        elif roe > 0: score += w.get('roe', 0) * 0.5
        growth = fund_data.get('growth', 0)
        if 'growth' in w:
            if growth >= t.get('growth', 10): score += w['growth']
            elif growth > 0: score += w['growth'] * 0.5
        div = fund_data.get('div', 0)
        if 'div' in w:
            if div >= t.get('div_min', 2.0): score += w['div']
            elif div > 0: score += w['div'] * 0.5
        pbr = fund_data.get('pbr', 0)
        per = fund_data.get('per', 0)
        # P2-3: PBR 계산 - 초과해도 부분 점수 (금융주 PBR=1.5~2.0 대응)
        if 'pbr' in w:
            pbr_max = t.get('pbr_max', 3.0)
            if 0 < pbr <= pbr_max:
                score += w['pbr']  # 만점
            elif pbr <= pbr_max * 1.5:
                score += w['pbr'] * 0.5  # 절반 점수
            # pbr_max*1.5 초과: 0 점
        if per > t.get('per_max', 30) * 1.5: score -= 5
        
        # P2-5: debt 페널티 로직 수정 (정상일 때 점수 추가, 나쁠 때 감점)
        debt = fund_data.get('debt_ratio', 0)
        debt_weight = abs(w.get('debt', 0))  # 음수 가중치를 양수로 변환
        if debt_weight > 0:
            if debt <= t.get('debt_max', 150):
                # 부채비율 정상: 만점
                score += debt_weight
            elif debt <= t.get('debt_max', 150) * 2:
                # 부채비율 약간 높음: 절반 점수
                score += debt_weight * 0.5
            # 부채비율 매우 높음: 점수 없음 (감점)
        
        rs_score = price_data.get('rs_score', 50)
        score += w.get('rs', 20) * (rs_score / 100.0)
        if 'momentum' in w:
            mom = price_data.get('momentum', 0)
            if mom > 10: score += w['momentum']
            elif mom > 0: score += w['momentum'] * 0.5
        return round(max(0, min(100, score)), 1), sector

    def get_macro_data(self):
        logging.info("[1/7] 글로벌 시장 & S&P500 분석 중...")
        results = {}
        try:
            spy = yf.download('^GSPC', period="1y", progress=False)
            if not spy.empty:
                close = spy['Close']
                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                curr = close.iloc[-1]
                ago = close.iloc[-63] if len(close) >= 63 else close.iloc[0]
                if hasattr(curr, 'item'): curr = curr.item()
                if hasattr(ago, 'item'): ago = ago.item()
                self.benchmark_returns['3m'] = (curr / ago - 1) * 100
                logging.info(f"S&P500 3 개월 수익률: {self.benchmark_returns['3m']:.2f}% (RS 기준)")
        except: self.benchmark_returns['3m'] = 0
        try:
            data = yf.download(list(MACRO_ASSETS.keys()), period=MACRO_PERIOD, group_by='ticker', progress=False)
            for t, name in MACRO_ASSETS.items():
                try:
                    df = pd.DataFrame()
                    if isinstance(data.columns, pd.MultiIndex) and t in data.columns.levels[0]: df = data[t].copy()
                    elif t in data.columns: df = data.copy()
                    if 'Close' in df.columns: df = df['Close'].dropna()
                    else: continue
                    if isinstance(df, pd.DataFrame): df = df.iloc[:, 0]
                    if len(df) > 2:
                        curr = df.iloc[-1]
                        prev = df.iloc[-2]
                        ma120 = df.rolling(120).mean().iloc[-1]
                        if hasattr(curr, 'item'): curr = curr.item()
                        if hasattr(prev, 'item'): prev = prev.item()
                        if hasattr(ma120, 'item'): ma120 = ma120.item()
                        pct = (curr - prev) / prev * 100
                        trend = "상승추세" if curr > ma120 else "하락추세"
                        results[t] = {'name': name, 'price': curr, 'pct': pct, 'trend': trend}
                except: pass
        except: pass
        return results

    def _extract_ticker_data(self, all_data, ticker):
        """P3-3: 멀티티커 파싱 안정화 (yfinance 버전 호환)"""
        if all_data.empty: return pd.DataFrame()
        try:
            if isinstance(all_data.columns, pd.MultiIndex):
                if ticker in all_data.columns.levels[0]:
                    df = all_data[ticker].copy()
                    df.dropna(subset=['Close'], inplace=True)
                    return df
            # P3-3: MY_TICKERS 에 대해서만 개별 다운로드 폴백 (Rate Limit 방지)
            if ticker in MY_TICKERS:
                logging.debug(f"{ticker} 보유종목, 개별 다운로드 시도")
                try:
                    df = yf.download(ticker, period=DOWNLOAD_PERIOD, progress=False)
                    if not df.empty and 'Close' in df.columns:
                        return df
                except Exception as e:
                    logging.debug(f"개별 다운로드 실패 ({ticker}): {e}")
        except Exception as e:
            logging.debug(f"Extract error for {ticker}: {e}")
        return pd.DataFrame()

    def calculate_rs_scores_bulk(self, all_tickers, all_data, fetcher=None):
        """
        P3-4: 전 종목 RS 수익률 계산 + 산업 섹터 내/전체 혼합 백분위
        
        Args:
            all_tickers: 전체 종목 목록
            all_data: yfinance 다운로드 데이터
            fetcher: SmartFundamentalsFetcher (섹터 정보 조회용)
        
        Returns:
            tuple: (blended_rs_map, overall_rs_map, returns_map)
                - blended_rs_map: 산업섹터내 30% + 전체 70% 혼합 백분위
                - overall_rs_map: 전체 유니버스 대비 백분위
                - returns_map: 3 개월 수익률 dict
        """
        returns = {}
        sector_returns = {}  # 산업 섹터별 수익률 저장 (Technology, Energy 등)
        
        for t in all_tickers:
            df = self._extract_ticker_data(all_data, t)
            if df.empty or len(df) < RS_PERIOD_DAYS: continue
            try:
                curr = df['Close'].iloc[-1]
                old = df['Close'].iloc[-RS_PERIOD_DAYS]
                if hasattr(curr, 'item'): curr = curr.item()
                if hasattr(old, 'item'): old = old.item()
                ret = (curr / old - 1) * 100
                returns[t] = ret
                
                # P3-4: 산업 섹터별 분류 (fetcher + 휴리스틱)
                sector = '기타'
                if fetcher:
                    fund_data = fetcher.get_data(t)
                    if fund_data and fund_data.get('sector'):
                        sector = fund_data['sector']
                
                # P3-4: fetcher 에 없으면 휴리스틱으로 섹터 할당
                if sector == '기타':
                    if t in SOX_TICKERS:
                        sector = 'Semiconductors'  # 반도체
                    # 나스닥 주요 기술주 (간단한 휴리스틱)
                    elif t in ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL', 'INTC']:
                        sector = 'Technology'
                
                if sector not in sector_returns:
                    sector_returns[sector] = {}
                sector_returns[sector][t] = ret
            except: continue
        
        if not returns:
            return {}, {}, {}
        
        # 전체 유니버스 백분위
        tickers = list(returns.keys())
        vals = [returns[t] for t in tickers]
        overall_ranks = pd.Series(vals).rank(pct=True) * 100
        overall_rs_map = {t: round(overall_ranks.iloc[i], 1) for i, t in enumerate(tickers)}
        
        # 산업 섹터 내 백분위 (P3-4: 진짜 섹터 특성 반영)
        sector_rs_map = {}
        for sector, stocks in sector_returns.items():
            if len(stocks) < 2:
                # 섹터 종목이 1 개면 전체 RS 사용
                for t in stocks:
                    sector_rs_map[t] = overall_rs_map[t]
            else:
                tickers_s = list(stocks.keys())
                vals_s = [stocks[t] for t in tickers_s]
                ranks_s = pd.Series(vals_s).rank(pct=True) * 100
                for i, t in enumerate(tickers_s):
                    sector_rs_map[t] = round(ranks_s.iloc[i], 1)
        
        # 혼합: 전체 70% + 산업섹터내 30% (P3-4)
        blended_rs_map = {}
        for t in tickers:
            overall = overall_rs_map.get(t, 50)
            sector = sector_rs_map.get(t, overall)
            blended = overall * 0.7 + sector * 0.3
            blended_rs_map[t] = round(blended, 1)
        
        return blended_rs_map, overall_rs_map, returns
    
    # P0-4: calculate_momentum_13612 는 utils.momentum 에서 import 사용
    def calculate_momentum_13612_wrapper(self, prices):
        try:
            return self.calculate_momentum_13612(prices)
        except Exception as e:
            logging.debug(f"13612W 계산 실패: {e}")
            return 0.0
    
    def calculate_rs_score(self, ticker, df, rs_map=None, returns_map=None):
        """V24.9: 단순 RS + 13612W 모멘텀 결합"""
        if df.empty or len(df) < RS_PERIOD_DAYS: 
            return {'score': 50, 'stock_ret': 0, 'diff': 0, 'momentum_13612': 0}
        try:
            # 1. 단순 RS (63 일)
            curr = df['Close'].iloc[-1]
            old = df['Close'].iloc[-RS_PERIOD_DAYS]
            if hasattr(curr, 'item'): curr = curr.item()
            if hasattr(old, 'item'): old = old.item()
            stock_ret = (curr / old - 1) * 100
            spy_ret = self.benchmark_returns.get('3m', 0)
            diff = stock_ret - spy_ret
            
            # 2. 13612W 모멘텀 (V24.9) - P0-4: 통합 함수 사용
            momentum_13612 = calculate_momentum_13612(df['Close'])
            benchmark_momentum = getattr(self, 'benchmark_momentum', 0)
            momentum_diff = momentum_13612 - benchmark_momentum
            momentum_score = 50 + (momentum_diff * 2.0)
            momentum_score = max(0, min(100, momentum_score))
            
            # 3. 종합 점수 (백분위 40% + 모멘텀 60%)
            if rs_map and ticker in rs_map:
                rs_percentile = rs_map[ticker]
            else:
                rs_percentile = 50 + (diff * 1.5)
                rs_percentile = max(0, min(100, rs_percentile))
            
            final_score = (rs_percentile * 0.4) + (momentum_score * 0.6)
            
            return {
                'score': round(final_score, 1),
                'stock_ret': round(stock_ret, 1),
                'diff': round(diff, 1),
                'momentum_13612': round(momentum_13612, 1)
            }
        except: 
            return {'score': 50, 'stock_ret': 0, 'diff': 0, 'momentum_13612': 0}
    
    def _calculate_stop_loss(self, entry_price, current_price, atr):
        """
        V24.9: Stop-loss 이중 로직
        
        1. 고정 손절: 진입가 기준 -8% (하드 스탑)
        2. 트레일링 스탑: 현재가 - 2×ATR (소프트 스탑)
        3. 둘 중 높은 값 사용 (더 보수적)
        """
        # 1. 고정 손절 (진입가 기준 -8%)
        fixed_stop = entry_price * 0.92
        
        # 2. 트레일링 스탑 (현재가 기준)
        trailing_stop = current_price - (2 * atr) if atr > 0 else 0
        
        # 3. 더 높은 값 사용 (더 보수적인 손절가)
        return max(fixed_stop, trailing_stop)

    def detect_volume_spike(self, df):
        if len(df) < VOLUME_AVG_DAYS: return False, 0
        avg_vol = df['Volume'].iloc[-VOLUME_AVG_DAYS:-1].mean()
        curr_vol = df['Volume'].iloc[-1]
        if hasattr(avg_vol, 'item'): avg_vol = avg_vol.item()
        if hasattr(curr_vol, 'item'): curr_vol = curr_vol.item()
        if avg_vol <= 0: return False, 0.0
        ratio = curr_vol / avg_vol
        return ratio >= 1.5, round(ratio, 1)

    def run_turtle_strategy(self, all_tickers, universe_map, all_data, rs_map=None):
        logging.info("[4/7] Turtle 전략 실행...")
        results = []
        sector_stats = {'S&P500': {'total': 0, 'picked': 0}, '나스닥 100': {'total': 0, 'picked': 0}, '반도체 (SOX)': {'total': 0, 'picked': 0}}
        market_score = self.macro_analyzer.market_score
        ignore_s1 = market_score < 60 
        for t in all_tickers:
            if t in BLACKLIST: continue
            univ = universe_map.get(t, 'S&P500')
            if univ in sector_stats: sector_stats[univ]['total'] += 1
        for t in all_tickers:
            if t in BLACKLIST: continue
            univ = universe_map.get(t, 'S&P500')
            try:
                df = self._extract_ticker_data(all_data, t)
                if df.empty or len(df) < 100: continue
                df['High_20'] = df['High'].rolling(20).max().shift(1)
                df['Low_10'] = df['Low'].rolling(10).min().shift(1)
                df['High_55'] = df['High'].rolling(55).max().shift(1)
                df['MA200'] = df['Close'].rolling(200).mean()
                df['TR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
                # P0-3: ATR 계산 - Wilder's Smoothing (Turtle 시스템 공식)
                df['N'] = df['TR'].ewm(alpha=1/20, adjust=False).mean()
                curr = df.iloc[-1]
                high_55 = curr['High_55']; high_20 = curr['High_20']; low_10 = curr['Low_10']; ma200 = curr['MA200']; atr = curr['N']
                price = curr['Close']
                if hasattr(price, 'item'): price = price.item()
                if hasattr(ma200, 'item'): ma200 = ma200.item()
                if hasattr(atr, 'item'): atr = atr.item()
                vol_spike, vol_ratio = self.detect_volume_spike(df)
                is_uptrend = price > ma200 
                is_vol_good = vol_ratio > 1.0 
                avg_atr = df['N'].iloc[-60:-1].mean()
                is_atr_expanding = atr > avg_atr * 0.9 
                signal = "관망"
                if price > high_55 and is_uptrend: signal = "S2 매수" 
                elif price > high_20 and is_uptrend and is_vol_good and is_atr_expanding:
                    if not ignore_s1: signal = "S1 매수"
                if t in MY_TICKERS:
                    my_pos = next((p for p in MY_POSITIONS if p['ticker'] == t), None)
                    if my_pos:
                        profit_pct = (price / my_pos['price'] - 1) * 100
                        if price > high_55 and profit_pct >= 20 and is_atr_expanding: signal = "불타기"
                        elif price < low_10: signal = "매도 (청산)"
                        else: signal = "HOLD"
                trend_score = 0
                if ma200 > 0: trend_score = (price - ma200) / ma200 * 100
                unit_qty = 0
                if atr > 0:
                    # 1. 변동성 기반 수량 (1% 리스크)
                    vol_qty = int((self.cash_balance * RISK_RATIO) / (atr * self.usd_krw))
                    
                    # 2. 최대 비중 제한 (20%)
                    max_qty = int((self.total_account_value * MAX_ALLOC_PER_STOCK) / (price * self.usd_krw))
                    
                    # 3. 현금 제한
                    cash_qty = int(self.cash_balance / (price * self.usd_krw))
                    
                    # 4. V24.9: 하프 켈리 추가 (승률 50%, 손익비 1.5 고정)
                    # Kelly 공식: f* = (승률×평균수익 - 패율×평균손실) / 평균수익
                    # Kelly = (0.50 * 1.5 - 0.50 * 1) / 1.5 = (0.75 - 0.50) / 1.5 = 0.167 (16.7%)
                    # Half Kelly = 0.167 * 0.5 = 0.083 (8.3%) → 보수적으로 8% 사용
                    half_kelly_ratio = DEFAULT_CONFIG['account'].get('half_kelly_ratio', 0.08)
                    kelly_qty = int((self.total_account_value * half_kelly_ratio) / (price * self.usd_krw))
                    
                    # 5. 4 중 안전장치 (변동성, 최대비중, 현금, 켈리)
                    unit_qty = min(vol_qty, max_qty, cash_qty, kelly_qty)
                exit_p = low_10
                if hasattr(exit_p, 'item'): exit_p = exit_p.item()
                res = {'ticker': t, 'price': price, 'signal': signal, 'trend_score': trend_score, 'N': atr, 
                       'vol_ratio': vol_ratio, 'vol_spike': vol_spike, 'exit_price': exit_p, 'universe': univ,
                       'unit_qty': unit_qty}
                if '매수' in signal or '불타기' in signal or t in MY_TICKERS:
                    results.append(res)
                    if ('매수' in signal or '불타기' in signal) and univ in sector_stats: sector_stats[univ]['picked'] += 1
                if t in TICKERS_TO_CHART: self.save_chart(t, df, df['Low_10'])
            except: pass
        for k, v in sector_stats.items():
            logging.info(f"  • {k}: {v['total']}개 중 {v['picked']}개 선정 ({v['picked']/v['total']*100:.1f}%)" if v['total']>0 else f"  • {k}: 데이터 없음")
        return {'results': results, 'stats': sector_stats}

    def save_chart(self, ticker, df, exit_line):
        """P0-2: 메모리 누수 방지 (try/finally)"""
        fig = None
        try:
            d = df.iloc[-120:]
            fig = plt.figure(figsize=(10, 5))
            plt.plot(d.index, d['Close'], 'k', label='Close')
            plt.plot(d.index, exit_line.iloc[-120:], 'r--', label='Exit(10d Low)')
            plt.title(f"{ticker} Chart")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(CHART_DIR, f"{ticker}_chart.png"), dpi=100)
        except Exception as e:
            logging.warning(f"Chart save error ({ticker}): {e}")
        finally:
            if fig is not None:
                plt.close(fig)

    def run_expert_strategy(self, all_tickers, universe_map, all_data, rs_map=None):
        logging.info("[5/7] Expert+ 우량주 전략 실행 중...")
        results = []
        sector_stats = {'S&P500': {'total': 0, 'picked': 0}, '나스닥 100': {'total': 0, 'picked': 0}, '반도체 (SOX)': {'total': 0, 'picked': 0}}
        failed_count = 0
        for t in all_tickers:
            if t in BLACKLIST: continue
            univ = universe_map.get(t, 'S&P500')
            if univ in sector_stats: sector_stats[univ]['total'] += 1
        candidates = []
        for t in all_tickers:
            if t in BLACKLIST: continue
            try:
                df = self._extract_ticker_data(all_data, t)
                if df.empty or len(df) < 100: continue
                # P0-2: 후보 필터링에도 rs_map 전달 (절대값 → 백분위 일관성)
                rs_temp = self.calculate_rs_score(t, df, rs_map, None)
                if rs_temp['score'] >= 65 or t in MY_TICKERS: candidates.append((t, df))
            except Exception as e:
                logging.warning(f"Candidate filter error for {t}: {e}")
                failed_count += 1
        logging.info(f"  • 정밀 분석 대상: {len(candidates)}개 (RS>65)")
        logging.info(f"  • 펀더멘털 데이터 수집 시작...")
        count = 0
        for t, df in candidates:
            count += 1
            if count % 10 == 0: logging.info(f"    진행률: {count}/{len(candidates)}...")
            is_priority = t in MY_TICKERS
            fund = self.fetcher.fetch_with_retry(t, priority=is_priority)
            if not fund: continue 
            try:
                univ = universe_map.get(t, 'S&P500')
                curr_price = df['Close'].iloc[-1]
                if hasattr(curr_price, 'item'): curr_price = curr_price.item()
                p_60 = df['Close'].iloc[-60]
                if hasattr(p_60, 'item'): p_60 = p_60.item()
                momentum = (curr_price/p_60-1)*100
                rs_data = self.calculate_rs_score(t, df, rs_map, None)
                price_data = {'rs_score': rs_data['score'], 'momentum': momentum}
                score, sector_name = self.calculate_sector_score(t, fund, price_data)
                max_qty = int((self.total_account_value * MAX_ALLOC_PER_STOCK) / (curr_price * self.usd_krw))
                cash_qty = int(self.cash_balance / (curr_price * self.usd_krw))
                final_qty = min(max_qty, cash_qty)
                if score >= EXPERT_CUTOFF or t in MY_TICKERS:
                    if score >= EXPERT_CUTOFF and univ in sector_stats: sector_stats[univ]['picked'] += 1
                    results.append({'ticker': t, 'score': score, 'roe': fund['roe'], 'growth': fund['growth'], 'div': fund['div'], 'pbr': fund['pbr'], 'per': fund['per'], 'debt': fund['debt_ratio'], 'rs_score': rs_data['score'], 'rs_diff': rs_data['diff'], 'sector': sector_name, 'universe': univ, 'close': curr_price, 'qty': final_qty})
            except Exception as e:
                logging.warning(f"Expert strategy error for {t}: {e}")
                failed_count += 1
                continue
        if failed_count > 0:
            logging.info(f"  • 예외 발생: {failed_count}개 (로그 기록됨)")
        for k, v in sector_stats.items():
            logging.info(f"  • {k}: {v['total']}개 중 {v['picked']}개 선정 ({v['picked']/v['total']*100:.1f}%)" if v['total']>0 else f"  • {k}: 데이터 없음")
        return {'results': results, 'stats': sector_stats}

    def run_rs_strategy(self, all_tickers, universe_map, all_data, rs_map=None):
        logging.info("[6/7] RS 주도주 전략 실행 중...")
        results = []
        sector_stats = {'S&P500': {'total': 0, 'picked': 0}, '나스닥 100': {'total': 0, 'picked': 0}, '반도체 (SOX)': {'total': 0, 'picked': 0}}
        for t in all_tickers:
            univ = universe_map.get(t, 'S&P500')
            if univ in sector_stats: sector_stats[univ]['total'] += 1
        for t in all_tickers:
            univ = universe_map.get(t, 'S&P500')
            try:
                df = self._extract_ticker_data(all_data, t)
                if df.empty or len(df) < 100: continue
                rs_data = self.calculate_rs_score(t, df, rs_map, None)
                vol_spike, vol_ratio = self.detect_volume_spike(df)
                curr_price = df['Close'].iloc[-1]
                if hasattr(curr_price, 'item'): curr_price = curr_price.item()
                if rs_data['score'] >= 70:
                    if univ in sector_stats: sector_stats[univ]['picked'] += 1
                    results.append({
                        'ticker': t, 
                        'rs_score': rs_data['score'], 
                        'stock_ret': rs_data['stock_ret'], 
                        'diff': rs_data['diff'], 
                        'momentum_13612': rs_data.get('momentum_13612', 0),  # V24.9 추가
                        'vol_ratio': vol_ratio, 
                        'universe': univ, 
                        'close': curr_price
                    })
            except: pass
        return {'results': results, 'stats': sector_stats}

    def _get_soxx_tickers(self):
        """SOXX ETF 구성종목 자동 수집 (SOXX ETF holdings + Wikipedia 병행)"""
        global SOX_TICKERS
        
        # 방법 1: SOXX ETF 상위 30 개 holdings (가장 정확)
        try:
            soxx = yf.Ticker("SOXX")
            if hasattr(soxx, 'top_holdings'):
                holdings = soxx.top_holdings
                if len(holdings) > 0:
                    tickers = holdings.index.tolist()[:30]
                    tickers = [t.strip().upper() for t in tickers if len(t.strip()) <= 5 and t.strip()]
                    if len(tickers) >= 20:
                        SOX_TICKERS = tickers
                        logging.info(f"  ✅ SOXX ETF holdings 수집: {len(SOX_TICKERS)}개 종목")
                        return SOX_TICKERS
        except Exception as e:
            logging.debug(f"  SOXX ETF holdings 실패: {e}")
        
        # 방법 2: Wikipedia PHLX Semiconductor Sector
        try:
            url = "https://en.wikipedia.org/wiki/PHLX_Semiconductor_Sector"
            headers = {'User-Agent': 'Mozilla/5.0'}
            html = requests.get(url, headers=headers, timeout=10).text
            
            # HTML 에서 티커 추출 (정규표현식 + 문맥)
            import re
            # "ticker" 근처에 있는 대문자 2-5 자 찾기
            pattern = r'(?:ticker|symbol|code)[^A-Z]*\b([A-Z]{2,5})\b'
            matches = re.findall(pattern, html, re.IGNORECASE)
            
            if len(matches) >= 20:
                tickers = list(dict.fromkeys(matches))[:30]  # 중복 제거
                SOX_TICKERS = tickers
                logging.info(f"  ✅ SOXX Wikipedia 수집: {len(SOX_TICKERS)}개 종목")
                return SOX_TICKERS
        except Exception as e:
            logging.debug(f"  Wikipedia SOX 실패: {e}")
        
        # 방법 3: yfinance 로 개별 티커 검증
        try:
            # 반도체 산업 대표 종목들 (최소 목록)
            candidates = ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 
                         'KLAC', 'ADI', 'MCHP', 'NXPI', 'MRVL', 'ON', 'MPWR', 'SWKS', 'QRVO', 'TER',
                         'ENTG', 'ALGM', 'WOLF', 'COHR', 'RMBS', 'LSCC', 'SLAB', 'CRUS', 'SITM', 'UCTT']
            
            valid_tickers = []
            for t in candidates:
                try:
                    ticker = yf.Ticker(t)
                    info = ticker.info
                    if info.get('sector') == 'Technology' and len(valid_tickers) < 30:
                        valid_tickers.append(t)
                except:
                    pass
            
            if len(valid_tickers) >= 20:
                SOX_TICKERS = valid_tickers
                logging.info(f"  ✅ SOXX yfinance 검증 수집: {len(SOX_TICKERS)}개 종목")
                return SOX_TICKERS
        except Exception as e:
            logging.debug(f"  yfinance 검증 실패: {e}")
        
        SOX_TICKERS = []
        return SOX_TICKERS
    
    def fetch_dynamic_tickers(self):
        """V24.9: SOXX 자동 수집 + 확장 유니버스 지원"""
        logging.info("[2/7] 유니버스 최신화...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        nasdaq, sp500 = [], []
        backup_dir = os.path.join(DATA_DIR, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        # S&P 500
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            html_text = requests.get(url, headers=headers, timeout=10).text
            sp500 = pd.read_html(io.StringIO(html_text))[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            logging.info(f"  • S&P 500: {len(sp500)}개")
            # 백업 저장
            with open(os.path.join(backup_dir, "sp500_backup.json"), 'w') as f:
                json.dump(sp500, f)
        except Exception as e: 
            logging.warning(f"  • S&P 500 크롤링 에러: {e}")
            # 백업에서 로드
            try:
                with open(os.path.join(backup_dir, "sp500_backup.json"), 'r') as f:
                    sp500 = json.load(f)
                logging.info(f"  • S&P 500 백업 로드: {len(sp500)}개")
            except:
                sp500 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK.B']

        # Nasdaq 100
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            html_text = requests.get(url, headers=headers, timeout=10).text
            dfs = pd.read_html(io.StringIO(html_text))
            for df in dfs:
                if 'Ticker' in df.columns: 
                    nasdaq = df['Ticker'].tolist()
                    break
            logging.info(f"  • Nasdaq 100: {len(nasdaq)}개")
            # 백업 저장
            with open(os.path.join(backup_dir, "nasdaq_backup.json"), 'w') as f:
                json.dump(nasdaq, f)
        except Exception as e: 
            logging.warning(f"  • Nasdaq 100 크롤링 에러: {e}")
            # 백업에서 로드
            try:
                with open(os.path.join(backup_dir, "nasdaq_backup.json"), 'r') as f:
                    nasdaq = json.load(f)
                logging.info(f"  • Nasdaq 100 백업 로드: {len(nasdaq)}개")
            except:
                nasdaq = ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMZN', 'META', 'TSLA', 'GOOGL']

        # SOX 반도체 (자동 수집 + 폴백)
        sox = self._get_soxx_tickers()
        
        # 기본 유니버스
        all_tickers = sorted(list(set(nasdaq + sp500 + sox + MY_TICKERS)))
        logging.info(f"  • 기본 유니버스: {len(all_tickers)}개")
        
        # 확장 유니버스 (러셀 2000 + 테마) - config.json 에서 제어
        try:
            config_path = os.path.join(BASE_DIR, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                expand = config.get('expand_universe', False)
                api_keys = config.get('api_keys', {})
                polygon_key = api_keys.get('polygon_api_key', '')
                
                logging.info(f"  [DEBUG] expand_universe: {expand}")
                logging.info(f"  [DEBUG] polygon_api_key: {'설정됨' if polygon_key else '없음'}")
                
                if expand:
                    logging.info("  🚀 확장 유니버스 활성화...")
                    
                    # 러셀 2000 상위 200 개 (IWM ETF holdings 기반 동적 수집)
                    if config.get('include_russell', True):
                        russell_added = False
                        russell = []
                        
                        # 캐시 확인 (7 일간 재사용)
                        cache_file = os.path.join(CACHE_DIR, "russell2000_tickers.json")
                        cache_valid = False
                        try:
                            if os.path.exists(cache_file):
                                with open(cache_file, 'r', encoding='utf-8') as f:
                                    cached_data = json.load(f)
                                cache_date = cached_data.get('date', '')
                                cache_tickers = cached_data.get('tickers', [])
                                from datetime import datetime, timedelta
                                if cache_date and len(cache_tickers) >= 50:
                                    cache_dt = datetime.strptime(cache_date, '%Y-%m-%d')
                                    if datetime.now() - cache_dt < timedelta(days=7):
                                        russell = cache_tickers[:200]
                                        logging.info(f"  • 러셀 2000: {len(russell)}개 (캐시, {cache_date})")
                                        all_tickers = sorted(list(set(all_tickers + russell)))
                                        russell_added = True
                                        cache_valid = True
                        except Exception as e:
                            logging.debug(f"  캐시 읽기 실패: {e}")
                        
                        # P1-7: Polygon.io 러셀 2000 필터 수정 (index=RUT 만 수집)
                        if polygon_key:
                            try:
                                logging.info(f"  [DEBUG] Polygon.io API 시도...")
                                # P1-7: index=RUT 필터 추가 (러셀 2000 만)
                                url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&index=RUT&limit=2000&apiKey={polygon_key}"
                                response = requests.get(url, timeout=15)
                                if response.status_code == 200:
                                    data = response.json()
                                    if 'results' in data:
                                        tickers = [t['ticker'] for t in data['results'] if 'ticker' in t]
                                        if len(tickers) >= 50:
                                            logging.info(f"  • 러셀 2000: {len(tickers)}개 (Polygon.io API)")
                                            all_tickers = sorted(list(set(all_tickers + tickers)))
                                            russell_added = True
                                else:
                                    logging.warning(f"  Polygon.io 응답: {response.status_code}")
                            except Exception as e:
                                logging.warning(f"  Polygon.io 실패: {e}")
                        
                        # IWM ETF holdings (폴백)
                        if not russell_added:
                            try:
                                iwm = yf.Ticker("IWM")
                                if hasattr(iwm, 'top_holdings'):
                                    holdings = iwm.top_holdings
                                    if len(holdings) > 0:
                                        tickers = holdings.index.tolist()[:200]
                                        tickers = [t.strip().upper() for t in tickers if len(t.strip()) <= 5 and t.strip()]
                                        if len(tickers) >= 50:
                                            russell = tickers
                                            logging.info(f"  • 러셀 2000: {len(russell)}개 (IWM ETF holdings)")
                                            all_tickers = sorted(list(set(all_tickers + russell)))
                                            russell_added = True
                            except Exception as e:
                                logging.debug(f"  IWM ETF holdings 실패: {e}")
                        
                        # 실패 시: 수집 안 함 (잘못된 데이터보다 없는 데이터가 낫음)
                        if not russell_added:
                            logging.warning(f"  ⚠️ 러셀 2000 수집 실패 - 스킵 (하드코딩 제거)")
                        
                        # 캐시 저장
                        if russell_added and not cache_valid and len(russell) >= 50:
                            try:
                                os.makedirs(CACHE_DIR, exist_ok=True)
                                cache_data = {
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'tickers': russell,
                                    'count': len(russell)
                                }
                                with open(cache_file, 'w', encoding='utf-8') as f:
                                    json.dump(cache_data, f, indent=2)
                                logging.debug(f"  러셀 2000 캐시 저장됨 ({len(russell)}개)")
                            except Exception as e:
                                logging.debug(f"  캐시 저장 실패: {e}")
                    
                    # 테마 종목 - ETF holdings 기반 동적 수집
                    if config.get('include_thematic', True):
                        thematic_added = False
                        thematic_tickers = []
                        
                        # 캐시 확인
                        cache_file = os.path.join(CACHE_DIR, "thematic_tickers.json")
                        try:
                            if os.path.exists(cache_file):
                                with open(cache_file, 'r', encoding='utf-8') as f:
                                    cached_data = json.load(f)
                                cache_date = cached_data.get('date', '')
                                cache_tickers = cached_data.get('tickers', [])
                                from datetime import datetime, timedelta
                                if cache_date and len(cache_tickers) >= 10:
                                    cache_dt = datetime.strptime(cache_date, '%Y-%m-%d')
                                    if datetime.now() - cache_dt < timedelta(days=7):
                                        thematic_tickers = cache_tickers
                                        logging.info(f"  • 테마 종목: {len(thematic_tickers)}개 (캐시)")
                                        thematic_added = True
                        except:
                            pass
                        
                        # ETF top_holdings 에서 수집
                        if not thematic_added:
                            try:
                                etf_map = {
                                    '사이버보안': 'CIBR',
                                    '클라우드': 'CLOU',
                                    '바이오': 'IBB',
                                    '핀테크': 'FINX',
                                    '클린에너지': 'ICLN'
                                }
                                
                                for theme, etf in etf_map.items():
                                    try:
                                        etf_ticker = yf.Ticker(etf)
                                        if hasattr(etf_ticker, 'top_holdings'):
                                            holdings = etf_ticker.top_holdings
                                            if len(holdings) > 0:
                                                tickers = holdings.index.tolist()[:8]
                                                thematic_tickers.extend([t for t in tickers if len(t) <= 5])
                                                thematic_added = True
                                    except Exception as e:
                                        logging.debug(f"  {theme} ETF 실패: {e}")
                                
                                if thematic_added and len(thematic_tickers) >= 10:
                                    all_tickers = sorted(list(set(all_tickers + thematic_tickers)))
                                    logging.info(f"  • 테마 종목: {len(thematic_tickers)}개 (ETF holdings)")
                            except Exception as e:
                                logging.debug(f"  ETF holdings 실패: {e}")
                        
                        # 캐시 저장
                        if thematic_added and len(thematic_tickers) >= 10:
                            try:
                                os.makedirs(CACHE_DIR, exist_ok=True)
                                with open(cache_file, 'w', encoding='utf-8') as f:
                                    json.dump({'date': datetime.now().strftime('%Y-%m-%d'), 'tickers': thematic_tickers}, f)
                            except:
                                pass
                    
                    logging.info(f"  • 최종 유니버스: {len(all_tickers)}개")
        except Exception as e:
            logging.debug(f"  확장 유니버스 설정 읽기 실패: {e}")
        
        return all_tickers, nasdaq, sp500, sox
    
    def _build_holdings_data(self, turtle_map, expert_map):
        """P2-1: 보유 종목 데이터 구축 (분리)"""
        my_holdings_data = []
        for p in MY_POSITIONS:
            t = p['ticker']
            curr_t = turtle_map.get(t, {})
            curr_e = expert_map.get(t, {})
            price = curr_t.get('price', p['price'])
            atr = curr_t.get('N', 0)
            # V24.9: Stop-loss 이중 로직 (진입가 + 현재가)
            stop_loss = self._calculate_stop_loss(p['price'], price, atr)
            reasons = []
            e_score = curr_e.get('score', 0)
            rs_score = curr_e.get('rs_score', 0)
            if e_score > 0 and e_score < 40: reasons.append(f"Expert {e_score}점 (미달)")
            if rs_score > 0 and rs_score < 40: reasons.append(f"RS {rs_score}점 (약세)")
            signal = curr_t.get('signal', 'HOLD')
            if reasons: signal += f"|{' '.join(reasons)}"
            my_holdings_data.append({
                'ticker': t, 'qty': p['qty'], 'current_price': price,
                'signal': signal, 'expert_score': e_score, 'rs_score': rs_score,
                'trend_score': curr_t.get('trend_score', 0), 'stop_loss': stop_loss,
                'entry_date': p.get('entry_date', '')  # ✅ 진입일 추가 (신규 종목 매도 방지)
            })
        return my_holdings_data
    
    def _build_canary_html(self):
        """
        V24.9: 카나리아 현황판 (3 레이어)
        
        레이어 1: 최상단 — 즉각적인 결론
        레이어 2: 중간 — 4 종목 상세 카드
        레이어 3: 하단 — 히스토리
        """
        canary = getattr(self.macro_analyzer, 'canary_signals', {})
        if not canary:
            return ""
        
        mode = getattr(self.macro_analyzer, 'canary_mode', '공격')
        mode_color = getattr(self.macro_analyzer, 'canary_mode_color', '#2ecc71')
        mode_icon = getattr(self.macro_analyzer, 'canary_mode_icon', '🟢')
        neg_count = getattr(self.macro_analyzer, 'canary_negative_count', 0)
        
        # 레이어 1: 최상단 — 즉각적인 결론
        if mode == '공격':
            mode_desc = "카나리아 이상 없음"
            action_desc = "공격 모드 (풀 투자)"
        elif mode == '주의':
            mode_desc = "1 개 종목 이탈"
            action_desc = "주의 모드 (50% 방어)"
        else:
            mode_desc = f"{neg_count}개 종목 이탈"
            action_desc = "방어 모드 (현금/채권)"
        
        html = f"""
        <div style='margin-bottom:25px;background:#fff;padding:20px;border-radius:12px;box-shadow:0 4px 8px rgba(0,0,0,0.1)'>
            <h2 style='margin:0 0 15px 0;color:#2c3e50'>🕊️ 카나리아 현황판</h2>
            
            <!-- 레이어 1: 최상단 — 즉각적인 결론 -->
            <div style='background:{mode_color};color:white;padding:20px;border-radius:8px;margin-bottom:20px;text-align:center'>
                <div style='font-size:1.5em;font-weight:bold;margin-bottom:5px'>{mode_icon} {mode} 모드</div>
                <div style='font-size:1.1em;opacity:0.95'>{mode_desc}</div>
                <div style='font-size:0.95em;opacity:0.9;margin-top:5px'>{action_desc}</div>
            </div>
            
            <!-- 레이어 2: 중간 — 4 종목 상세 카드 -->
            <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px'>
        """
        
        for ticker, data in canary.items():
            ret_3m = data.get('ret_3m', 0)
            is_neg = data.get('negative', False)
            color = "#e74c3c" if is_neg else "#2ecc71"
            icon = "🔴" if is_neg else "🟢"
            
            ret_1m = data.get('ret_1m', 0)
            ret_12m = data.get('ret_12m', 0)
            ma200_status = data.get('ma200_status', '→')
            momentum_13612 = data.get('momentum_13612', 0)
            
            html += f"""
                <div style='background:{color}10;border:2px solid {color};border-radius:8px;padding:15px'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
                        <div style='font-weight:bold;font-size:1.2em'>{ticker}</div>
                        <div style='font-size:1.5em'>{icon}</div>
                    </div>
                    <div style='font-size:0.85em;color:#666;margin-bottom:8px'>{data.get('name','')}</div>
                    <div style='font-size:1.3em;font-weight:bold;color:{color};margin-bottom:10px'>{ret_3m:+.1f}%</div>
                    <div style='font-size:0.8em;color:#555;line-height:1.6'>
                        <div>1M: <b style='color={ "#e74c3c" if ret_1m<0 else "#2ecc71"}'>{ret_1m:+.1f}%</b></div>
                        <div>12M: <b style='color={ "#e74c3c" if ret_12m<0 else "#2ecc71"}'>{ret_12m:+.1f}%</b></div>
                        <div>MA200: <b>{ma200_status}</b></div>
                        <div>13612W: <b>{momentum_13612:+.1f}%</b></div>
                    </div>
                </div>
            """
        
        html += """</div>"""
        
        # 레이어 3: 하단 — 히스토리 (수정: self.macro_analyzer.data_dir 사용)
        history_file = os.path.join(self.macro_analyzer.data_dir, "canary_history.json")
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            records = history.get('records', [])[-5:]  # 최근 5 개만
            consecutive = history.get('consecutive_days', 0)
            
            if records or consecutive > 0:
                html += f"""
                <div style='background:#f8f9fa;border-radius:8px;padding:15px'>
                    <div style='font-weight:bold;color:#2c3e50;margin-bottom:10px'>📅 신호 변화 이력</div>
                """
                
                if consecutive > 0:
                    html += f"""<div style='font-size:0.9em;color:#666;margin-bottom:10px'>
                        현재 {mode} 모드 지속: <b style='color:{mode_color}'>{consecutive}일째</b>
                    </div>"""
                
                if records:
                    html += "<div style='font-size:0.85em;line-height:1.8'>"
                    for rec in reversed(records):
                        from_color = "#2ecc71" if rec['from'] == '공격' else ("#f1c40f" if rec['from'] == '주의' else "#e74c3c")
                        to_color = "#2ecc71" if rec['to'] == '공격' else ("#f1c40f" if rec['to'] == '주의' else "#e74c3c")
                        html += f"""<div>
                            {rec['date']} 
                            <b style='color:{from_color}'>{rec['from']}</b> → 
                            <b style='color:{to_color}'>{rec['to']}</b>
                            <span style='color:#888'>({rec.get('reason','')}</span>)
                        </div>"""
                    html += "</div>"
                
                html += "</div>"
        except:
            pass
        
        html += "</div>"
        return html
    
    def _build_sector_bar(self, current_alloc, forced_cash_ratio=0.0):
        """
        P2-1: 섹터 비중 바 HTML 생성 (현금 + 강제 현금 비중 표시)
        
        Args:
            current_alloc: 섹터별 실제 비중 dict ({'Technology': 0.25, '현금': 0.30, ...})
            forced_cash_ratio: 시장 모드에 따른 강제 현금 비중 (0.0~1.0)
        """
        # 섹터 색상 (현금은 별도 처리)
        sector_colors = {
            'Technology': '#3498db',
            'Semiconductors': '#e74c3c',
            'Financial Services': '#f1c40f',
            'Energy': '#2ecc71',
            'Healthcare': '#9b59b6',
            'Consumer Cyclical': '#34495e',
            'Industrials': '#1abc9c',
            'Consumer Defensive': '#e67e22',
            'Basic Materials': '#9b59b6',
            'Utilities': '#16a085',
            'Real Estate': '#d35400',
            'Communication Services': '#8e44ad',
        }
        default_colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6', '#34495e', '#1abc9c', '#e67e22']
        
        # 1. 실제 현금 비중과 방어 현금 분리
        actual_cash = current_alloc.get('현금', 0.0)
        defensive_cash = forced_cash_ratio  # 방어모드에서 잠긴 현금 (투자 불가)
        available_cash = max(0, actual_cash - defensive_cash)  # 투자 가능 현금
        
        sector_bar_html = "<div style='margin-bottom:20px;background:#fff;padding:15px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.05)'>"
        sector_bar_html += "<h3 style='margin:0 0 10px 0;display:flex;justify-content:space-between;align-items:center'>"
        sector_bar_html += "<span>내 계좌 섹터 비중</span>"
        # 현금 상태 표시
        if defensive_cash > 0:
            sector_bar_html += f"<span style='font-size:0.85em;color:#2c3e50'>💰 현금 {actual_cash*100:.0f}% (투자가능 {available_cash*100:.0f}%)</span>"
        else:
            sector_bar_html += f"<span style='font-size:0.85em;color:#2c3e50'>💰 현금 {actual_cash*100:.0f}%</span>"
        sector_bar_html += "</h3>"
        
        # 2. 바 차트 렌더링 - 주식 섹터 먼저 (왼쪽), 현금은 맨 오른쪽
        sector_bar_html += "<div style='display:flex;height:35px;border-radius:5px;overflow:hidden;margin-top:10px'>"
        
        # 주식 섹터 렌더링 (현금 제외)
        color_idx = 0
        for sec, ratio in sorted(current_alloc.items(), key=lambda x: x[1], reverse=True):
            if ratio < 0.01: continue
            if sec == '현금': continue  # 현금은 맨 오른쪽에 별도 처리
            
            pct = ratio * 100
            kor_name = SECTOR_KOR.get(sec, sec)
            color = sector_colors.get(sec, default_colors[color_idx % len(default_colors)])
            
            sector_bar_html += f"<div style='width:{pct}%;background:{color};color:white;text-align:center;font-size:0.85em;line-height:35px;font-weight:bold' title='{kor_name}: {pct:.1f}%'>{kor_name} {pct:.0f}%</div>"
            color_idx += 1
        
        # 3. 현금 표시 (하나로 통합) - 맨 오른쪽
        if actual_cash > 0.01:
            if defensive_cash > 0.01 and actual_cash > defensive_cash:
                # 현금이 방어비중보다 많음: 투자가능 + 방어현금
                sector_bar_html += f"<div style='width:{available_cash*100}%;background:#bdc3c7;color:#2c3e50;text-align:center;font-size:0.85em;line-height:35px;font-weight:bold' title='투자 가능 현금: {available_cash*100:.1f}%'>💰 투자가능 {available_cash*100:.0f}%</div>"
                sector_bar_html += f"<div style='width:{defensive_cash*100}%;background:linear-gradient(135deg,#7f8c8d,#95a5a6);color:white;text-align:center;font-size:0.85em;line-height:35px;font-weight:bold;border-left:2px solid #fff' title='방어 현금 (투자불가): {defensive_cash*100:.1f}%'>🛡️ {defensive_cash*100:.0f}%</div>"
            elif defensive_cash > 0.01:
                # 현금이 모두 방어비중 (투자불가)
                sector_bar_html += f"<div style='width:{actual_cash*100}%;background:linear-gradient(135deg,#7f8c8d,#95a5a6);color:white;text-align:center;font-size:0.85em;line-height:35px;font-weight:bold;border-left:2px solid #fff' title='방어 현금 (투자불가): {actual_cash*100:.1f}%'>🛡️ {actual_cash*100:.0f}%</div>"
            else:
                # 방어모드 아님: 일반 현금
                sector_bar_html += f"<div style='width:{actual_cash*100}%;background:#95a5a6;color:white;text-align:center;font-size:0.85em;line-height:35px;font-weight:bold' title='현금: {actual_cash*100:.1f}%'>💰 {actual_cash*100:.0f}%</div>"
        
        sector_bar_html += "</div>"
        
        # 4. 범례 (하단)
        sector_bar_html += "<div style='margin-top:10px;font-size:0.8em;color:#666;display:flex;gap:15px;flex-wrap:wrap'>"
        sector_bar_html += "<span>📊 유색: 주식 섹터</span>"
        if defensive_cash > 0.01:
            sector_bar_html += "<span>💰 밝회색: 투자가능현금</span>"
            sector_bar_html += "<span>🛡️ 진회색: 방어현금 (투자불가)</span>"
        elif actual_cash > 0.01:
            sector_bar_html += "<span>⚪ 회색: 현금</span>"
        sector_bar_html += "</div>"
        
        sector_bar_html += "</div>"
        return sector_bar_html
    
    def generate_html_report(self, macro_data, turtle_data, expert_data, rs_data, u_map):
        """P2-1: 리포트 생성 (모듈화)"""
        # 1. 보유 종목 데이터 구축
        turtle_map = {r['ticker']: r for r in turtle_data['results']}
        expert_map = {r['ticker']: r for r in expert_data['results']}
        my_holdings_data = self._build_holdings_data(turtle_map, expert_map)
        
        # 2. 리밸런싱 매니저 초기화
        self.rebalance_manager = RebalanceManager(my_holdings_data, self.macro_analyzer, self.cash_balance, self.usd_krw, self.fetcher)
        current_alloc = self.rebalance_manager.get_current_sector_allocation(self.total_account_value)
        
        # 3. 섹터 바 생성 (강제 현금 비중 전달)
        forced_cash_ratio = self.macro_analyzer.cash_ratio  # 시장 모드에 따른 권장 현금 비중
        sector_bar_html = self._build_sector_bar(current_alloc, forced_cash_ratio)
        
        # 4. 리밸런싱 계획
        sells, scenarios = self.rebalance_manager.generate_plan(expert_data['results'], turtle_data['results'], rs_data['results'], self.total_account_value)

        logging.info("[8/8] 최종 리포트 생성 중...")
        current_stock_val = sum([p['current_price']*p['qty']*self.usd_krw for p in my_holdings_data])
        total_asset = current_stock_val + self.cash_balance
        market_status = self.macro_analyzer.regime
        market_score = self.macro_analyzer.market_score
        cash_ratio = int(self.macro_analyzer.cash_ratio * 100)
        ind = self.macro_analyzer.indicators
        ind_summary = f"VIX: {ind.get('VIX',0)} | 10 년물：{ind.get('10Y_Yield',0)}% | 유가: ${ind.get('Oil',0)}"
        if self.macro_analyzer.warnings: ind_summary += f"<br><span style='color:#e74c3c;font-size:0.9em'>⚠️ {', '.join(self.macro_analyzer.warnings)}</span>"
        status_color = "#2ecc71" 
        if "Caution" in market_status: status_color = "#f1c40f"
        elif "Panic" in market_status or "Correction" in market_status: status_color = "#e74c3c"
        
        # V24.9: 카나리아 신호등 HTML
        canary_html = self._build_canary_html()
        
        style = "body{background:#f0f2f5;font-family:'Malgun Gothic',sans-serif;padding:20px;color:#333}.container{max-width:1400px;margin:auto}h1{text-align:center;color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:15px}.market-alert{background:" + status_color + ";color:white;padding:15px;border-radius:8px;margin-bottom:20px;text-align:center;font-weight:700;font-size:1.2em;box-shadow:0 4px 6px rgba(0,0,0,.1)}.asset-dashboard{display:flex;justify-content:space-around;background:linear-gradient(135deg,#2c3e50,#34495e);color:#fff;padding:15px;border-radius:12px;margin-bottom:25px;box-shadow:0 4px 6px rgba(0,0,0,.1)}.asset-item{text-align:center}.asset-label{font-size:.9em;opacity:.8;margin-bottom:5px}.asset-value{font-size:1.4em;font-weight:700}.asset-sub{font-size:.8em;color:#f1c40f}.card{background:#fff;padding:25px;border-radius:12px;box-shadow:0 3px 10px rgba(0,0,0,.08);margin-bottom:25px}h2{border-left:6px solid #2c3e50;padding-left:15px;color:#2c3e50;margin-top:0}.folder-header{font-size:1.1em;font-weight:700;color:#fff;margin-top:25px;margin-bottom:0;display:flex;align-items:center;background:#34495e;padding:10px 15px;border-radius:8px 8px 0 0}.folder-info{margin-left:auto;font-size:.9em;opacity:.9}.summary-table{width:100%;border-collapse:collapse;margin-bottom:20px;background:#eaf2f8;border:1px solid #d6eaf8;border-radius:8px;overflow:hidden}.summary-table th{background:#2980b9;color:#fff;padding:10px;text-align:center}.summary-table td{padding:8px;text-align:center;border-bottom:1px solid #ddd}.detail-table{width:100%;border-collapse:collapse;font-size:14px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1);border:1px solid #eee}.detail-table th{background:#ecf0f1;color:#2c3e50;padding:12px;text-align:center;font-weight:700;border-bottom:2px solid #bdc3c7}.detail-table td{border-bottom:1px solid #eee;padding:12px;text-align:center;color:#333}.detail-table tr:nth-child(even){background-color:#fcfcfc}.up{color:#e74c3c;font-weight:700}.down{color:#3498db;font-weight:700}.tag{padding:5px 10px;border-radius:15px;color:#fff;font-size:12px;font-weight:700}.buy{background:#e74c3c}.section-intersection{border-left:6px solid #e67e22}.strategy-tabs{display:flex;margin-bottom:20px}.strategy-tab{padding:12px 25px;cursor:pointer;background:#ecf0f1;border:1px solid #bdc3c7;font-weight:700;color:#7f8c8d}.strategy-tab.active{background:#3498db;color:#fff;border-color:#3498db}.strategy-content{display:none}.strategy-content.active{display:block}.volume-spike{background-color:#fffde7!important;border-left:4px solid #fbc02d}.vol-badge{background:#fbc02d;color:#333;font-size:.8em;padding:2px 6px;border-radius:4px;font-weight:700;margin-left:5px}.rs-detail{font-size:.85em;color:#666;margin-left:5px}.fund-cell{font-size:.85em;color:#555;line-height:1.4;text-align:left;background:#fafafa;padding:5px 10px;border-radius:4px}"
        
        # V25.0: 조기경보 시스템 HTML
        early_warning_html = ""
        if sells:
            urgency_sells = [s for s in sells if s.get('urgency', 0) > 0]
            if urgency_sells:
                early_warning_html = "<div class='card' style='border-left: 6px solid #e74c3c;'><h2>🚨 V25.0 조기경보 시스템</h2><table class='detail-table'><tr><th>종목</th><th>긴급도</th><th>모멘텀추세</th><th>교체후보</th></tr>"
                for s in urgency_sells:
                    urgency = s.get('urgency', 0)
                    urgency_color = "#e74c3c" if urgency >= 70 else ("#f1c40f" if urgency >= 40 else "#2ecc71")
                    urgency_icon = "🔴" if urgency >= 70 else ("🟡" if urgency >= 40 else "🟢")
                    details = s.get('urgency_details', {})
                    momentum_trend = details.get('momentum_trend', [])
                    trend_str = " → ".join(momentum_trend[-4:]) if momentum_trend else "N/A"
                    candidates = details.get('replace_candidates', [])
                    candidates_str = ", ".join(candidates[:3]) if candidates else "-"
                    early_warning_html += f"<tr><td><b>{s['ticker']}</b></td><td style='color:{urgency_color};font-weight:bold;font-size:1.2em'>{urgency_icon} {urgency}점</td><td style='font-size:0.85em'>{trend_str}</td><td style='font-size:0.85em'>{candidates_str}</td></tr>"
                early_warning_html += "</table><p style='font-size:0.85em;color:#666;margin-top:10px'>긴급도 기준: 🟢 0-39 유지 | 🟡 40-69 관찰 (공격적: 교체 검토) | 🔴 70+ 즉시 교체 검토</p></div>"
        
        rebalance_html = f"<div class='card' style='border-left: 6px solid #9b59b6;'><h2>포트폴리오 리밸런싱 제안 (V24.7 Korean Patch)</h2><p>※ <b>최소 1,000 만원 집중 투자</b> + 섹터 쿼터 제한 (반도체 30% 등) 적용</p>"
        if sells:
            rebalance_html += "<h3>매도 권장 (Weak)</h3><table class='detail-table'><tr><th>종목</th><th>수량</th><th>예상 금액</th><th>매도 사유</th></tr>"
            for s in sells: rebalance_html += f"<tr><td><b>{s['ticker']}</b></td><td>{s['qty']} 주</td><td>{s['est_value']/10000:,.0f} 만원</td><td style='color:#c0392b;font-weight:bold'>{s['reason']}</td></tr>"
            rebalance_html += "</table>"
        else: rebalance_html += "<p>✅ 매도 대상 종목이 없습니다.</p>"
        rebalance_html += "<h3 style='margin-top:20px'>시나리오별 매수 제안</h3><div style='display:flex;gap:10px;'>"
        for key, sc in scenarios.items():
            rebalance_html += f"<div style='flex:1;background:#f8f9fa;padding:10px;border-radius:5px;border:1px solid #ddd'><h4>{key}. {sc['name']}</h4><p>💰 현금 보유：{sc['cash_reserve']/10000:,.0f} 만원</p><ul style='padding-left:20px'>"
            if sc['buy_list']:
                for b in sc['buy_list']: rebalance_html += f"<li><b>{b['ticker']}</b>: {b['qty']} 주 ({b['amount']/10000:,.0f} 만원)</li>"
            else: rebalance_html += "<li>매수 추천 없음 (쿼터 초과 등)</li>"
            rebalance_html += "</ul></div>"
        rebalance_html += "</div></div>"

        m_rows = ""
        for asset in macro_data.values():
            pct = asset['pct']
            m_rows += f"<tr><td>{asset['name']}</td><td>{asset['price']:,.2f}</td><td class='{'up' if pct>0 else 'down'}'>{pct:+.2f}%</td><td>{asset['trend']}</td></tr>"

        my_rows = ""
        for p in my_holdings_data:
            orig_price = next(pos['price'] for pos in MY_POSITIONS if pos['ticker'] == p['ticker'])
            profit = (p['current_price'] / orig_price - 1) * 100
            diag = p['signal'].split('|')[0]
            if len(p['signal'].split('|')) > 1: diag += f" <br><span style='font-size:0.8em;color:red'>⚠️ {p['signal'].split('|')[1]}</span>"
            fund = self.fetcher.get_data(p['ticker']) or {}
            def fmt(val, suffix=''): return f"{val}{suffix}" if val else "-"
            fund_info = f"PBR: <b>{fmt(fund.get('pbr',0))}</b> | PER: <b>{fmt(fund.get('per',0))}</b><br>PSR: <b>{fmt(fund.get('psr',0))}</b> | 부채： <b>{fmt(fund.get('debt_ratio',0), '%')}</b>"
            my_rows += f"<tr><td style='text-align:left'><div style='font-weight:bold;font-size:1.1em'>{p['ticker']}</div></td><td>${orig_price:.2f}</td><td><div style='font-weight:bold'>${p['current_price']:.2f}</div><div style='font-size:0.9em;color:#555'>T:{p['trend_score']:.0f} | E:{p['expert_score']} | RS:{p['rs_score']:.0f}</div></td><td><div class='fund-cell'>{fund_info}</div></td><td class='{'up' if profit>0 else 'down'}'>{profit:.1f}%</td><td style='color:#c0392b;font-weight:bold;font-size:1.1em'>${p['stop_loss']:.2f}</td><td>{diag}</td></tr>"
        
        # P3-1: 손절가 설명을 트레일링 스탑으로 수정 (현재가 기준)
        stop_loss_description = "※ <b>📉 트레일링 스탑 (10 일 저점)</b>: 주가가 10 일 최저가 (Exit) 를 깨면 매도 (익절/청산 라인)<br>※ <b>🛑 손절가 (2N)</b>: 현재가 - (2 x ATR), 변동성 고려한 동적 손절가"

        intersection_rows = ""
        for t, e in expert_map.items():
            if t in turtle_map:
                tu = turtle_map[t]
                if '매수' in tu['signal']:
                    vol_cls = 'class="volume-spike"' if tu['vol_spike'] else ""
                    vol_txt = "<span class='vol-badge'>🔥Vol</span>" if tu['vol_spike'] else ""
                    rs_info = f"{e['rs_score']:.0f} <span class='rs-detail'>({e['rs_diff']:+.1f}%)</span>"
                    intersection_rows += f"<tr {vol_cls}><td><b>{t}</b>{vol_txt}</td><td>{e['universe']}</td><td>{e['sector']}</td><td><span class='tag buy'>{tu['signal']}</span></td><td>{tu['trend_score']:.0f}</td><td class='up'>{e['score']}</td><td>{rs_info}</td><td>${e['close']:.2f}</td><td>{tu['unit_qty']} 주</td></tr>"
        if not intersection_rows: intersection_rows = "<tr><td colspan='9'>조건을 만족하는 종목이 없습니다.</td></tr>"

        def get_stat_table(stats_dict):
            rows = ""
            sorted_keys = sorted(stats_dict.keys(), key=lambda x: CUSTOM_ORDER.index(x) if x in CUSTOM_ORDER else 99)
            for sec in sorted_keys:
                data = stats_dict[sec]
                ratio = (data['picked']/data['total']*100) if data['total']>0 else 0
                rows += f"<tr><td>{sec}</td><td>{data['total']} 개</td><td><b>{data['picked']} 개</b></td><td>{ratio:.1f}%</td></tr>"
            return f"<table class='summary-table'><tr><th>대상 그룹</th><th>전체 스캔</th><th>선정 개수</th><th>선정 비율</th></tr>{rows}</table>"

        def get_grouped_html(data_list, type_):
            if not data_list: return "<p>데이터 없음</p>"
            grouped = {}
            for x in data_list:
                univ = x.get('universe', '기타')
                if univ not in grouped: grouped[univ] = []
                grouped[univ].append(x)
            html_out = ""
            sorted_univs = sorted(grouped.keys(), key=lambda x: CUSTOM_ORDER.index(x) if x in CUSTOM_ORDER else 99)
            for univ in sorted_univs:
                items = grouped[univ]
                rows = ""
                key = 'trend_score' if type_=='turtle' else 'score' if type_=='expert' else 'rs_score'
                items.sort(key=lambda x: x.get(key, 0), reverse=True)
                for x in items:
                    if type_ == 'turtle' and '매수' not in x['signal']: continue
                    if type_ == 'expert' and x['score'] < EXPERT_CUTOFF: continue
                    if type_ == 'rs' and x['rs_score'] < 70: continue
                    if type_ == 'turtle':
                        vol_cls = 'class="volume-spike"' if x['vol_spike'] else ""
                        vol_txt = "<span class='vol-badge'>🔥Vol</span>" if x['vol_spike'] else ""
                        rows += f"<tr {vol_cls}><td><b>{x['ticker']}</b>{vol_txt}</td><td>{x['signal']}</td><td>{x['trend_score']:.0f}</td><td>${x['price']:.2f}</td><td>{x['unit_qty']} 주</td></tr>"
                    elif type_ == 'expert':
                        rs_detail = f"{x['rs_score']:.0f} <span class='rs-detail'>({x['rs_diff']:+.1f}%)</span>"
                        rows += f"<tr><td><b>{x['ticker']}</b></td><td>{x['score']}</td><td>{rs_detail}</td><td>{x['roe']}%</td><td>{x['growth']}%</td><td>{x['div']}%</td><td>{x['pbr']}</td><td>{x['debt']}%</td><td>${x['close']:.2f}</td></tr>"
                    elif type_ == 'rs':
                        vol_str = f"{x['vol_ratio']}x" 
                        vol_style = ""
                        if x['vol_ratio'] >= 2.5: vol_style = "font-weight:bold;color:#e74c3c"
                        elif x['vol_ratio'] >= 1.5: vol_style = "font-weight:bold;color:#e67e22"
                        else: vol_style = "color:#2ecc71"
                        mom_13612 = x.get('momentum_13612', 0)
                        mom_color = "#e74c3c" if mom_13612 < 0 else "#2ecc71"
                        rows += f"<tr><td><b>{x['ticker']}</b></td><td class='up'>{x['rs_score']:.1f}</td><td>{x['stock_ret']:.1f}%</td><td>{x['diff']:+.1f}%</td><td style='{vol_style}'>{vol_str}</td><td style='color:{mom_color};font-weight:bold'>{mom_13612:+.1f}%</td><td>${x['close']:.2f}</td></tr>"
                if rows:
                    if type_ == 'turtle': headers = "<th>종목</th><th>신호</th><th>추세</th><th>가격</th><th>수량</th>"
                    elif type_ == 'expert': headers = "<th>종목</th><th>점수</th><th>RS( 시장대비)</th><th>ROE</th><th>성장률</th><th>배당</th><th>PBR</th><th>부채비율</th><th>가격</th>"
                    else: headers = "<th>종목</th><th>RS 점수</th><th>3 개월 수익</th><th>시장대비</th><th>거래량</th><th>13612W</th><th>가격</th>"
                    html_out += f"<div class='folder-header'><span>{univ}</span><span class='folder-info'>{len(items)} 개</span></div><table class='detail-table'><tr>{headers}</tr>{rows}</table>"
            return html_out

        turtle_stat = get_stat_table(turtle_data['stats'])
        expert_stat = get_stat_table(expert_data['stats'])
        rs_stat = get_stat_table(rs_data['stats'])
        turtle_html = get_grouped_html(turtle_data['results'], 'turtle')
        expert_html = get_grouped_html(expert_data['results'], 'expert')
        rs_html = get_grouped_html(rs_data['results'], 'rs')

        # 현재 날짜/시간 생성
        now = datetime.now()
        report_date = now.strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>{style}</style></head><body><div class="container"><h1>통합 퀀트 전략 보고서 V24.9 (13612W + Canary)</h1><div style="text-align:right;font-size:0.85em;color:#666;margin-bottom:10px">리포트 생성일: {report_date}</div><div class="market-alert">{market_status} (Score: {market_score})<br><span style="font-size:0.8em; font-weight:normal;">{ind_summary}</span><br><span style="font-size:0.8em; font-weight:normal;">권장 현금 비중：{cash_ratio}% 이상</span></div>{canary_html}{sector_bar_html}{rebalance_html}<div class="asset-dashboard"><div class="asset-item"><div class="asset-label">실시간 총 자산</div><div class="asset-value">{total_asset/10000:,.0f} 만원</div></div><div class="asset-item"><div class="asset-label">주식 평가금</div><div class="asset-value">{current_stock_val/10000:,.0f} 만원</div></div><div class="asset-item"><div class="asset-label">가용 현금</div><div class="asset-value">{self.cash_balance/10000:,.0f} 만원</div></div><div class="asset-item"><div class="asset-label">적용 환율</div><div class="asset-value">{self.usd_krw:,.2f} 원</div></div></div><div class="card"><h2>글로벌 시장 현황</h2><table class='detail-table'><tr><th>지수</th><th>현재가</th><th>등락률</th><th>추세</th></tr>{m_rows}</table></div><div class="card"><h2>[MY] 보유 종목 진단 (매도 기준)</h2><p>※ <b>트레일링 스탑 (10 일 저점)</b>: 주가가 10 일 최저가 (Exit) 를 깨면 매도 (익절/청산 라인)<br>※ <b>손절가 (2N)</b>: 현재가 - (2 x ATR), 변동성 고려한 동적 손절가 (트레일링)</p><table class='detail-table'><tr><th>종목 (그룹/산업)</th><th>매수가</th><th>현재가 / 점수</th><th>펀더멘털 지표 (PBR/PER/PSR/ 부채)</th><th>수익률</th><th>손절가 (2N)</th><th>진단</th></tr>{my_rows}</table></div><div class="card section-intersection"><h2>[교집합] 강력 추천 (Turtle + Expert)</h2><p>※ <b>Turtle 매수신호</b> + <b>Expert 70 점 이상</b>을 모두 충족하는 종목입니다.</p><table class='detail-table'><tr><th>종목</th><th>그룹</th><th>산업</th><th>신호</th><th>T 점수</th><th>E 점수</th><th>RS 점수 (초과%)</th><th>가격</th><th>수량</th></tr>{intersection_rows}</table></div><div class="strategy-tabs"><div class="strategy-tab active" onclick="showStrategy('turtle')">Turtle (돌파)</div><div class="strategy-tab" onclick="showStrategy('expert')">Expert (우량주)</div><div class="strategy-tab" onclick="showStrategy('rs')">RS (주도주)</div></div><div class="card strategy-content active" id="strategy-turtle"><h2>Turtle 돌파 매수</h2>{turtle_stat}{turtle_html}</div><div class="card strategy-content" id="strategy-expert"><h2>Expert 우량주 선정 (섹터별 가중치 적용)</h2>{expert_stat}{expert_html}</div><div class="card strategy-content" id="strategy-rs"><h2>RS 주도주 (시장 대비 강세)</h2>{rs_stat}{rs_html}</div><script>function showStrategy(type) {{document.querySelectorAll('.strategy-tab').forEach(t => t.classList.remove('active'));event.target.classList.add('active');document.querySelectorAll('.strategy-content').forEach(c => c.classList.remove('active'));document.getElementById('strategy-' + type).classList.add('active');}}</script></div></body></html>"""
        
        # P7: 날짜별 리포트 보관
        today_str = datetime.now().strftime('%Y-%m-%d')
        daily_report_path = os.path.join(REPORT_DIR, f"report_{today_str}.html")
        
        # index.html (최신)
        with open(os.path.join(REPORT_DIR, "index.html"), "w", encoding='utf-8') as f: f.write(html)
        
        # 날짜별 리포트 (보관용)
        with open(daily_report_path, "w", encoding='utf-8') as f: f.write(html)
        logging.info(f"📅 일별 리포트 저장: {daily_report_path}")
        
        webbrowser.open(os.path.join(REPORT_DIR, "index.html"))

    def push_to_github(self):
        logging.info("[9/9] GitHub 로 업로드 중...")
        if not os.path.exists(GITHUB_DIR):
            logging.warning("GitHub 디렉토리 없음, 업로드 건너뜀")
            return
        src = os.path.join(REPORT_DIR, "index.html")
        dst = os.path.join(GITHUB_DIR, "index.html")
        shutil.copy(src, dst)
        
        git_commands = [
            (["git", "add", "."], "git add"),
            (["git", "commit", "-m", f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"], "git commit"),
            (["git", "push"], "git push")
        ]
        
        for cmd, name in git_commands:
            try:
                result = subprocess.run(cmd, cwd=GITHUB_DIR, check=True, capture_output=True, text=True, timeout=30)
                logging.info(f"✅ {name} 성공")
            except subprocess.CalledProcessError as e:
                logging.error(f"❌ {name} 실패：{e.stderr.strip() if e.stderr else str(e)}")
                return
            except subprocess.TimeoutExpired:
                logging.error(f"❌ {name} 타임아웃 (30 초)")
                return
            except Exception as e:
                logging.error(f"❌ {name} 오류：{e}")
                return
        
        logging.info("✅ GitHub 업로드 완료!")

    def run(self):
        logging.info("=" * 70)
        logging.info("통합 퀀트 시스템 V24.7 실행 시작")
        logging.info("=" * 70)
        self.get_exchange_rate()
        self.macro_analyzer.run_deep_dive() 
        
        # 1. 먼저 데이터 다운로드 (현재가 계산을 위해 필수!)
        macro = self.get_macro_data()
        all_tickers, nasdaq, sp500, sox = self.fetch_dynamic_tickers()
        all_data = self.data_downloader.download_all(all_tickers)
        logging.info(f"[3/7] 보유 종목 ({len(MY_TICKERS)}개) 정밀 분석 중...")
        for t in MY_TICKERS: self.fetcher.fetch_with_retry(t, priority=True)
        
        # 2. 현재가 기준 주식 평가금 계산
        current_stock_eval_krw = 0
        for p in MY_POSITIONS:
            t = p['ticker']
            try:
                df = self._extract_ticker_data(all_data, t)
                if not df.empty:
                    curr_price = df['Close'].iloc[-1]
                    if hasattr(curr_price, 'item'): curr_price = curr_price.item()
                    current_stock_eval_krw += curr_price * p['qty'] * self.usd_krw
            except Exception as e:
                logging.warning(f"보유 종목 평가 오류 ({t}): {e}")
        
        # 3. 현금 잔고 = 총자산 (사용자 입력) - 현재주식평가금 (음수 방지)
        self.cash_balance = max(0, self.total_account_value - current_stock_eval_krw)
        
        # P1-4: cash_balance=0 시 경고 로그
        if self.cash_balance == 0 and current_stock_eval_krw > self.total_account_value:
            logging.warning(f"⚠️ 주식 평가금 ({current_stock_eval_krw/10000:,.0f} 만원) 이 총자산 ({self.total_account_value/10000:,.0f} 만원) 보다 큽니다. 매수 제안이 제한될 수 있습니다.")
        
        logging.info(f"환율：{self.usd_krw:,.2f} 원 | 총 자산：{self.total_account_value/10000:,.0f} 만원 (고정)")
        logging.info(f"주식 평가금：{current_stock_eval_krw/10000:,.0f} 만원 | 현금：{self.cash_balance/10000:,.0f} 만원")
        logging.info("-" * 70)
        
        # V24.9: 벤치마크 모멘텀 계산 (13612W 기준)
        try:
            spy_df = yf.download('^GSPC', period='1y', progress=False)
            if not spy_df.empty and len(spy_df) >= 252:
                close_prices = spy_df['Close'].dropna()  # NaN 제거
                if len(close_prices) >= 252:
                    self.benchmark_momentum = calculate_momentum_13612(close_prices)
                    logging.info(f"V24.9: S&P500 13612W 모멘텀 = {self.benchmark_momentum:.1f}%")
                else:
                    logging.warning(f"S&P500 데이터 부족 ({len(close_prices)}일/252 일)")
                    self.benchmark_momentum = 0
            else:
                logging.warning("S&P500 데이터 없음")
                self.benchmark_momentum = 0
        except Exception as e:
            self.benchmark_momentum = 0
            logging.warning(f"벤치마크 모멘텀 계산 실패: {e}")
        
        # 4. Universe map 생성 (RS 계산 전에 필요)
        u_map = {}
        for t in sp500: u_map[t] = 'S&P500'
        for t in nasdaq: u_map[t] = '나스닥 100'
        for t in sox: u_map[t] = '반도체 (SOX)'
        for t in MY_TICKERS:
            if t not in u_map:
                if t in sox: u_map[t] = '반도체 (SOX)'
                elif t in nasdaq: u_map[t] = '나스닥 100'
                elif t in sp500: u_map[t] = 'S&P500'
                else: u_map[t] = '기타 (ETF/Other)'
        
        # 러셀 2000 추가 (나머지 종목)
        russell_count = 0
        for t in all_tickers:
            if t not in u_map and t not in sp500 and t not in nasdaq and t not in sox:
                u_map[t] = '러셀 2000'
                russell_count += 1
        
        logging.info(f"  • Universe map: {len(u_map)}개 (러셀 2000: {russell_count}개)")
        
        # 5. RS 백분위 랭킹 계산 (P3-4: 산업 섹터 내/전체 혼합)
        blended_rs_map, overall_rs_map, returns_map = self.calculate_rs_scores_bulk(all_tickers, all_data, self.fetcher)
        logging.info(f"[RS] 백분위 랭킹 계산 완료 ({len(blended_rs_map)}개 종목, 산업섹터혼합)")
        
        # 6. 전략 실행 (혼합 RS map 전달)
        turtle_res = self.run_turtle_strategy(all_tickers, u_map, all_data, blended_rs_map)
        expert_res = self.run_expert_strategy(all_tickers, u_map, all_data, blended_rs_map)
        rs_res = self.run_rs_strategy(all_tickers, u_map, all_data, blended_rs_map)
        self.generate_html_report(macro, turtle_res, expert_res, rs_res, u_map)
        self.push_to_github() 
        logging.info("🎉 모든 작업 완료!")

if __name__ == "__main__":
    # V24.9: config.json 자동 동기화 (9 번 항목)
    config_changed = False
    
    # 1. 파일 없으면 생성
    if not os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            logging.info(f"새 설정 파일 생성: {CONFIG_FILE}")
            config_changed = True
        except Exception as e:
            logging.error(f"설정 파일 생성 실패: {e}")
    else:
        # 2. 파일 있으면 로드 + DEFAULT_CONFIG 업데이트
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            # P0-3: config.json 우선 (사용자 설정이 DEFAULT 보다 우선)
            for key, value in loaded.items():
                if key in DEFAULT_CONFIG and isinstance(value, dict):
                    # 하위 키도 사용자 설정 우선
                    for subkey, subvalue in value.items():
                        DEFAULT_CONFIG[key][subkey] = subvalue  # 사용자 값으로 덮어쓰기
                else:
                    DEFAULT_CONFIG[key] = value  # 새로운 키 추가
            
            # V24.9: 새로운 설정 항목 자동 추가
            if 'half_kelly_ratio' not in DEFAULT_CONFIG['account']:
                DEFAULT_CONFIG['account']['half_kelly_ratio'] = 0.08
                config_changed = True
            
            # 변경사항이 있으면 config.json 업데이트
            if config_changed:
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
                logging.info(f"설정 파일 동기화 완료: {CONFIG_FILE}")
            else:
                logging.info(f"설정 파일 로드 완료: {CONFIG_FILE}")
                
        except Exception as e:
            logging.error(f"설정 파일 로드 실패: {e}")
    
    # 3. 상수 재할당 (config.json 반영)
    TOTAL_ACCOUNT_VALUE_KRW = DEFAULT_CONFIG['account']['total_krw']
    RISK_RATIO = DEFAULT_CONFIG['account']['risk_ratio']
    MAX_ALLOC_PER_STOCK = DEFAULT_CONFIG['account']['max_alloc_per_stock']
    EXPERT_CUTOFF = DEFAULT_CONFIG['account']['expert_cutoff']
    MIN_INVEST_PER_STOCK = DEFAULT_CONFIG['account']['min_invest_per_stock']
    MY_POSITIONS = DEFAULT_CONFIG['positions']
    MY_TICKERS = [p['ticker'] for p in MY_POSITIONS]
    TICKERS_TO_CHART = MY_TICKERS
    CHUNK_SIZE = DEFAULT_CONFIG['data']['chunk_size']
    DELAY_BETWEEN_CHUNKS = DEFAULT_CONFIG['data']['delay_between_chunks']
    DOWNLOAD_TIMEOUT = DEFAULT_CONFIG['data']['download_timeout']
    DOWNLOAD_PERIOD = DEFAULT_CONFIG['data']['download_period']
    MACRO_PERIOD = DEFAULT_CONFIG['data']['macro_period']
    RS_PERIOD_DAYS = DEFAULT_CONFIG['data']['rs_period_days']
    VOLUME_AVG_DAYS = DEFAULT_CONFIG['data']['volume_avg_days']
    DEFAULT_EXCHANGE_RATE = DEFAULT_CONFIG['exchange']['default_rate']
    EXCHANGE_CACHE_MAX_AGE_HOURS = DEFAULT_CONFIG['exchange']['cache_max_age_hours']
    LOG_MAX_BYTES = DEFAULT_CONFIG['cache']['log_max_bytes']
    LOG_BACKUP_COUNT = DEFAULT_CONFIG['cache']['log_backup_count']
    
    logging.info(f"상수 재할당 완료 (총자산={TOTAL_ACCOUNT_VALUE_KRW:,.0f}, ExpertCutoff={EXPERT_CUTOFF}, 보유종목={len(MY_TICKERS)}개)")
    
    # 4. 실행
    IntegratedQuantSystem(TOTAL_ACCOUNT_VALUE_KRW).run()
