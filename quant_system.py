"""
통합 퀀트 시스템 V24.9
"""
import os
import sys
import logging
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

from core.data_loader import EnhancedDataDownloader
from core.fundamentals import SmartFundamentalsFetcher
from core.macro import MacroAnalyzer
from core.rebalance import RebalanceManager
from core.strategies.turtle import TurtleStrategy
from core.strategies.expert import ExpertStrategy
from core.strategies.rs import calculate_momentum_13612
from core.universe_expander import get_soxx_tickers, get_all_extra_tickers
from utils.sector_mapper import normalize_sector, get_sector_limit, get_sector_config, SECTOR_SCORING
from reports.html_builder import HTMLReportBuilder

# 로깅 설정 (콘솔 출력 + 파일)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러 (한글 인코딩 설정)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.stream = open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    log_file = r"C:\Quant\system.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


class IntegratedQuantSystem:
    """통합 퀀트 시스템 V24.9"""
    
    def __init__(self, config, as_of_date=None):
        """
        Args:
            config: 설정 객체
            as_of_date: 백테스트 기준일 (None 이면 오늘)
        """
        self.config = config
        self.as_of_date = as_of_date  # 백테스트용 기준일
        
        # Config 객체에서 실제 값 추출 (SimpleNamespace 또는 dict)
        if hasattr(config, 'account'):
            acc = config.account
            if isinstance(acc, dict):
                self.total_account_value = acc.get('total_krw', 76826207)
            else:
                self.total_account_value = getattr(acc, 'total_krw', 76826207)
        else:
            self.total_account_value = 76826207
        
        # 데이터 디렉토리 설정
        self.data_dir = config.dirs.data if hasattr(config, 'dirs') else r"C:\Quant\Data"
        self.reports_dir = config.dirs.reports if hasattr(config, 'dirs') else r"C:\Quant\Reports"
        
        # 하위 모듈 초기화
        self.data_downloader = EnhancedDataDownloader(config)
        self.fetcher = SmartFundamentalsFetcher(config)
        self.macro_analyzer = MacroAnalyzer(data_dir=self.data_dir, as_of_date=self.as_of_date)  # as_of_date 전달
        self.turtle_strategy = TurtleStrategy(config)
        self.expert_strategy = ExpertStrategy(config, type('obj', (object,), {
            'normalize_sector': normalize_sector,
            'get_sector_config': get_sector_config,
            'get_sector_limit': get_sector_limit
        })())
        
        self.rebalance_manager = None
        self.usd_krw = config.exchange.default_rate if hasattr(config, 'exchange') else 1450.0
        self.cash_balance = 0
        self.benchmark_returns = {}
        # SOX 티커는 fetch_dynamic_tickers 에서 자동 수집
        self.sox_tickers = []
        # 유니버스 확장 설정 (config 에서 읽음)
        self.expand_universe = getattr(config, 'expand_universe', False)
        self.include_russell = getattr(config, 'include_russell', True)  # 러셀 2000 포함
        self.include_thematic = getattr(config, 'include_thematic', True)  # 테마 종목 포함
    
    def get_exchange_rate(self):
        """환율 조회 (실제 구현)"""
        cache_file = os.path.join(self.data_dir, "Cache", "exchange_rate.json")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        try:
            df = yf.download("USDKRW=X", period="5d", progress=False)
            if not df.empty:
                rate = df['Close'].iloc[-1]
                if hasattr(rate, 'item'): rate = rate.item()
                self.usd_krw = round(rate, 2)
                import json
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'rate': self.usd_krw, 'ts': datetime.now().timestamp()}, f)
                logger.info(f"환율: {self.usd_krw:,.2f} 원 (실시간)")
                return
        except Exception as e:
            logger.warning(f"환율 조회 실패: {e}")
        
        # 캐시에서 로드
        try:
            if os.path.exists(cache_file):
                import json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                self.usd_krw = cached['rate']
                logger.info(f"환율: {self.usd_krw:,.2f} 원 (캐시)")
                return
        except:
            pass
        
        # 기본값
        self.usd_krw = 1450.0
        logger.info(f"환율: {self.usd_krw:,.2f} 원 (기본값)")
    
    def fetch_dynamic_tickers(self):
        """유니버스 최신화 (SOXX 자동 수집 + 확장 유니버스 지원)"""
        logger.info("[2/7] 유니버스 최신화...")
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # S&P 500
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            html = requests.get(url, headers=headers, timeout=10).text
            sp500 = pd.read_html(html)[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
            logger.info(f"  • S&P 500: {len(sp500)}개")
        except:
            sp500 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
            logger.warning("  ⚠️ S&P 500 스크래핑 실패 - 폴백 사용")
        
        # Nasdaq 100
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            html = requests.get(url, headers=headers, timeout=10).text
            dfs = pd.read_html(html)
            nasdaq = []
            for df in dfs:
                if 'Ticker' in df.columns:
                    nasdaq = df['Ticker'].tolist()
                    break
            logger.info(f"  • Nasdaq 100: {len(nasdaq)}개")
        except:
            nasdaq = ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMZN', 'META', 'TSLA']
            logger.warning("  ⚠️ Nasdaq 100 스크래핑 실패 - 폴백 사용")
        
        # SOX 반도체 (자동 수집 + 폴백)
        sox = get_soxx_tickers()
        self.sox_tickers = sox  # 인스턴스 변수 업데이트
        
        # 기본 유니버스
        all_tickers = sorted(list(set(nasdaq + sp500 + sox)))
        logger.info(f"  • 기본 유니버스: {len(all_tickers)}개")
        
        # 확장 유니버스 (선택사항)
        if self.expand_universe:
            logger.info("  🚀 확장 유니버스 활성화...")
            extra = get_all_extra_tickers(
                include_russell=self.include_russell,
                include_thematic=self.include_thematic
            )
            # 중복 제거
            extra_new = [t for t in extra if t not in all_tickers]
            all_tickers = sorted(list(set(all_tickers + extra_new)))
            logger.info(f"  • 확장 추가: {len(extra_new)}개")
            logger.info(f"  • 최종 유니버스: {len(all_tickers)}개")
        
        return all_tickers, nasdaq, sp500, sox
    
    def _extract_ticker_data(self, all_data, ticker):
        """데이터 추출 (MultiIndex + 단일 티커 처리)"""
        if all_data.empty:
            return pd.DataFrame()
        try:
            if isinstance(all_data.columns, pd.MultiIndex):
                if ticker in all_data.columns.get_level_values(0):
                    df = all_data[ticker].copy()
                    df.dropna(subset=['Close'], inplace=True)
                    return df
            else:
                # 단일 티커 다운로드 케이스 (청크 내 티커 1 개)
                if 'Close' in all_data.columns:
                    return all_data.copy()
        except Exception as e:
            logger.debug(f"데이터 추출 실패 ({ticker}): {e}")
        return pd.DataFrame()
    
    def _run_turtle(self, all_tickers, all_data, u_map):
        """Turtle 전략 실행 (실제 TurtleStrategy 사용)"""
        logger.info("[4/7] Turtle 전략 실행...")
        results = []
        sector_stats = {
            'S&P500': {'total': 0, 'picked': 0},
            '나스닥 100': {'total': 0, 'picked': 0},
            '반도체 (SOX)': {'total': 0, 'picked': 0},
            '기타 (확장)': {'total': 0, 'picked': 0}  # 확장 유니버스용
        }
        
        for t in all_tickers:
            univ = u_map.get(t, 'S&P500')
            if univ in sector_stats:
                sector_stats[univ]['total'] += 1
            
            try:
                df = self._extract_ticker_data(all_data, t)
                if df.empty or len(df) < 100:
                    continue
                
                curr_price = df['Close'].iloc[-1]
                if hasattr(curr_price, 'item'): curr_price = curr_price.item()
                
                ma200 = df['Close'].rolling(200).mean().iloc[-1]
                if hasattr(ma200, 'item'): ma200 = ma200.item()
                
                # TurtleStrategy.analyze() 사용
                signal_data = self.turtle_strategy.analyze(df, t, curr_price, ma200)
                
                if signal_data.get('signal') != '관망':
                    results.append({
                        'ticker': t,
                        'signal': signal_data.get('signal', '관망'),
                        'trend_score': signal_data.get('trend_score', 0),
                        'atr': signal_data.get('atr', 0),
                        'price': curr_price
                    })
                    if univ in sector_stats:
                        sector_stats[univ]['picked'] += 1
            except Exception as e:
                logger.debug(f"Turtle 분석 실패 ({t}): {e}")
        
        return {'results': results, 'stats': sector_stats}
    
    def _run_expert(self, all_tickers, all_data, u_map, spy_ret):
        """Expert 전략 실행 (실제 expert.py 사용)"""
        logger.info("[5/7] Expert 전략 실행...")
        results = []
        sector_stats = {'S&P500': {'total': 0, 'picked': 0}, '나스닥 100': {'total': 0, 'picked': 0}, '반도체 (SOX)': {'total': 0, 'picked': 0}}
        
        # RS 스코어 미리 계산 (spy_ret 은 run() 에서 한 번만 계산해서 전달)
        rs_scores = {}
        
        for t in all_tickers:
            univ = u_map.get(t, 'S&P500')
            if univ in sector_stats:
                sector_stats[univ]['total'] += 1
            
            try:
                df = self._extract_ticker_data(all_data, t)
                if df.empty or len(df) < 63:
                    continue
                
                # RS 계산 (.item() 변환으로 Series 방지)
                curr = df['Close'].iloc[-1]
                old = df['Close'].iloc[-63]
                if hasattr(curr, 'item'): curr = curr.item()
                if hasattr(old, 'item'): old = old.item()
                stock_ret = (curr / old - 1) * 100
                rs_score = 50 + ((stock_ret - spy_ret) * 2.0)
                rs_scores[t] = rs_score
                
                fund = self.fetcher.get_data(t)
                if not fund:
                    continue
                
                # 실제 expert.py 의 calculate_score 사용
                price_data = {
                    'rs_score': rs_score,
                    'momentum': stock_ret
                }
                score, sector = self.expert_strategy.calculate_score(t, fund, price_data)
                
                # 현재가 가져오기
                curr_price = df['Close'].iloc[-1]
                if hasattr(curr_price, 'item'): curr_price = curr_price.item()
                
                if score >= 70:
                    results.append({
                        'ticker': t,
                        'score': score,
                        'sector': sector,
                        'close': curr_price,
                        'rs_score': rs_score
                    })
                    if univ in sector_stats:
                        sector_stats[univ]['picked'] += 1
            except Exception as e:
                logger.debug(f"Expert 분석 실패 ({t}): {e}")
        
        return {'results': results, 'stats': sector_stats}
    
    def _run_rs(self, all_tickers, all_data, spy_ret):
        """RS 전략 실행 (전체 종목 처리, 13612W 모멘텀 사용)"""
        logger.info("[6/7] RS 전략 실행...")
        results = []
        
        for t in all_tickers:  # 전체 종목 처리
            try:
                df = self._extract_ticker_data(all_data, t)
                if df.empty or len(df) < 252:  # 13612W 는 252 일 필요
                    continue
                
                # ✅ rs.py 의 calculate_momentum_13612 사용 (Keller 공식)
                momentum_13612 = calculate_momentum_13612(df['Close'])
                
                # RS 점수 계산 (시장 대비)
                diff = momentum_13612 - spy_ret
                rs_score = 50 + (diff * 2.0)
                
                if rs_score >= 70:
                    # 현재가 가져오기
                    curr_price = df['Close'].iloc[-1]
                    if hasattr(curr_price, 'item'): curr_price = curr_price.item()
                    
                    results.append({
                        'ticker': t,
                        'rs_score': rs_score,
                        'momentum_13612': momentum_13612,
                        'close': curr_price
                    })
            except Exception as e:
                logger.debug(f"RS 분석 실패 ({t}): {e}")
        
        return {'results': results, 'stats': {}}
    
    def _build_holdings_data(self, my_positions, turtle_results, expert_results, all_data):
        """보유 종목 데이터 구축 (all_data 에서 직접 현재가 추출)"""
        # turtle_results 와 expert_results 는 {'results': [...], 'stats': {...}} 형태
        t_list = turtle_results.get('results', []) if isinstance(turtle_results, dict) else turtle_results
        e_list = expert_results.get('results', []) if isinstance(expert_results, dict) else expert_results
        
        t_map = {r['ticker']: r for r in t_list}
        e_map = {r['ticker']: r for r in e_list}
        
        holdings_data = []
        for p in my_positions:
            t = p['ticker']
            curr_t = t_map.get(t, {})
            curr_e = e_map.get(t, {})
            
            # 보유종목은 우선순위 조회 (2 일마다 갱신)
            fund = self.fetcher.fetch_with_retry(t, priority=True)
            sector = fund.get('sector', 'Default') if fund else 'Default'
            
            # all_data 에서 직접 현재가 추출 (turtle 결과에 없어도 정확함)
            df = self._extract_ticker_data(all_data, t)
            curr_price = p['price']  # 기본값은 매수단가
            if not df.empty:
                curr_price = df['Close'].iloc[-1]
                if hasattr(curr_price, 'item'): curr_price = curr_price.item()
            
            # rs_score 는 expert_results 에서 가져옴 (_run_expert 에서 계산됨)
            holdings_data.append({
                'ticker': t,
                'qty': p['qty'],
                'entry_price': p['price'],
                'current_price': curr_price,  # all_data 에서 직접 추출
                'signal': curr_t.get('signal', 'HOLD'),
                'expert_score': curr_e.get('score', 0),
                'rs_score': curr_e.get('rs_score', 0),  # expert_results 에서 가져옴
                'trend_score': curr_t.get('trend_score', 0),
                'sector': sector
            })
        
        return holdings_data
    
    def _generate_rebalance_plan(self, expert_results, turtle_results, rs_results, holdings_data):
        """리밸런싱 계획 생성"""
        logger.info("[7/7] 리밸런싱 계산 중...")
        
        # sector_mapper 는 함수를 직접 전달 (래퍼 객체 사용 안함)
        self.rebalance_manager = RebalanceManager(
            holdings_data=holdings_data,
            macro_analyzer=self.macro_analyzer,
            cash_balance=self.cash_balance,
            usd_krw=self.usd_krw,
            sector_mapper=type('obj', (object,), {
                'normalize_sector': staticmethod(normalize_sector),
                'get_sector_limit': staticmethod(get_sector_limit)
            })(),
            data_dir=self.data_dir
        )
        
        sells, scenarios = self.rebalance_manager.generate_plan(
            expert_results=expert_results.get('results', []),
            turtle_results=turtle_results.get('results', []),
            rs_results=rs_results.get('results', []),
            total_asset_val=self.total_account_value
        )
        
        return sells, scenarios
    
    def _generate_report(self, turtle_data, expert_data, rs_data, macro_data, holdings_data, scenarios):
        """리포트 생성"""
        logger.info("[8/8] 리포트 생성 중...")
        
        builder = HTMLReportBuilder(
            holdings_data=holdings_data,
            macro_data=macro_data,
            turtle_data=turtle_data,
            expert_data=expert_data,
            rs_data=rs_data,
            scenarios=scenarios,
            total_asset=self.total_account_value,
            cash_balance=self.cash_balance,
            usd_krw=self.usd_krw,
            macro_analyzer=self.macro_analyzer,
            sector_mapper=type('obj', (object,), {'normalize_sector': normalize_sector})()
        )
        
        html = builder.build()
        
        os.makedirs(self.reports_dir, exist_ok=True)
        report_path = os.path.join(self.reports_dir, "index.html")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"리포트 저장: {report_path}")
    
    def run(self):
        """시스템 실행"""
        logger.info("=" * 70)
        logger.info("통합 퀀트 시스템 V24.9 실행 시작")
        logger.info("=" * 70)
        
        # 1. 매크로 분석
        self.macro_analyzer.run_deep_dive()
        
        # 2. 환율 조회
        self.get_exchange_rate()
        
        # 3. 티커 목록 가져오기
        all_tickers, nasdaq, sp500, sox = self.fetch_dynamic_tickers()
        
        # 4. Universe map 생성
        u_map = {}
        for t in sp500: u_map[t] = 'S&P500'
        for t in nasdaq: u_map[t] = '나스닥 100'
        for t in sox: u_map[t] = '반도체 (SOX)'
        
        # 5. 데이터 다운로드 (as_of_date 지원 - 백테스트용)
        all_data = self.data_downloader.download_all(all_tickers, end_date=self.as_of_date)
        
        # 6. 보유 종목 분석 (CSV 에서 로드)
        my_positions = self.config.load_positions_from_csv()
        if not my_positions:
            logger.warning("holdings.csv 에서 포지션을 로드하지 못했습니다. config.json 의 positions 를 사용합니다.")
            my_positions = self.config.positions if hasattr(self.config, 'positions') else []
        logger.info(f"[3/7] 보유 종목 ({len(my_positions)}개) 정밀 분석 중...")
        
        # 7. 주식 평가금 계산 (현재가 사용, 실패시 진입가)
        current_stock_eval = 0
        for p in my_positions:
            t = p['ticker']
            price = p['price']  # 기본값: 진입가
            try:
                df = self._extract_ticker_data(all_data, t)
                if not df.empty and 'Close' in df.columns:
                    current_price = df['Close'].iloc[-1]
                    if hasattr(current_price, 'item'): current_price = current_price.item()
                    if current_price > 0:
                        price = current_price  # 현재가 사용
            except Exception as e:
                logger.debug(f"{t} 현재가 조회 실패, 진입가 사용: {e}")
            
            current_stock_eval += price * p['qty'] * self.usd_krw
        
        self.cash_balance = max(0, self.total_account_value - current_stock_eval)
        logger.info(f"환율: {self.usd_krw:,.2f} 원 | 총 자산: {self.total_account_value/10000:,.0f} 만원")
        logger.info(f"주식 평가금: {current_stock_eval/10000:,.0f} 만원 | 현금: {self.cash_balance/10000:,.0f} 만원")
        
        # 8. S&P500 벤치마크 한 번만 계산 (중복 다운로드 방지, as_of_date 지원)
        spy_ret = 0
        try:
            if self.as_of_date:
                # 백테스트: 과거 날짜 기준
                from datetime import timedelta
                end = self.as_of_date
                start = end - timedelta(days=90)
                spy_df = yf.download('^GSPC', start=start, end=end, progress=False)
            else:
                # 실전: 오늘 기준
                spy_df = yf.download('^GSPC', period='3mo', progress=False)
            if not spy_df.empty and len(spy_df) >= 63:
                spy_ret = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-63] - 1) * 100
        except:
            pass
        
        # 8. 전략 실행 (spy_ret 공유)
        turtle_results = self._run_turtle(all_tickers, all_data, u_map)
        expert_results = self._run_expert(all_tickers, all_data, u_map, spy_ret)
        rs_results = self._run_rs(all_tickers, all_data, spy_ret)
        
        # 9. 보유 종목 데이터 구축 (all_data 전달하여 현재가 정확히 추출)
        holdings_data = self._build_holdings_data(my_positions, turtle_results, expert_results, all_data)
        
        # 10. 리밸런싱
        sells, scenarios = self._generate_rebalance_plan(expert_results, turtle_results, rs_results, holdings_data)
        
        # 11. 매크로 데이터
        macro_data = self.macro_analyzer.indicators
        
        # 12. 리포트 생성
        self._generate_report(turtle_results, expert_results, rs_results, macro_data, holdings_data, scenarios)
        
        logger.info("🎉 모든 작업 완료!")
