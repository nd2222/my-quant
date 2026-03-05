"""
백테스팅 모듈 (Phase 3-2 완성)
- 과거 데이터 다운로드
- 리밸런싱 기록 저장
- 성과 분석 (수익률, 샤프비율, MDD)
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """과거 데이터 관리자"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_historical_data(self, ticker: str, as_of_date: pd.Timestamp, period: str = '1y') -> pd.DataFrame:
        """
        특정 날짜까지의 과거 데이터 다운로드
        
        Args:
            ticker: 종목 티커
            as_of_date: 분석 기준 날짜
            period: 다운로드 기간 ('1y', '6mo' 등)
        
        Returns:
            pd.DataFrame: 주가 데이터
        """
        # 캐시 확인
        cache_file = self.data_dir / f"{ticker}_history.csv"
        
        # 과거 날짜 계산
        if isinstance(period, str):
            if period.endswith('y'):
                years = int(period[:-1])
                start_date = as_of_date - pd.DateOffset(years=years)
            elif period.endswith('mo'):
                months = int(period[:-2])
                start_date = as_of_date - pd.DateOffset(months=months)
            else:
                start_date = as_of_date - pd.DateOffset(years=1)
        else:
            start_date = period
        
        # 캐시 사용 (같은 날짜면 재사용)
        if cache_file.exists():
            try:
                cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if len(cached_df) > 0:
                    cached_end = cached_df.index[-1]
                    if cached_end >= as_of_date:
                        # 캐시에서 자르기
                        result = cached_df[cached_df.index <= as_of_date]
                        logger.debug(f"{ticker}: 캐시 사용 ({len(result)}일)")
                        return result
            except Exception as e:
                logger.warning(f"{ticker}: 캐시 로드 실패, 새로 다운로드: {e}")
        
        # yfinance 로 다운로드
        try:
            logger.debug(f"{ticker}: 다운로드 중 ({start_date} ~ {as_of_date})...")
            data = yf.download(ticker, start=start_date, end=as_of_date + pd.Timedelta(days=1), progress=False)
            
            if data is not None and not data.empty:
                # 캐시 저장
                data.to_csv(cache_file)
                logger.debug(f"{ticker}: 다운로드 완료 ({len(data)}일)")
                return data
            else:
                logger.warning(f"{ticker}: 데이터 없음")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"{ticker}: 다운로드 실패: {e}")
            return pd.DataFrame()
    
    def get_batch_data(self, tickers: list, as_of_date: pd.Timestamp, period: str = '1y', max_workers: int = 4) -> dict:
        """
        여러 종목의 과거 데이터 일괄 다운로드
        
        Args:
            tickers: 티커 리스트
            as_of_date: 분석 기준 날짜
            period: 다운로드 기간
            max_workers: 병렬 작업자 수
        
        Returns:
            dict: {ticker: DataFrame}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.get_historical_data, ticker, as_of_date, period): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[ticker] = data
                except Exception as e:
                    logger.error(f"{ticker}: 배치 다운로드 실패: {e}")
        
        logger.info(f"과거 데이터 다운로드 완료: {len(results)}/{len(tickers)}개")
        return results


class RebalanceLogger:
    """리밸런싱 기록 관리자"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'rebalance_history.json'
    
    def save_rebalance_log(self, date: pd.Timestamp, portfolio: dict, metrics: dict):
        """
        리밸런싱 결과 저장
        
        Args:
            date: 리밸런싱 날짜
            portfolio: 포트폴리오 구성
                {ticker: {'qty': 수량, 'price': 가격, 'weight': 비중}}
            metrics: 성과 지표
                {'total_value': 총자산, 'cash': 현금, 'return': 수익률 등}
        """
        # 기존 기록 로드
        history = []
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []
        
        # 새 기록 추가
        record = {
            'date': date.strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'portfolio': portfolio,
            'metrics': metrics
        }
        history.append(record)
        
        # 저장
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"리밸런싱 기록 저장: {date.strftime('%Y-%m-%d')}")
    
    def load_history(self) -> list:
        """과거 리밸런싱 기록 로드"""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


class PerformanceAnalyzer:
    """성과 분석기"""
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Args:
            risk_free_rate: 무위험 수익률 (연율, 기본값 3%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, values: list) -> pd.Series:
        """수익률 계산"""
        series = pd.Series(values)
        return series.pct_change().dropna()
    
    def calculate_cagr(self, values: list, years: float) -> float:
        """
        연평균 수익률 (CAGR) 계산
        
        Args:
            values: 포트폴리오 가치 시계열
            years: 투자 기간 (년)
        
        Returns:
            float: 연평균 수익률
        """
        if len(values) < 2 or years <= 0:
            return 0.0
        
        total_return = (values[-1] - values[0]) / values[0]
        cagr = (1 + total_return) ** (1 / years) - 1
        return cagr
    
    def calculate_sharpe_ratio(self, values: list, years: float) -> float:
        """
        샤프 비율 계산
        
        Args:
            values: 포트폴리오 가치 시계열
            years: 투자 기간 (년)
        
        Returns:
            float: 샤프 비율
        """
        if len(values) < 2:
            return 0.0
        
        returns = self.calculate_returns(values)
        
        if len(returns) < 2:
            return 0.0
        
        # 일일 수익률을 연율화
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe
    
    def calculate_max_drawdown(self, values: list) -> float:
        """
        최대 낙폭 (MDD) 계산
        
        Args:
            values: 포트폴리오 가치 시계열
        
        Returns:
            float: 최대 낙폭 (음수)
        """
        if len(values) < 2:
            return 0.0
        
        series = pd.Series(values)
        rolling_max = series.expanding().max()
        drawdowns = (series - rolling_max) / rolling_max
        
        return drawdowns.min()
    
    def analyze(self, values: list, dates: list = None) -> dict:
        """
        종합 성과 분석
        
        Args:
            values: 포트폴리오 가치 시계열
            dates: 날짜 리스트 (선택사항)
        
        Returns:
            dict: 성과 지표
        """
        if len(values) < 2:
            return {}
        
        # 기간 계산 (년)
        if dates and len(dates) >= 2:
            days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
            years = days / 365.25
        else:
            years = len(values) / 252  # 영업일 기준
        
        metrics = {
            'total_return': (values[-1] - values[0]) / values[0],
            'cagr': self.calculate_cagr(values, years),
            'sharpe_ratio': self.calculate_sharpe_ratio(values, years),
            'max_drawdown': self.calculate_max_drawdown(values),
            'volatility': np.std(self.calculate_returns(values)) * np.sqrt(252),
            'period_days': len(values),
            'period_years': round(years, 2)
        }
        
        return metrics


class BacktestRunner:
    """백테스팅 실행기 (Phase 3-2 완성: 타임루프 + 가상 포트폴리오)"""
    
    def __init__(self, config, data_dir: str = 'backtest_data', log_dir: str = 'backtest_logs'):
        """
        Args:
            config: 설정 객체
            data_dir: 데이터 저장 디렉토리
            log_dir: 로그 저장 디렉토리
        """
        self.config = config
        self.data_manager = HistoricalDataManager(data_dir)
        self.rebalance_logger = RebalanceLogger(log_dir)
        self.performance_analyzer = PerformanceAnalyzer()
    
    def run_backtest(self, start_date: str, end_date: str, rebalance_freq: str = 'ME', initial_capital: float = None) -> dict:
        """
        백테스팅 실행 (타임루프 + 가상 포트폴리오)
        
        Args:
            start_date: 시작 날짜 ('2023-01-01')
            end_date: 종료 날짜 ('2023-12-31')
            rebalance_freq: 리밸런싱 빈도 ('ME': 월간, 'QE': 분기, 'YE': 연간)
            initial_capital: 초기 자본 (None 이면 config 의 total_krw 사용)
        
        Returns:
            dict: 백테스팅 결과
        """
        from core.quant_system import IntegratedQuantSystem
        
        # P0: pandas 호환성 수정 (M → ME)
        freq_map = {
            'M': 'ME',   # 월간 (구버전 → 신버전)
            'Q': 'QE',   # 분기
            'Y': 'YE',   # 연간
            'ME': 'ME',  # 월간 (신버전)
            'QE': 'QE',  # 분기 (신버전)
            'YE': 'YE'   # 연간 (신버전)
        }
        pandas_freq = freq_map.get(rebalance_freq, rebalance_freq)
        
        # 날짜 범위 생성 (타임루프)
        dates = pd.date_range(start_date, end_date, freq=pandas_freq)
        
        # 가상 포트폴리오 초기화
        if initial_capital is None:
            if hasattr(self.config.account, 'total_krw'):
                initial_capital = self.config.account.total_krw
            else:
                initial_capital = 76826207
        
        portfolio_values = []
        dates_list = []
        trades = []  # 매매 기록
        
        logger.info(f"백테스팅 시작: {start_date} ~ {end_date} ({rebalance_freq})")
        logger.info(f"초기 자본: ${initial_capital:,.0f}")
        
        # 타임루프 시작
        for date in dates:
            logger.info(f"\n{'='*60}")
            logger.info(f"백테스팅 날짜: {date.strftime('%Y-%m-%d')}")
            logger.info(f"{'='*60}")
            
            # 과거 데이터로 퀀트 시스템 실행
            system = IntegratedQuantSystem(self.config, as_of_date=date)
            system.run()
            
            # 포트폴리오 기록
            portfolio = {
                'date': date.strftime('%Y-%m-%d'),
                'tickers': [p['ticker'] for p in system.config.positions],
                'total_value': system.total_account_value,
                'cash': system.cash_balance,
                'stocks_value': current_stock_eval_krw if 'current_stock_eval_krw' in locals() else 0
            }
            
            # 매매 신호 기록 (이전과 비교)
            if len(dates_list) > 0:
                prev_portfolio = self.rebalance_logger.load_history()[-1] if self.rebalance_logger.load_history() else None
                if prev_portfolio:
                    trades.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'action': 'rebalance',
                        'prev_value': prev_portfolio['metrics']['total_value'],
                        'curr_value': system.total_account_value,
                        'return': (system.total_account_value - prev_portfolio['metrics']['total_value']) / prev_portfolio['metrics']['total_value']
                    })
            
            # 리밸런싱 기록 저장
            self.rebalance_logger.save_rebalance_log(date, portfolio, {
                'total_value': system.total_account_value,
                'cash': system.cash_balance
            })
            
            # 포트폴리오 가치 기록
            portfolio_values.append(system.total_account_value)
            dates_list.append(date.strftime('%Y-%m-%d'))
        
        # 성과 분석
        metrics = self.performance_analyzer.analyze(portfolio_values, dates_list)
        metrics['initial_capital'] = initial_capital
        metrics['final_value'] = portfolio_values[-1] if portfolio_values else 0
        metrics['total_trades'] = len(trades)
        
        logger.info(f"\n{'='*60}")
        logger.info("백테스팅 완료!")
        logger.info(f"{'='*60}")
        logger.info(f"기간: {start_date} ~ {end_date} ({len(dates)}회 리밸런싱)")
        logger.info(f"초기 자본: ${initial_capital:,.0f}")
        logger.info(f"최종 가치: ${metrics['final_value']:,.0f}")
        logger.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
        logger.info(f"연평균 수익률 (CAGR): {metrics['cagr']*100:.2f}%")
        logger.info(f"샤프 비율: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"최대 낙폭 (MDD): {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"변동성: {metrics['volatility']*100:.2f}%")
        
        return {
            'dates': dates_list,
            'values': portfolio_values,
            'trades': trades,
            'metrics': metrics
        }


# 사용 예시
if __name__ == "__main__":
    from core.config import Config
    
    # 설정 로드
    config = Config('config.json')
    config.load()
    
    # 백테스팅 실행 (2023 년 1 년치, 월별 리밸런싱)
    runner = BacktestRunner(config)
    results = runner.run_backtest('2023-01-01', '2023-12-31', rebalance_freq='M')
    
    # 결과 출력
    print("\n" + "="*60)
    print("백테스팅 결과 요약")
    print("="*60)
    print(f"기간: {results['dates'][0]} ~ {results['dates'][-1]}")
    print(f"초기 자산: ${results['values'][0]:,.0f}")
    print(f"최종 자산: ${results['values'][-1]:,.0f}")
    print(f"총 수익률: {results['metrics']['total_return']*100:.2f}%")
    print(f"연평균 수익률: {results['metrics']['cagr']*100:.2f}%")
    print(f"샤프 비율: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"최대 낙폭: {results['metrics']['max_drawdown']*100:.2f}%")
