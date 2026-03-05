"""
데이터 로더 모듈
- yfinance 일괄 다운로드
- 청크 단위 처리
- 메모리 효율적 로딩
"""
import logging
import time
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class EnhancedDataDownloader:
    """
    향상된 데이터 다운로더
    
    기능:
        - 청크 단위 다운로드 (Rate Limit 대응)
        - Lazy loading (메모리 효율)
        - 에러 처리 및 재시도
    """
    
    def __init__(self, config):
        self.config = config
        self.chunk_size = config.data.chunk_size if hasattr(config, 'data') else 50
        self.delay_between_chunks = config.data.delay_between_chunks if hasattr(config, 'data') else 2.0
        self.download_timeout = config.data.download_timeout if hasattr(config, 'data') else 30
        self.download_period = config.data.download_period if hasattr(config, 'data') else "1y"
    
    def download_all(self, tickers: list, period: str = None, use_lazy: bool = False, end_date=None) -> pd.DataFrame:
        """
        일괄 다운로드 (과거 날짜 지원)
        
        Args:
            tickers: 티커 리스트
            period: 다운로드 기간
            use_lazy: Lazy loading 사용 여부
            end_date: 종료 날짜 (None 이면 오늘, 백테스트용)
        
        Returns:
            pd.DataFrame: 가격 데이터
        """
        if period is None:
            period = self.download_period
        
        tickers = list(set(tickers))
        chunks = [tickers[i:i + self.chunk_size] for i in range(0, len(tickers), self.chunk_size)]
        
        # end_date 가 있으면 기간 계산
        if end_date:
            from datetime import timedelta
            period_days = {'1y': 365, '6mo': 180, '3mo': 90, '1mo': 30}.get(period, 365)
            start_date = end_date - timedelta(days=period_days)
            logger.info(f"총 {len(tickers)}개 종목 데이터 수집 중 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})...")
        else:
            logger.info(f"총 {len(tickers)}개 종목 데이터 수집 중 (최근 {period})...")
        
        if use_lazy:
            return self._download_lazy(chunks, period)
        
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"청크 {i+1}/{len(chunks)} 다운로드 중 ({len(chunk)}개)...")
            try:
                if end_date:
                    # 과거 날짜 지정 다운로드
                    data = yf.download(
                        chunk, 
                        start=start_date,
                        end=end_date,
                        group_by='ticker', 
                        progress=False, 
                        threads=False, 
                        timeout=self.download_timeout
                    )
                else:
                    # 일반 다운로드 (오늘 기준)
                    data = yf.download(
                        chunk, 
                        period=period, 
                        group_by='ticker', 
                        progress=False, 
                        threads=False, 
                        timeout=self.download_timeout
                    )
                if not data.empty:
                    results.append(data)
                time.sleep(self.delay_between_chunks)
            except Exception as e:
                logger.error(f"Download Error: {e}")
        
        if results:
            try:
                return pd.concat(results, axis=1)
            except:
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def _download_lazy(self, chunks: list, period: str) -> pd.DataFrame:
        """
        Lazy loading - 메모리 효율적 다운로드
        """
        import gc
        final_df = None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"청크 {i+1}/{len(chunks)} 다운로드 중 ({len(chunk)}개)... [Lazy Mode]")
            try:
                data = yf.download(
                    chunk, 
                    period=period, 
                    group_by='ticker', 
                    progress=False, 
                    threads=False, 
                    timeout=self.download_timeout
                )
                if not data.empty:
                    if final_df is None:
                        final_df = data.copy()
                    else:
                        final_df = pd.concat([final_df, data], axis=1, copy=False)
                    del data
                    gc.collect()
                time.sleep(self.delay_between_chunks)
            except Exception as e:
                logger.error(f"Download Error (chunk {i+1}): {e}")
        
        return final_df if final_df is not None else pd.DataFrame()


if __name__ == "__main__":
    # 테스트용
    from core.config import load_config
    config = load_config()
    downloader = EnhancedDataDownloader(config)
    print("Data loader ready")
