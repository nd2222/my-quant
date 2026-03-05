"""
펀더멘털 데이터 수집 모듈
- yfinance info 기반 지표 수집
- JSON 캐싱
- 실패 티커 관리
"""
import os
import json
import logging
import time
import random
from datetime import datetime
import yfinance as yf

from utils.sector_mapper import normalize_sector

logger = logging.getLogger(__name__)


class SmartFundamentalsFetcher:
    """
    스마트 펀더멘털 수집기
    
    기능:
        - JSON 캐싱 (중복 조회 방지)
        - 재시도 로직
        - 실패 티커 추적
        - 섹터 정규화
    """
    
    def __init__(self, config):
        self.config = config
        data_dir = config.dirs.data if hasattr(config, 'dirs') else r"C:\Quant\Data"
        
        self.financials_file = os.path.join(data_dir, "financials.json")
        self.failed_file = os.path.join(data_dir, "failed_tickers.json")
        
        self.financials = self._load_json(self.financials_file, {"stocks": {}})
        self.failed = self._load_json(self.failed_file, {"failed": {}})
    
    def _load_json(self, filepath: str, default: dict) -> dict:
        """JSON 파일 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default
    
    def _save_json(self, filepath: str, data: dict):
        """JSON 파일 저장"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def fetch_with_retry(self, ticker: str, max_retries: int = 3, priority: bool = False) -> dict:
        """
        재시도付き 펀더멘털 조회
        
        Args:
            ticker: 종목 티커
            max_retries: 최대 재시도 횟수
            priority: 우선순위 (대기시간 증가)
        
        Returns:
            dict: 펀더멘털 데이터
        """
        # 1. 캐시 확인
        if ticker in self.financials['stocks']:
            data = self.financials['stocks'][ticker]
            last_update = datetime.strptime(data.get('updated', '2000-01-01'), "%Y-%m-%d")
            limit_days = 2 if priority else 14
            
            if (datetime.now() - last_update).days < limit_days:
                return data
        
        # 2. 재시도 루프
        for attempt in range(max_retries):
            try:
                # 대기 (우선순위에 따라 차등)
                wait = (attempt + 1) * 1.0 if priority else 0.3
                time.sleep(wait + random.uniform(0, 0.5))
                
                # yfinance 조회
                info = yf.Ticker(ticker).info
                
                if not info or len(info) < 5:
                    raise ValueError("Empty Data")
                
                # 섹터 정규화
                sector = info.get('sector', '기타')
                sector = normalize_sector(sector)
                
                # 데이터 정제
                def clean(val):
                    return round(val * 100, 2) if val else 0
                
                def clean_raw(val):
                    return round(val, 2) if val else 0
                
                data = {
                    'roe': clean(info.get('returnOnEquity', 0)),
                    'pbr': clean_raw(info.get('priceToBook', 0)),
                    'per': clean_raw(info.get('trailingPE', 0)),
                    'psr': clean_raw(info.get('priceToSalesTrailing12Months', 0)),
                    'debt_ratio': clean_raw(info.get('debtToEquity', 0)),
                    'growth': clean(info.get('revenueGrowth', 0)),
                    'div': clean(info.get('dividendYield', 0)),
                    'sector': sector,
                    'updated': datetime.now().strftime("%Y-%m-%d"),
                    'quality': 'good'
                }
                
                # 캐시 저장
                self.financials['stocks'][ticker] = data
                self._save_json(self.financials_file, self.financials)
                
                logger.info(f"✅ {ticker}: 수신 완료")
                return data
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"⚠️ {ticker}: 수신 실패 - {str(e)[:50]}")
                    self.failed['failed'][ticker] = {
                        'last_attempt': datetime.now().strftime("%Y-%m-%d"),
                        'error': str(e)[:50]
                    }
                    self._save_json(self.failed_file, self.failed)
        
        return None
    
    def get_data(self, ticker: str) -> dict:
        """
        캐시된 데이터 조회
        
        Args:
            ticker: 종목 티커
        
        Returns:
            dict: 펀더멘털 데이터
        """
        return self.financials['stocks'].get(ticker, None)
