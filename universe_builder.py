"""
유니버스 빌더 모듈
- S&P 500, Nasdaq 100, SOX, Russell 2000 수집
- 하드코딩 제거
- 캐시 관리
"""
import os
import json
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Set, Dict, Optional
import io

logger = logging.getLogger(__name__)


class TickerSource:
    """티커 수집 소스 인터페이스"""
    
    def fetch(self) -> List[str]:
        """티커 목록 수집"""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """소스 이름 반환"""
        raise NotImplementedError


class SP500Source(TickerSource):
    """S&P 500 티커 수집 (Wikipedia)"""
    
    def __init__(self, cache_dir: str = None, cache_days: int = 7):
        self.cache_dir = cache_dir or r"C:\Quant\Data\Cache"
        self.cache_days = cache_days
        self.cache_file = os.path.join(self.cache_dir, "sp500_cache.json")
    
    def fetch(self) -> List[str]:
        # 캐시 확인
        cached = self._load_cache()
        if cached:
            logger.info(f"  • S&P 500: {len(cached)}개 (캐시)")
            return cached
        
        # Wikipedia 에서 수집
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0'}
            html = requests.get(url, headers=headers, timeout=10).text
            dfs = pd.read_html(io.StringIO(html))
            
            for df in dfs:
                if 'Symbol' in df.columns:
                    tickers = df['Symbol'].dropna().tolist()
                    tickers = [str(t).strip().replace('.', '-') for t in tickers if len(str(t)) <= 5]
                    
                    if len(tickers) >= 400:
                        self._save_cache(tickers)
                        logger.info(f"  • S&P 500: {len(tickers)}개 (Wikipedia)")
                        return tickers
        except Exception as e:
            logger.warning(f"  S&P 500 수집 실패: {e}")
        
        # 폴백
        fallback = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK.B']
        logger.warning(f"  • S&P 500: {len(fallback)}개 (폴백)")
        return fallback
    
    def _load_cache(self) -> Optional[List[str]]:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cache_date = datetime.strptime(data['date'], '%Y-%m-%d')
                if datetime.now() - cache_date < timedelta(days=self.cache_days):
                    return data['tickers']
        except:
            pass
        return None
    
    def _save_cache(self, tickers: List[str]):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'tickers': tickers
                }, f)
        except Exception as e:
            logger.debug(f"  S&P 500 캐시 저장 실패: {e}")
    
    def get_name(self) -> str:
        return "S&P 500"


class Nasdaq100Source(TickerSource):
    """Nasdaq 100 티커 수집 (Wikipedia)"""
    
    def __init__(self, cache_dir: str = None, cache_days: int = 7):
        self.cache_dir = cache_dir or r"C:\Quant\Data\Cache"
        self.cache_days = cache_days
        self.cache_file = os.path.join(self.cache_dir, "nasdaq100_cache.json")
    
    def fetch(self) -> List[str]:
        cached = self._load_cache()
        if cached:
            logger.info(f"  • Nasdaq 100: {len(cached)}개 (캐시)")
            return cached
        
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            headers = {'User-Agent': 'Mozilla/5.0'}
            html = requests.get(url, headers=headers, timeout=10).text
            dfs = pd.read_html(io.StringIO(html))
            
            for df in dfs:
                if 'Ticker' in df.columns:
                    tickers = df['Ticker'].dropna().tolist()
                    tickers = [str(t).strip() for t in tickers if len(str(t)) <= 5]
                    
                    if len(tickers) >= 90:
                        self._save_cache(tickers)
                        logger.info(f"  • Nasdaq 100: {len(tickers)}개 (Wikipedia)")
                        return tickers
        except Exception as e:
            logger.warning(f"  Nasdaq 100 수집 실패: {e}")
        
        fallback = ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMZN', 'META', 'TSLA', 'GOOGL']
        logger.warning(f"  • Nasdaq 100: {len(fallback)}개 (폴백)")
        return fallback
    
    def _load_cache(self) -> Optional[List[str]]:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cache_date = datetime.strptime(data['date'], '%Y-%m-%d')
                if datetime.now() - cache_date < timedelta(days=self.cache_days):
                    return data['tickers']
        except:
            pass
        return None
    
    def _save_cache(self, tickers: List[str]):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'tickers': tickers
                }, f)
        except Exception as e:
            logger.debug(f"  Nasdaq 100 캐시 저장 실패: {e}")
    
    def get_name(self) -> str:
        return "Nasdaq 100"


class SOXXSource(TickerSource):
    """SOX 반도체 티커 수집 (yfinance SOXX ETF)"""
    
    def __init__(self):
        self.fallback = [
            'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC',
            'ADI', 'MCHP', 'NXPI', 'MRVL', 'ON', 'MPWR', 'SWKS', 'QRVO', 'TER', 'ENTG',
            'ALGM', 'WOLF', 'COHR', 'RMBS', 'LSCC', 'SLAB', 'CRUS', 'SITM', 'HIMX', 'UCTT'
        ]
    
    def fetch(self) -> List[str]:
        # yfinance SOXX ETF holdings
        try:
            soxx = yf.Ticker("SOXX")
            if hasattr(soxx, 'top_holdings') and soxx.top_holdings is not None:
                holdings = soxx.top_holdings
                if len(holdings) > 0:
                    tickers = holdings.index.tolist()[:30]
                    tickers = [t.strip().upper() for t in tickers if len(t.strip()) <= 5]
                    
                    if len(tickers) >= 20:
                        logger.info(f"  ✅ SOXX: {len(tickers)}개 (yfinance)")
                        return tickers
        except Exception as e:
            logger.debug(f"  SOXX yfinance 실패: {e}")
        
        logger.warning(f"  ⚠️ SOXX: {len(self.fallback)}개 (폴백)")
        return self.fallback
    
    def get_name(self) -> str:
        return "SOX 반도체"


class Russell2000Source(TickerSource):
    """러셀 2000 티커 수집 (IWM ETF + 실제 구성종목)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.constituents = self._load_constituents()
    
    def _load_constituents(self) -> List[str]:
        """러셀 2000 실제 구성종목 로드 (하드코딩 제거 - 외부 파일)"""
        # 실제 구성종목은 외부 CSV 파일로 관리
        constituents_file = r"C:\Quant\data\russell2000_constituents.csv"
        
        if os.path.exists(constituents_file):
            try:
                df = pd.read_csv(constituents_file)
                if 'Ticker' in df.columns:
                    tickers = df['Ticker'].dropna().tolist()
                    logger.debug(f"  러셀 2000 구성종목 로드: {len(tickers)}개")
                    return tickers
            except Exception as e:
                logger.debug(f"  러셀 2000 파일 로드 실패: {e}")
        
        # 폴백: IWM ETF holdings
        try:
            iwm = yf.Ticker("IWM")
            if hasattr(iwm, 'top_holdings') and iwm.top_holdings is not None:
                holdings = iwm.top_holdings
                if len(holdings) > 0:
                    tickers = holdings.index.tolist()[:200]
                    logger.debug(f"  러셀 2000 IWM holdings: {len(tickers)}개")
                    return tickers
        except:
            pass
        
        return []
    
    def fetch(self, exclude: Set[str] = None) -> List[str]:
        if exclude is None:
            exclude = set()
        
        # API 키 있으면 Polygon.io 사용
        if self.api_key:
            try:
                url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&exchange=NYSE,NASDAQ,AMEX&limit=2000&apiKey={self.api_key}"
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data:
                        tickers = [t['ticker'] for t in data['results'] if 'ticker' in t]
                        tickers = [t for t in tickers if t not in exclude][:500]
                        
                        if len(tickers) >= 50:
                            logger.info(f"  • 러셀 2000: {len(tickers)}개 (Polygon.io)")
                            return tickers
            except Exception as e:
                logger.debug(f"  Polygon.io 실패: {e}")
        
        # 폴백: 구성종목 사용
        if self.constituents:
            tickers = [t for t in self.constituents if t not in exclude][:500]
            if len(tickers) >= 50:
                logger.info(f"  • 러셀 2000: {len(tickers)}개 (구성종목)")
                return tickers
        
        return []
    
    def get_name(self) -> str:
        return "러셀 2000"


class ThematicSource(TickerSource):
    """테마 종목 수집 (ETF holdings 기반)"""
    
    def __init__(self):
        self.etf_map = {
            '사이버보안': 'CIBR',
            '클라우드': 'CLOU',
            '바이오': 'IBB',
            '핀테크': 'FINX',
            '클린에너지': 'ICLN'
        }
    
    def fetch(self, exclude: Set[str] = None) -> List[str]:
        if exclude is None:
            exclude = set()
        
        all_tickers = []
        
        for theme, etf in self.etf_map.items():
            try:
                etf_ticker = yf.Ticker(etf)
                if hasattr(etf_ticker, 'top_holdings') and etf_ticker.top_holdings is not None:
                    holdings = etf_ticker.top_holdings
                    if len(holdings) > 0:
                        tickers = holdings.index.tolist()[:8]
                        tickers = [t for t in tickers if t not in exclude and len(t) <= 5]
                        all_tickers.extend(tickers)
            except Exception as e:
                logger.debug(f"  {theme} ETF 실패: {e}")
        
        if len(all_tickers) >= 10:
            logger.info(f"  • 테마 종목: {len(all_tickers)}개 (ETF)")
        
        return all_tickers
    
    def get_name(self) -> str:
        return "테마 종목"


class UniverseBuilder:
    """유니버스 빌더 (모든 소스 통합)"""
    
    def __init__(self, config: dict = None, cache_dir: str = None):
        """
        Args:
            config: 설정 dict (api_keys 포함)
            cache_dir: 캐시 디렉토리
        """
        self.config = config or {}
        self.cache_dir = cache_dir or r"C:\Quant\Data\Cache"
        
        api_keys = self.config.get('api_keys', {})
        
        # 소스 초기화
        self.sources = [
            SP500Source(self.cache_dir),
            Nasdaq100Source(self.cache_dir),
            SOXXSource(),
            Russell2000Source(api_keys.get('polygon_api_key')),
            ThematicSource()
        ]
    
    def build(self, exclude: Set[str] = None) -> Dict[str, List[str]]:
        """
        유니버스 구축
        
        Args:
            exclude: 제외할 티커 집합
        
        Returns:
            Dict[str, List[str]]: {소스명: [티커목록]}
        """
        if exclude is None:
            exclude = set()
        
        result = {}
        all_tickers = set()
        
        for source in self.sources:
            try:
                if source.get_name() in ['러셀 2000', '테마 종목']:
                    tickers = source.fetch(exclude=all_tickers)
                else:
                    tickers = source.fetch()
                
                if tickers:
                    result[source.get_name()] = tickers
                    all_tickers.update(tickers)
            except Exception as e:
                logger.error(f"  {source.get_name()} 수집 실패: {e}")
        
        logger.info(f"  • 총 유니버스: {len(all_tickers)}개")
        return result
    
    def get_all_tickers(self) -> List[str]:
        """모든 티커 반환 (중복 제거)"""
        result = self.build()
        all_tickers = set()
        
        for tickers in result.values():
            all_tickers.update(tickers)
        
        return sorted(list(all_tickers))
    
    def get_universe_map(self, all_tickers: List[str] = None) -> Dict[str, str]:
        """
        티커 → 유니버스 매핑 생성
        
        Args:
            all_tickers: 전체 티커 목록 (없으면 자동 수집)
        
        Returns:
            Dict[str, str]: {티커: 유니버스명}
        """
        if all_tickers is None:
            all_tickers = self.get_all_tickers()
        
        result = self.build()
        u_map = {}
        
        # 각 소스별로 매핑
        for source_name, tickers in result.items():
            for t in tickers:
                if t not in u_map:
                    u_map[t] = source_name
        
        # 매핑되지 않은 티커
        for t in all_tickers:
            if t not in u_map:
                u_map[t] = '기타'
        
        return u_map


# 편의 함수
def build_universe(config: dict = None) -> tuple:
    """
    유니버스 구축 (간편 함수)
    
    Args:
        config: 설정 dict
    
    Returns:
        tuple: (all_tickers, u_map)
    """
    builder = UniverseBuilder(config)
    all_tickers = builder.get_all_tickers()
    u_map = builder.get_universe_map(all_tickers)
    
    return all_tickers, u_map
