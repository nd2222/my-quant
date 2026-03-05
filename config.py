"""
설정 관리 모듈
- config.json 로드 및 자동 업데이트
- DEFAULT_CONFIG 와의 병합 관리
- 설정 유효성 검사
"""
import os
import json
import logging
from datetime import datetime
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# ================= [기본 설정] =================
DEFAULT_CONFIG = {
    # 계좌 설정
    "account": {
        "total_krw": 76826207,          # 총 자산 (원)
        "risk_ratio": 0.01,             # 개별 종목 리스크 (1%)
        "max_alloc_per_stock": 0.20,    # 종목별 최대 비중 (20%)
        "expert_cutoff": 70,            # Expert 전략 점수 커트오프
        "min_invest_per_stock": 10000000  # 종목별 최소 투자금 (1,000 만원)
    },
    # 데이터 수집 설정
    "data": {
        "chunk_size": 50,               # yfinance 일괄 다운로드 청크 크기
        "delay_between_chunks": 2.0,    # 청크 간 딜레이 (초)
        "download_timeout": 30,         # 다운로드 타임아웃 (초)
        "download_period": "1y",        # 기본 다운로드 기간
        "macro_period": "6mo",          # 매크로 데이터 기간
        "rs_period_days": 63,           # RS 계산 기간
        "volume_avg_days": 21           # 거래량 평균 기간
    },
    # 환율 설정
    "exchange": {
        "default_rate": 1450.0,         # 기본 환율
        "cache_max_age_hours": 24       # 환율 캐시 유효기간
    },
    # 캐시/로그 설정
    "cache": {
        "log_max_bytes": 5242880,       # 로그 파일 최대 크기 (5MB)
        "log_backup_count": 3,          # 로그 백업 개수
        "log_dir": "logs"               # 로그 디렉토리
    },
    # CSV 설정 (경로는 Config 클래스에서 동적 생성)
    "csv": {
        "holdings_path": None,  # None 이면 config.py 기준으로 자동 생성
        "encoding": "cp949"
    },
    # API 키 설정 (선택사항)
    "api_keys": {
        "polygon_api_key": "",  # Polygon.io (러셀 2000 등)
        "massive_api_key": "",  # Massive.com
        "fmp_api_key": ""       # Financial Modeling Prep (선택)
    },
    # 디렉토리 설정 (None 이면 config.py 기준으로 자동 생성)
    "dirs": {
        "base": None,
        "data": None,
        "reports": None,
        "charts": None,
        "cache": None,
        "github": None,
        "logs": None
    },
    # 포지션 (CSV 로드 실패시 폴백용 - 실제 데이터는 holdings.csv 에서 로드)
    # 보안상 실제 매수 단가/수량은 소스코드가 아닌 CSV 에 저장
    "positions": []
}


class Config:
    """
    설정 관리 클래스
    
    사용법:
        config = Config(config_path)
        config.load()  # config.json 로드 및 병합
        
        # 설정 접근
        total_krw = config.account.total_krw
        chunk_size = config.data.chunk_size
        
        # 저장
        config.save()
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config = {}
        self._loaded = False
    
    @property
    def account(self):
        return SimpleNamespace(**self._config.get('account', DEFAULT_CONFIG['account']))
    
    @property
    def data(self):
        return SimpleNamespace(**self._config.get('data', DEFAULT_CONFIG['data']))
    
    @property
    def exchange(self):
        return SimpleNamespace(**self._config.get('exchange', DEFAULT_CONFIG['exchange']))
    
    @property
    def cache(self):
        return SimpleNamespace(**self._config.get('cache', DEFAULT_CONFIG['cache']))
    
    @property
    def csv(self):
        """CSV 설정 (하드코딩 제거 - config.py 위치 기준 상대경로)"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_csv = {
            'holdings_path': os.path.join(base_dir, 'holdings.csv'),
            'encoding': 'cp949'
        }
        # config.json 에서 오버라이드 있으면 사용
        user_csv = self._config.get('csv', {})
        if user_csv.get('holdings_path') is None:
            user_csv['holdings_path'] = default_csv['holdings_path']
        if 'encoding' not in user_csv:
            user_csv['encoding'] = default_csv['encoding']
        return SimpleNamespace(**user_csv)
    
    @property
    def dirs(self):
        """디렉토리 설정 (하드코딩 제거 - config.py 위치 기준 상대경로)"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dirs = {
            'base': base_dir,
            'data': os.path.join(base_dir, 'Data'),
            'reports': os.path.join(base_dir, 'Reports'),
            'charts': os.path.join(base_dir, 'Charts'),
            'cache': os.path.join(base_dir, 'Data', 'Cache'),
            'github': os.path.join(base_dir, 'my-quant'),
            'logs': os.path.join(base_dir, 'logs')
        }
        # config.json 에서 오버라이드 있으면 사용, 없으면 상대경로 기본값
        user_dirs = self._config.get('dirs', {})
        for k, v in default_dirs.items():
            if k not in user_dirs or user_dirs[k] is None:
                user_dirs[k] = v
        return SimpleNamespace(**user_dirs)
    
    @property
    def positions(self):
        """포지션 조회 (CSV 에서 로드 권장)"""
        return self._config.get('positions', DEFAULT_CONFIG['positions'])
    
    def load_positions_from_csv(self, csv_path: str = None) -> list:
        """
        holdings.csv 에서 포지션 로드 (컬럼명 기반 파싱 - CTO 지시사항 반영)
        
        Args:
            csv_path: CSV 파일 경로
        
        Returns:
            list: 포지션 리스트
        """
        if csv_path is None:
            # Phase 1-1: 상대 경로 사용 (하드코딩 제거)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, 'holdings.csv')
        
        if not os.path.exists(csv_path):
            logger.warning(f"holdings.csv 없음 ({csv_path}), config.json 의 positions 를 사용합니다")
            return self._config.get('positions', [])
        
        try:
            import chardet
            
            # 1. 인코딩 감지
            with open(csv_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'cp949'
            
            logger.info(f"CSV 인코딩 감지: {encoding}")
            
            # 2. CSV 파싱 (헤더 포함)
            import pandas as pd
            df = pd.read_csv(csv_path, encoding=encoding, skiprows=1, header=0)
            
            # Phase 1-2: 컬럼명 공백 제거 (CTO 팁)
            df.columns = df.columns.str.strip()
            
            # 3. 컬럼명 매핑 (인덱스 대신 이름 사용)
            # HTS 헤더: 종목, 종목명, 등락률, 등락률%, 진입가, 수량, ...
            # 또는: 종목코드, 종목명, ...
            
            # 가능한 컬럼명 변형 목록 (한글 + 영어)
            ticker_cols = ['종목', '종목코드', 'Ticker', 'Ticker Code', '코드']
            price_cols = ['진입가', '매입가', 'Entry Price', 'EntryPrice', 'Price', '매입금액']
            qty_cols = ['수량', '보유수량', 'Quantity', 'Qty', '보유량']
            
            # 실제 컬럼명 찾기
            ticker_col = next((col for col in ticker_cols if col in df.columns), None)
            price_col = next((col for col in price_cols if col in df.columns), None)
            qty_col = next((col for col in qty_cols if col in df.columns), None)
            
            if not all([ticker_col, price_col, qty_col]):
                logger.warning(f"CSV 컬럼을 찾을 수 없음 (티커:{ticker_col}, 가격:{price_col}, 수량:{qty_col})")
                logger.warning(f"실제 컬럼: {list(df.columns)}")
                return self._config.get('positions', [])
            
            positions = []
            for idx, row in df.iterrows():
                try:
                    # Phase 1-2: 컬럼명 기반 파싱 + 공백 제거
                    ticker = str(row[ticker_col]).strip().strip("'").strip()
                    
                    # HTS 에 따라 '매입금액' (총액) 또는 '매입가' (단가) 일 수 있음
                    price_raw = str(row[price_col]).strip().replace(',', '').replace('"', '')
                    price = float(price_raw)
                    
                    qty_raw = str(row[qty_col]).strip().replace(',', '')
                    qty = int(float(qty_raw))
                    
                    # '매입금액' 인 경우 주당 가격으로 변환
                    if price_col and '금액' in price_col and qty > 0:
                        price = price / qty  # 총액 → 단가
                    
                    if ticker and qty > 0 and price > 0:
                        positions.append({
                            'ticker': ticker,
                            'price': price,
                            'qty': qty,
                            'entry_date': ''
                        })
                        logger.info(f"✅ {ticker}: {qty}주 @ ${price:.2f}")
                except Exception as e:
                    logger.debug(f"행 {idx} 파싱 실패: {e}")
                    continue
            
            if positions:
                total_value = sum(p['price'] * p['qty'] for p in positions)
                logger.info(f"holdings.csv 에서 {len(positions)}개 종목 로드")
                logger.info(f"총 평가금: ${total_value:,.2f} (약 {total_value * 1465 / 10000000:,.0f} 만원)")
                
                # P3: config.json 의 positions 도 업데이트 (지속성)
                self._config['positions'] = positions
                logger.info(f"config.json 의 positions 업데이트 ({len(positions)}개)")
                
                # 자동으로 config.json 에 저장
                try:
                    self.save()
                    logger.info(f"config.json 저장 완료")
                except Exception as e:
                    logger.warning(f"config.json 저장 실패: {e}")
            else:
                logger.warning("CSV 에서 종목을 로드하지 못했습니다. config.json 을 사용합니다")
                return self._config.get('positions', [])
            
            return positions
        
        except Exception as e:
            logger.error(f"CSV 로드 실패: {e}")
            logger.warning("config.json 의 positions 를 사용합니다")
            return self._config.get('positions', [])
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        딕셔너리 깊은 병합
        override 의 값으로 base 를 업데이트하되, 중첩된 dict 도 재귀적으로 병합
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def load(self, use_csv: bool = True) -> bool:
        """
        config.json 로드 및 DEFAULT_CONFIG 와 병합
        
        Args:
            use_csv: holdings.csv 에서 positions 로드 여부 (기본값: True)
        
        Returns:
            bool: 로드 성공 여부
        """
        if not os.path.exists(self.config_path):
            logger.info(f"설정 파일 없음 ({self.config_path}), 기본 설정 사용")
            self._config = DEFAULT_CONFIG.copy()
            self._loaded = True
            return True
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # DEFAULT_CONFIG 와 병합 (누락된 키 자동 추가)
            self._config = self._deep_merge(DEFAULT_CONFIG, loaded_config)
            self._loaded = True
            
            logger.info(f"설정 파일 로드: {self.config_path}")
            
            # P4: holdings.csv 에서 positions 자동 로드 (우선순위 높음)
            csv_path = self.csv.holdings_path  # ✅ 하드코딩 제거, 프로퍼티 활용
            if use_csv and os.path.exists(csv_path):
                csv_positions = self.load_positions_from_csv(csv_path)
                if csv_positions:
                    self._config['positions'] = csv_positions
                    logger.info(f"holdings.csv 에서 {len(csv_positions)}개 종목 로드 (우선 적용)")
            
            # config.json 업데이트 (누락된 키 추가)
            if loaded_config != self._config:
                self.save()
                logger.info("설정 파일 자동 업데이트 (누락된 키 추가)")
            
            return True
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}, 기본 설정 사용")
            self._config = DEFAULT_CONFIG.copy()
            self._loaded = True
            return False
    
    def save(self) -> bool:
        """현재 설정을 config.json 에 저장"""
        if not self._loaded:
            logger.warning("설정이 로드되지 않음, 저장 건너뜀")
            return False
        
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"설정 파일 저장: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"설정 파일 저장 실패: {e}")
            return False
    
    def update(self, key_path: str, value) -> bool:
        """
        설정 값 업데이트
        
        Args:
            key_path: 도트 구분 키 경로 (예: "account.total_krw")
            value: 설정 값
        
        Returns:
            bool: 업데이트 성공 여부
        """
        if not self._loaded:
            logger.warning("설정이 로드되지 않음")
            return False
        
        keys = key_path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                logger.error(f"키 경로 없음: {key_path}")
                return False
            current = current[key]
        
        current[keys[-1]] = value
        logger.info(f"설정 업데이트: {key_path} = {value}")
        return True
    
    def get(self, key_path: str, default=None):
        """
        설정 값 조회
        
        Args:
            key_path: 도트 구분 키 경로 (예: "account.total_krw")
            default: 기본 값
        
        Returns:
            설정 값
        """
        if not self._loaded:
            return default
        
        keys = key_path.split('.')
        current = self._config
        
        for key in keys:
            if key not in current:
                return default
            current = current[key]
        
        return current
    
    def __getitem__(self, key):
        return self._config.get(key, DEFAULT_CONFIG.get(key))
    
    def __repr__(self):
        return f"Config(loaded={self._loaded}, path={self.config_path})"


# ================= [편의 함수] =================
def load_config(config_path: str = None) -> Config:
    """
    설정 로드 편의 함수
    
    Args:
        config_path: 설정 파일 경로 (기본값: C:\\Quant\\config.json)
    
    Returns:
        Config 인스턴스
    """
    if config_path is None:
        config_path = r"C:\Quant\config.json"
    
    config = Config(config_path)
    config.load()
    return config


# ================= [전역 설정 인스턴스] =================
# import 시점에 자동으로 로드되지 않음 (명시적 로드 필요)
_config_instance = None

def get_config() -> Config:
    """전역 설정 인스턴스 반환 (로드되지 않았으면 자동 로드)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance
