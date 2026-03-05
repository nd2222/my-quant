"""
설정 관리 모듈 (Pydantic 기반)
- 타입 안전성
- 자동 검증
- 단일 소스 (config.json)
"""
import os
import json
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class AccountConfig(BaseModel):
    """계좌 설정"""
    total_krw: int = Field(default=76826207, description="총 자산 (원)")
    risk_ratio: float = Field(default=0.01, ge=0.001, le=0.1, description="개별 종목 리스크")
    max_alloc_per_stock: float = Field(default=0.20, ge=0.1, le=0.5, description="종목별 최대 비중")
    expert_cutoff: int = Field(default=70, ge=50, le=90, description="Expert 전략 점수 커트오프")
    min_invest_per_stock: int = Field(default=10000000, ge=1000000, description="종목별 최소 투자금")
    half_kelly_ratio: float = Field(default=0.08, ge=0.01, le=0.5, description="하프 켈리 비율")
    min_cash_buffer: float = Field(default=0.05, ge=0.0, le=0.5, description="현금 최소 버퍼")


class DataConfig(BaseModel):
    """데이터 수집 설정"""
    chunk_size: int = Field(default=50, ge=10, le=100, description="다운로드 청크 크기")
    delay_between_chunks: float = Field(default=2.0, ge=0.5, le=10.0, description="청크 간 딜레이")
    download_timeout: int = Field(default=30, ge=10, le=120, description="다운로드 타임아웃")
    download_period: str = Field(default="1y", description="다운로드 기간")
    macro_period: str = Field(default="6mo", description="매크로 데이터 기간")
    rs_period_days: int = Field(default=63, ge=21, le=252, description="RS 계산 기간")
    volume_avg_days: int = Field(default=21, ge=5, le=60, description="거래량 평균 기간")


class ExchangeConfig(BaseModel):
    """환율 설정"""
    default_rate: float = Field(default=1450.0, ge=1000.0, le=2000.0, description="기본 환율")
    cache_max_age_hours: int = Field(default=24, ge=1, le=168, description="환율 캐시 유효기간")


class CacheConfig(BaseModel):
    """캐시/로그 설정"""
    log_max_bytes: int = Field(default=5242880, ge=1048576, le=10485760, description="로그 파일 최대 크기")
    log_backup_count: int = Field(default=3, ge=1, le=10, description="로그 백업 개수")
    log_dir: str = Field(default="logs", description="로그 디렉토리")


class CsvConfig(BaseModel):
    """CSV 설정"""
    holdings_path: Optional[str] = Field(default=None, description="holdings.csv 경로")
    encoding: str = Field(default="cp949", description="CSV 인코딩")


class DirsConfig(BaseModel):
    """디렉토리 설정"""
    base: Optional[str] = Field(default=None, description="기본 디렉토리")
    data: Optional[str] = Field(default=None, description="데이터 디렉토리")
    reports: Optional[str] = Field(default=None, description="리포트 디렉토리")
    charts: Optional[str] = Field(default=None, description="차트 디렉토리")
    cache: Optional[str] = Field(default=None, description="캐시 디렉토리")
    github: Optional[str] = Field(default=None, description="GitHub 동기화 디렉토리")
    logs: Optional[str] = Field(default=None, description="로그 디렉토리")


class ApiKeysConfig(BaseModel):
    """API 키 설정"""
    polygon_api_key: str = Field(default="", description="Polygon.io API 키")
    massive_api_key: str = Field(default="", description="Massive.com API 키")
    fmp_api_key: str = Field(default="", description="FMP API 키")


class Config(BaseModel):
    """
    통합 설정 클래스
    
    사용법:
        config = Config.load("config.json")
        print(config.account.total_krw)
        
        config.save("config.json")
    """
    account: AccountConfig = Field(default_factory=AccountConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    csv: CsvConfig = Field(default_factory=CsvConfig)
    dirs: DirsConfig = Field(default_factory=DirsConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    
    # 확장 옵션
    expand_universe: bool = Field(default=True, description="확장 유니버스 활성화")
    include_russell: bool = Field(default=True, description="러셀 2000 포함")
    include_thematic: bool = Field(default=True, description="테마 종목 포함")
    
    # 포지션 (CSV 에서 로드 권장)
    positions: List[Dict] = Field(default_factory=list, description="보유 종목")
    
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """
        config.json 에서 로드
        
        Args:
            config_path: config.json 경로
        
        Returns:
            Config: 설정 인스턴스
        """
        if not os.path.exists(config_path):
            logger.warning(f"config.json 없음 ({config_path}), 기본값 사용")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = cls(**data)
            logger.info(f"설정 로드 완료: {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            return cls()
    
    def save(self, config_path: str) -> bool:
        """
        config.json 에 저장
        
        Args:
            config_path: 저장할 경로
        
        Returns:
            bool: 성공 여부
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.dict(), f, indent=4, ensure_ascii=False)
            
            logger.info(f"설정 저장 완료: {config_path}")
            return True
        
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
            return False
    
    def validate_all(self) -> List[str]:
        """
        모든 설정 검증
        
        Returns:
            List[str]: 경고 목록
        """
        warnings = []
        
        # 계좌 검증
        if self.account.total_krw < 10000000:
            warnings.append("총 자산이 1000 만원 미만입니다")
        
        if self.account.risk_ratio > 0.05:
            warnings.append(f"리스크 비율이 높습니다 ({self.account.risk_ratio*100:.1f}%)")
        
        # API 키 검증
        if self.expand_universe and not self.api_keys.polygon_api_key:
            warnings.append("확장 유니버스를 사용하려면 Polygon.io API 키가 필요합니다")
        
        if warnings:
            logger.warning(f"설정 검증 경고: {len(warnings)}개")
            for w in warnings:
                logger.warning(f"  - {w}")
        
        return warnings


# 편의 함수
def get_config(config_path: str = None) -> Config:
    """
    설정 인스턴스 생성 (단일 인스턴스 권장)
    
    Args:
        config_path: config.json 경로 (기본: C:\Quant\config.json)
    
    Returns:
        Config: 설정 인스턴스
    """
    if config_path is None:
        config_path = r"C:\Quant\config.json"
    
    return Config.load(config_path)


# 스크립트 실행 시 테스트
if __name__ == "__main__":
    # 테스트
    config = get_config()
    
    print("=" * 60)
    print("설정 검증 결과")
    print("=" * 60)
    print(f"총 자산: {config.account.total_krw:,} 원")
    print(f"리스크 비율: {config.account.risk_ratio*100:.1f}%")
    print(f"Expert 커트오프: {config.account.expert_cutoff}점")
    print(f"확장 유니버스: {config.expand_universe}")
    print(f"Polygon API 키: {'설정됨' if config.api_keys.polygon_api_key else '없음'}")
    
    warnings = config.validate_all()
    if warnings:
        print(f"\n경고: {len(warnings)}개")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✅ 모든 설정이 정상입니다")
