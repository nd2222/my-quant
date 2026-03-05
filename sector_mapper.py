"""
섹터 매핑 및 정규화 모듈 (V24.9)
- yfinance 섹터명 정규화
- 한글 섹터명 매핑
- 섹터별 스코어링 설정
- 섹터별 비중 제한
"""

# ================= [섹터 정규화 딕셔너리] =================
SECTOR_NORMALIZE = {
    'Financials': 'Financial Services',
    'Financial': 'Financial Services',
    'Finance': 'Financial Services',
    'Banking': 'Financial Services',
    'Insurance': 'Financial Services',
    'Tech': 'Technology',
    'Information Technology': 'Technology',
    'IT': 'Technology',
    'Software': 'Technology',
    'Hardware': 'Technology',
    'Semi': 'Semiconductors',
    'Semiconductor': 'Semiconductors',
    'Chip': 'Semiconductors',
    'Health': 'Healthcare',
    'Pharma': 'Healthcare',
    'Pharmaceuticals': 'Healthcare',
    'Biotech': 'Healthcare',
    'Medical': 'Healthcare',
    'Oil & Gas': 'Energy',
    'Oil': 'Energy',
    'Gas': 'Energy',
    'Renewable Energy': 'Energy',
    'Consumer': 'Consumer Cyclical',
    'Retail': 'Consumer Cyclical',
    'Automotive': 'Consumer Cyclical',
    'Food': 'Consumer Defensive',
    'Beverage': 'Consumer Defensive',
    'Household': 'Consumer Defensive',
    'Industrial': 'Industrials',
    'Manufacturing': 'Industrials',
    'Aerospace': 'Industrials',
    'Defense': 'Industrials',
    'Construction': 'Industrials',
    'Materials': 'Basic Materials',
    'Mining': 'Basic Materials',
    'Steel': 'Basic Materials',
    'Chemical': 'Basic Materials',
    'Telecom': 'Communication Services',
    'Media': 'Communication Services',
    'Entertainment': 'Communication Services',
    'Real Estate': 'Real Estate',
    'REIT': 'Real Estate',
    'Utilities': 'Utilities',
    'Electric': 'Utilities',
    'Water': 'Utilities',
    'None': 'Default',
    '': 'Default',
    None: 'Default',
}

# ================= [한글 섹터명 매핑] =================
SECTOR_KOR = {
    'Technology': '기술 (IT)',
    'Semiconductors': '반도체',
    'Financial Services': '금융',
    'Energy': '에너지',
    'Healthcare': '헬스케어',
    'Consumer Cyclical': '경기소비재',
    'Consumer Defensive': '필수소비재',
    'Industrials': '산업재',
    'Basic Materials': '원자재',
    'Communication Services': '통신',
    'Utilities': '유틸리티',
    'Real Estate': '부동산',
    'Default': '기타'
}

# ================= [섹터별 스코어링 설정] =================
SECTOR_SCORING = {
    'Technology': { 
        'weights': {'growth': 30, 'roe': 15, 'rs': 25, 'momentum': 20, 'debt': 10}, 
        'thresh': {'growth': 20, 'roe': 15, 'per_max': 50, 'pbr_max': 15, 'debt_max': 100} 
    },
    'Semiconductors': { 
        'weights': {'growth': 35, 'roe': 15, 'rs': 25, 'momentum': 15, 'debt': 10}, 
        'thresh': {'growth': 25, 'roe': 20, 'per_max': 60, 'pbr_max': 20, 'debt_max': 80} 
    },
    'Financial Services': { 
        'weights': {'roe': 35, 'pbr': 20, 'div': 20, 'debt': 15, 'rs': 10}, 
        'thresh': {'roe': 12, 'pbr_max': 1.5, 'div_min': 3.0, 'debt_max': 300} 
    },
    'Energy': { 
        'weights': {'div': 30, 'roe': 20, 'debt': 20, 'pbr': 10, 'rs': 20}, 
        'thresh': {'roe': 10, 'div_min': 4.0, 'per_max': 12, 'pbr_max': 2.5, 'debt_max': 150} 
    },
    'Healthcare': { 
        'weights': {'roe': 25, 'growth': 20, 'div': 15, 'rs': 15, 'debt': 5}, 
        'thresh': {'roe': 15, 'growth': 10, 'per_max': 30, 'debt_max': 120} 
    },
    'Consumer Cyclical': { 
        'weights': {'roe': 25, 'growth': 25, 'rs': 20, 'momentum': 15, 'debt': 10}, 
        'thresh': {'roe': 18, 'growth': 12, 'per_max': 40, 'pbr_max': 10, 'debt_max': 120} 
    },
    'Consumer Defensive': {
        'weights': {'roe': 20, 'div': 25, 'debt': 10, 'rs': 20, 'growth': 10},
        'thresh': {'roe': 15, 'div_min': 2.5, 'per_max': 25, 'debt_max': 100}
    },
    'Industrials': {
        'weights': {'roe': 25, 'growth': 20, 'rs': 20, 'debt': 10, 'momentum': 10},
        'thresh': {'roe': 12, 'growth': 10, 'per_max': 30, 'debt_max': 120}
    },
    'Basic Materials': {
        'weights': {'roe': 20, 'pbr': 20, 'div': 15, 'rs': 20, 'debt': 10},
        'thresh': {'roe': 10, 'pbr_max': 2.0, 'div_min': 2.0, 'debt_max': 150}
    },
    'Communication Services': {
        'weights': {'growth': 25, 'roe': 20, 'rs': 20, 'momentum': 15, 'debt': 10},
        'thresh': {'growth': 15, 'roe': 12, 'per_max': 35, 'debt_max': 120}
    },
    'Utilities': {
        'weights': {'div': 35, 'roe': 15, 'debt': 15, 'rs': 10, 'pbr': 10},
        'thresh': {'div_min': 3.5, 'roe': 8, 'pbr_max': 2.0, 'debt_max': 200}
    },
    'Real Estate': {
        'weights': {'div': 30, 'pbr': 20, 'roe': 15, 'debt': 20, 'rs': 10},
        'thresh': {'div_min': 3.0, 'pbr_max': 1.5, 'roe': 8, 'debt_max': 150}
    },
    'Default': { 
        'weights': {'roe': 25, 'growth': 15, 'pbr': 15, 'rs': 20, 'debt': 10}, 
        'thresh': {'roe': 15, 'growth': 10, 'per_max': 25, 'pbr_max': 4, 'debt_max': 150} 
    }
}

# ================= [섹터별 비중 제한] =================
SECTOR_LIMITS = {
    'Semiconductors': 0.30,
    'Energy': 0.20,
    'Technology': 0.30,
    'Financial Services': 0.20,
    'Healthcare': 0.20,
    'Consumer Cyclical': 0.20,
    'Industrials': 0.15,
    'Basic Materials': 0.15,
    'Communication Services': 0.15,
    'Utilities': 0.10,
    'Real Estate': 0.10,
    'Consumer Defensive': 0.15,
}

# ================= [편의 함수] =================
# 대소문자 무시 조회를 위한 최적화 딕셔너리
SECTOR_NORMALIZE_UPPER = {k.upper(): v for k, v in SECTOR_NORMALIZE.items() if k}

def normalize_sector(sector: str) -> str:
    """섹터명 정규화 (O(1) 조회)"""
    if sector is None:
        return 'Default'
    sector_upper = sector.strip().upper()
    return SECTOR_NORMALIZE_UPPER.get(sector_upper, 'Default')

def get_sector_limit(sector: str) -> float:
    """섹터별 비중 제한 조회"""
    sector = normalize_sector(sector)
    return SECTOR_LIMITS.get(sector, 0.15)

def get_sector_config(sector: str) -> dict:
    """섹터별 스코어링 설정 조회"""
    sector = normalize_sector(sector)
    return SECTOR_SCORING.get(sector, SECTOR_SCORING['Default'])

# ================= [클래스 버전 (객체지향)] =================
class SectorMapper:
    """섹터 매핑 클래스 (expert.py, rebalance.py 용)"""
    
    def __init__(self):
        self.normalize = normalize_sector
        self.get_limit = get_sector_limit
        self.get_config = get_sector_config
    
    def normalize_sector(self, sector: str) -> str:
        return normalize_sector(sector)
    
    def get_sector_limit(self, sector: str) -> float:
        return get_sector_limit(sector)
    
    def get_sector_config(self, sector: str) -> dict:
        return get_sector_config(sector)
