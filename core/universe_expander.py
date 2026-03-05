"""
유니버스 확장 모듈
- SOXX 반도체 종목 자동 수집
- 러셀 2000 소형주 추가
- 테마 종목 (바이오/사이버보안/클라우드)
"""
import requests
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ================= [하드코딩 폴백 - SOXX] =================
SOXX_FALLBACK = [
    'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC',
    'ADI', 'MCHP', 'NXPI', 'MRVL', 'ON', 'MPWR', 'SWKS', 'QRVO', 'TER', 'ENTG',
    'ALGM', 'WOLF', 'COHR', 'RMBS', 'LSCC', 'SLAB', 'CRUS', 'SITM', 'HIMX', 'UCTT'
]

# ================= [테마 종목 - S&P/나스닥 미포함 성장주] =================
THEMATIC_TICKERS = {
    '사이버보안': ['CRWD', 'ZS', 'PANW', 'S', 'CYBR', 'FTNT', 'CHKP', 'GEN'],
    '클라우드': ['NET', 'DDOG', 'SNOW', 'MDB', 'GTLB', 'ESTC', 'CFLT', 'PATH'],
    '바이오텍': ['RXRX', 'BEAM', 'CRSP', 'NTLA', 'EDIT', 'VERV', 'SGMO', 'BLUE'],
    '핀테크': ['SOFI', 'AFRM', 'UPST', 'HOOD', 'COIN', 'PYPL', 'SQ', 'MELI'],
    '에너지전환': ['ENPH', 'SEDG', 'RUN', 'ARRY', 'FSLR', 'PLUG', 'BLDP', 'CLNE'],
}

# ================= [러셀 2000 IWM ETF 상위] =================
# IWM 은 2000 개 종목이므로 시총 상위 200 개만 추출 (API 로 동적 수집)


def get_soxx_tickers() -> List[str]:
    """
    iShares SOXX ETF 구성종목 자동 수집
    
    Returns:
        List[str]: 반도체 종목 티커 리스트
    """
    # 방법 1: iShares 공식 JSON (가장 신뢰도 높음)
    try:
        url = "https://www.ishares.com/us/products/239705/ISHARES-PHLX-SEMICONDUCTOR-ETF/1467271812596.ajax?tab=holdings&fileType=json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # JSON 구조: aaData = [[ticker, name, weight, ...], ...]
        if 'aaData' in data:
            tickers = []
            for item in data['aaData']:
                if len(item) > 0 and item[0] not in ['-', '', 'Cash']:
                    ticker = item[0].strip()
                    if ticker and len(ticker) <= 5:  # 유효한 티커만
                        tickers.append(ticker)
            
            if tickers:
                logger.info(f"✅ SOXX 자동 수집: {len(tickers)}개 종목")
                return tickers[:30]  # 상위 30 개만
        
        logger.warning("SOXX JSON 파싱 실패 - 폴백 사용")
    
    except Exception as e:
        logger.warning(f"SOXX 자동 수집 실패: {e} - 폴백 사용")
    
    # 폴백: 하드코딩 사용
    logger.info(f"⚠️ SOXX 하드코딩 폴백: {len(SOXX_FALLBACK)}개 종목")
    return SOXX_FALLBACK.copy()


def get_iwm_top_tickers(limit: int = 200) -> List[str]:
    """
    iShares IWM (러셀 2000) ETF 상위 종목 수집
    
    Args:
        limit: 수집할 종목 수 (기본 200 개)
    
    Returns:
        List[str]: 러셀 2000 상위 티커 리스트
    """
    try:
        url = "https://www.ishares.com/us/products/239710/IWM/holdings.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if 'aaData' in data:
            tickers = []
            for item in data['aaData']:
                if len(item) > 0 and item[0] not in ['-', '', 'Cash']:
                    ticker = item[0].strip()
                    if ticker and len(ticker) <= 5:
                        tickers.append(ticker)
            
            if tickers:
                logger.info(f"✅ IWM 자동 수집: {len(tickers)}개 중 상위 {limit}개")
                return tickers[:limit]
        
        logger.warning("IWM JSON 파싱 실패")
    
    except Exception as e:
        logger.warning(f"IWM 자동 수집 실패: {e}")
    
    # 폴백: 빈 리스트 (기존 유니버스만 사용)
    logger.info("⚠️ IWM 폴백: 빈 리스트 (기존 유니버스만 사용)")
    return []


def get_thematic_tickers(categories: List[str] = None) -> dict:
    """
    테마 종목 반환
    
    Args:
        categories: 테마 카테고리 리스트 (None 이면 모두)
    
    Returns:
        dict: {테마명: [티커리스트]}
    """
    if categories is None:
        return THEMATIC_TICKERS.copy()
    
    result = {}
    for cat in categories:
        if cat in THEMATIC_TICKERS:
            result[cat] = THEMATIC_TICKERS[cat].copy()
    
    return result


def get_all_extra_tickers(include_russell: bool = True, 
                          include_thematic: bool = True,
                          thematic_categories: List[str] = None) -> List[str]:
    """
    모든 추가 유니버스 통합
    
    Args:
        include_russell: 러셀 2000 포함 여부
        include_thematic: 테마 종목 포함 여부
        thematic_categories: 포함할 테마 카테고리 (None 이면 모두)
    
    Returns:
        List[str]: 통합 추가 유니버스 (중복 제거)
    """
    all_tickers = set()
    
    # 1. SOXX 반도체 (항상 포함)
    soxx = get_soxx_tickers()
    all_tickers.update(soxx)
    logger.info(f"  • SOXX 반도체: {len(soxx)}개")
    
    # 2. 러셀 2000 상위
    if include_russell:
        iwm = get_iwm_top_tickers(limit=200)
        all_tickers.update(iwm)
        logger.info(f"  • 러셀 2000 상위: {len(iwm)}개")
    
    # 3. 테마 종목
    if include_thematic:
        thematic = get_thematic_tickers(thematic_categories)
        thematic_count = 0
        for cat, tickers in thematic.items():
            all_tickers.update(tickers)
            thematic_count += len(tickers)
        logger.info(f"  • 테마 종목: {thematic_count}개")
    
    result = sorted(list(all_tickers))
    logger.info(f"✅ 총 추가 유니버스: {len(result)}개 종목")
    
    return result


# ================= [테스트용 메인] =================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("유니버스 확장 모듈 테스트")
    print("=" * 60)
    
    # SOXX 테스트
    print("\n[1] SOXX 반도체 종목")
    soxx = get_soxx_tickers()
    print(f"  수집됨: {len(soxx)}개")
    print(f"  상위 10 개: {soxx[:10]}")
    
    # IWM 테스트
    print("\n[2] IWM 러셀 2000 상위")
    iwm = get_iwm_top_tickers(limit=10)
    print(f"  수집됨: {len(iwm)}개")
    print(f"  상위 10 개: {iwm[:10]}")
    
    # 테마 테스트
    print("\n[3] 테마 종목")
    thematic = get_thematic_tickers()
    for cat, tickers in thematic.items():
        print(f"  {cat}: {len(tickers)}개 - {tickers[:5]}")
    
    # 통합 테스트
    print("\n[4] 통합 추가 유니버스")
    all_extra = get_all_extra_tickers(include_russell=True, include_thematic=True)
    print(f"  총 {len(all_extra)}개 종목")
    print(f"  상위 20 개: {all_extra[:20]}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
