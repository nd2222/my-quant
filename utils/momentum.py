"""
모멘텀 계산 유틸리티
- Keller 13612W 모멘텀 공식
- 단일 소스 (중복 제거)
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_momentum_13612(prices: pd.Series) -> float:
    """
    Keller 의 13612W 모멘텀 계산
    
    공식:
        Momentum = (12 × 1 개월수익률) + (4 × 3 개월수익률) + (2 × 6 개월수익률) + (1 × 12 개월수익률)
        정규화: 합계 / 19
    
    Args:
        prices: 일별 종가 시리즈 (최소 252 일 데이터)
    
    Returns:
        float: 가중 모멘텀 점수 (%)
    """
    if len(prices) < 252:
        # 데이터 부족시 단순 63 일 수익률
        if len(prices) >= 63:
            return (prices.iloc[-1] / prices.iloc[-63] - 1) * 100
        return 0.0
    
    try:
        # 각 기간별 수익률 계산
        ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100    # 1 개월 (21 일)
        ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100    # 3 개월 (63 일)
        ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100   # 6 개월 (126 일)
        ret_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) * 100  # 12 개월 (252 일)
        
        # 가중치 적용 (12:4:2:1)
        momentum = (12 * ret_1m) + (4 * ret_3m) + (2 * ret_6m) + (1 * ret_12m)
        
        # 가중치 합으로 정규화 (12+4+2+1 = 19)
        momentum /= 19
        
        return momentum
    
    except Exception as e:
        logger.warning(f"모멘텀 계산 실패, 단순 63 일 수익률 사용: {e}")
        if len(prices) >= 63:
            return (prices.iloc[-1] / prices.iloc[-63] - 1) * 100
        return 0.0
