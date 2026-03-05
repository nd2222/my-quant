"""
RS (Relative Strength) 전략 모듈
- Keller 의 13612W 모멘텀 공식 구현
- 시장 대비 상대적 강도 계산
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_momentum_13612(prices: pd.Series) -> float:
    """
    Keller 의 13612W 모멘텀 계산
    
    공식:
        Momentum = (12 × 1 개월수익률) + (4 × 3 개월수익률) + (2 × 6 개월수익률) + (1 × 12 개월수익률)
    
    Args:
        prices: 일별 종가 시리즈 (최소 252 일 데이터)
    
    Returns:
        가중 모멘텀 점수
    """
    if len(prices) < 252:
        # 데이터 부족시 단순 63 일 수익률 사용
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


def calculate_rs_score(ticker: str, df: pd.DataFrame, benchmark_ret: float, period_days: int = 63) -> dict:
    """
    RS (상대강도) 점수 계산
    
    Args:
        ticker: 종목 티커
        df: 일별 가격 데이터 (Close 컬럼 필요)
        benchmark_ret: 벤치마크 (S&P500) 동기간 수익률
        period_days: 계산 기간 (기본 63 일)
    
    Returns:
        dict: {
            'score': RS 점수 (0-100),
            'stock_ret': 종목 수익률,
            'diff': 시장 대비 초과수익률,
            'momentum_13612': 13612W 모멘텀
        }
    """
    if df.empty or len(df) < period_days:
        return {'score': 50, 'stock_ret': 0, 'diff': 0, 'momentum_13612': 0}
    
    try:
        # 1. 단순 63 일 수익률 (기존 방식)
        curr = df['Close'].iloc[-1]
        old = df['Close'].iloc[-period_days]
        
        if hasattr(curr, 'item'):
            curr = curr.item()
        if hasattr(old, 'item'):
            old = old.item()
        
        stock_ret = (curr / old - 1) * 100
        diff = stock_ret - benchmark_ret
        
        # RS 점수: 50 점이 중립, 시장 대비 1% 초과시 +2 점
        score = 50 + (diff * 2.0)
        score = max(0, min(100, score))
        
        # 2. 13612W 모멘텀 (새로운 방식)
        momentum_13612 = calculate_momentum_13612(df['Close'])
        
        return {
            'score': round(score, 1),
            'stock_ret': round(stock_ret, 1),
            'diff': round(diff, 1),
            'momentum_13612': round(momentum_13612, 1)
        }
    
    except Exception as e:
        logger.error(f"RS 계산 실패 ({ticker}): {e}")
        return {'score': 50, 'stock_ret': 0, 'diff': 0, 'momentum_13612': 0}


def calculate_enhanced_rs_score(ticker: str, df: pd.DataFrame, benchmark_ret: float, benchmark_momentum: float) -> dict:
    """
    향상된 RS 점수 계산 (13612W 모멘텀 + 단순 RS 결합)
    
    Args:
        ticker: 종목 티커
        df: 일별 가격 데이터
        benchmark_ret: 벤치마크 63 일 수익률
        benchmark_momentum: 벤치마크 13612W 모멘텀
    
    Returns:
        dict: {
            'score': 종합 RS 점수 (0-100),
            'stock_ret': 종목 63 일 수익률,
            'diff': 시장 대비 초과수익률,
            'momentum_13612': 종목 13612W 모멘텀,
            'momentum_diff': 모멘텀 차이
        }
    """
    if df.empty or len(df) < 63:
        return {
            'score': 50, 'stock_ret': 0, 'diff': 0,
            'momentum_13612': 0, 'momentum_diff': 0
        }
    
    try:
        # 1. 단순 RS 계산
        rs_simple = calculate_rs_score(ticker, df, benchmark_ret, 63)
        
        # 2. 13612W 모멘텀 계산
        momentum_13612 = calculate_momentum_13612(df['Close'])
        momentum_diff = momentum_13612 - benchmark_momentum
        
        # 3. 종합 점수 (단순 RS 40% + 모멘텀 60%)
        # 모멘텀 점수: 50 점이 중립, 모멘텀 차이 1% 당 +2 점
        momentum_score = 50 + (momentum_diff * 2.0)
        momentum_score = max(0, min(100, momentum_score))
        
        # 가중 평균
        final_score = (rs_simple['score'] * 0.4) + (momentum_score * 0.6)
        
        return {
            'score': round(final_score, 1),
            'stock_ret': rs_simple['stock_ret'],
            'diff': rs_simple['diff'],
            'momentum_13612': round(momentum_13612, 1),
            'momentum_diff': round(momentum_diff, 1)
        }
    
    except Exception as e:
        logger.error(f"향상된 RS 계산 실패 ({ticker}): {e}")
        return {
            'score': 50, 'stock_ret': 0, 'diff': 0,
            'momentum_13612': 0, 'momentum_diff': 0
        }
