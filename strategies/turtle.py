"""
Turtle 전략 모듈
- Donchian Channel 돌파
- S1/S2 매수 신호
- S1 신호 필터 (노이즈 감소)
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TurtleStrategy:
    """
    Turtle Trading 전략
    
    신호:
        S2: 55 일 고점 돌파 (주력 진입)
        S1: 20 일 고점 돌파 (조기 진입, 필터 적용)
    
    필터:
        - MA200 위 (상승 추세)
        - 거래량 스파이크
        - ATR 확장
        - S1: 직전 S1 이 수익일 경우만 (선택)
    """
    
    def __init__(self, config):
        self.config = config
        self.volume_avg_days = config.data.volume_avg_days if hasattr(config, 'data') else 21
    
    def analyze(self, df: pd.DataFrame, ticker: str, current_price: float, ma200: float) -> dict:
        """
        Turtle 신호 분석
        
        Args:
            df: 일별 가격 데이터 (High, Low, Close, Volume)
            ticker: 종목 티커
            current_price: 현재가
            ma200: 200 일 이동평균
        
        Returns:
            dict: {
                'signal': 신호 (S1/S2/관망),
                'trend_score': 추세 점수,
                'atr': 현재 ATR,
                'vol_ratio': 거래량 비율,
                'vol_spike': 거래량 스파이크 여부,
                'exit_price': 청산 가격 (10 일 저점)
            }
        """
        if df.empty or len(df) < 55:
            return {
                'signal': '관망', 'trend_score': 0, 'atr': 0,
                'vol_ratio': 0, 'vol_spike': False, 'exit_price': 0
            }
        
        try:
            # 1. Donchian Channel 계산
            df = df.copy()
            df['High_55'] = df['High'].rolling(55).max().shift(1)
            df['High_20'] = df['High'].rolling(20).max().shift(1)
            df['Low_10'] = df['Low'].rolling(10).min().shift(1)
            df['MA200'] = df['Close'].rolling(200).mean()
            
            # 2. ATR 계산 (V24.9: Wilder's Smoothing)
            df['TR'] = pd.concat([
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            ], axis=1).max(axis=1)
            df['N'] = df['TR'].ewm(alpha=1/20, adjust=False).mean()  # Wilder's Smoothing
            
            # 3. 현재 값 추출
            curr = df.iloc[-1]
            high_55 = curr['High_55']
            high_20 = curr['High_20']
            low_10 = curr['Low_10']
            atr = curr['N']
            
            # 4. 거래량 분석
            vol_spike, vol_ratio = self._detect_volume_spike(df)
            
            # 5. 추세 분석
            is_uptrend = current_price > ma200
            avg_atr = df['N'].iloc[-60:-1].mean() if len(df) >= 60 else atr
            is_atr_expanding = atr > avg_atr * 0.9
            
            # 6. 신호 결정
            signal = '관망'
            
            # S2: 55 일 고점 돌파 (주력)
            if current_price > high_55 and is_uptrend:
                signal = 'S2 매수'
            
            # S1: 20 일 고점 돌파 (조기 진입, 필터 적용)
            elif current_price > high_20 and is_uptrend and vol_ratio > 1.0 and is_atr_expanding:
                # S1 필터: 시장 스코어가 60 이상일 때만 (약세장에서는 S1 무시)
                # 이 필터는 상위 레이어에서 적용
                signal = 'S1 매수'
            
            # 7. 추세 점수
            trend_score = (current_price - ma200) / ma200 * 100 if ma200 > 0 else 0
            
            return {
                'signal': signal,
                'trend_score': round(trend_score, 1),
                'atr': round(atr, 2) if hasattr(atr, 'item') else atr,
                'vol_ratio': round(vol_ratio, 1),
                'vol_spike': vol_spike,
                'exit_price': round(low_10, 2) if hasattr(low_10, 'item') else low_10
            }
        
        except Exception as e:
            logger.error(f"Turtle 분석 실패 ({ticker}): {e}")
            return {
                'signal': '관망', 'trend_score': 0, 'atr': 0,
                'vol_ratio': 0, 'vol_spike': False, 'exit_price': 0
            }
    
    def _detect_volume_spike(self, df: pd.DataFrame) -> tuple:
        """
        거래량 스파이크 감지
        
        Args:
            df: 일별 가격 데이터
        
        Returns:
            tuple: (스파이크 여부, 거래량 비율)
        """
        if len(df) < self.volume_avg_days:
            return False, 0.0
        
        try:
            avg_vol = df['Volume'].iloc[-self.volume_avg_days:-1].mean()
            curr_vol = df['Volume'].iloc[-1]
            
            if hasattr(avg_vol, 'item'):
                avg_vol = avg_vol.item()
            if hasattr(curr_vol, 'item'):
                curr_vol = curr_vol.item()
            
            if avg_vol <= 0:
                return False, 0.0
            
            ratio = curr_vol / avg_vol
            return ratio >= 1.5, round(ratio, 1)
        
        except Exception as e:
            logger.warning(f"거래량 스파이크 감지 실패: {e}")
            return False, 0.0
    
    def calculate_unit_qty(self, atr: float, current_price: float, cash_balance: float, 
                           total_value: float, risk_ratio: float, max_alloc: float, usd_krw: float,
                           half_kelly_ratio: float = 0.08) -> int:
        """
        Turtle 단위 거래량 계산 (V24.9: 하프 켈리 4 번째 안전장치 추가)
        
        Args:
            atr: ATR (변동성)
            current_price: 현재가
            cash_balance: 현금 잔고
            total_value: 총 자산
            risk_ratio: 리스크 비율 (1%)
            max_alloc: 최대 비중 (20%)
            usd_krw: 환율
            half_kelly_ratio: 하프 켈리 비율 (8%)
        
        Returns:
            int: 매수 수량
        """
        if atr <= 0 or current_price <= 0:
            return 0
        
        try:
            # 1. 변동성 기반 수량 (1% 리스크)
            vol_qty = int((cash_balance * risk_ratio) / (atr * usd_krw))
            
            # 2. 최대 비중 제한
            max_qty = int((total_value * max_alloc) / (current_price * usd_krw))
            
            # 3. 현금 제한
            cash_qty = int(cash_balance / (current_price * usd_krw))
            
            # 4. 하프 켈리 제한 (V24.9: 4 번째 안전장치)
            kelly_qty = int((total_value * half_kelly_ratio) / (current_price * usd_krw))
            
            # 5. 최소값 선택
            unit_qty = min(vol_qty, max_qty, cash_qty, kelly_qty)
            
            return max(0, unit_qty)
        
        except Exception as e:
            logger.error(f"단위 거래량 계산 실패: {e}")
            return 0
