"""
Early Warning System 모듈
- 보유 종목 선제 교체 신호 감지
- 4 가지 신호 → 복합 긴급도 점수
- 이미 다운받은 데이터 재사용 (중복 다운로드 방지)
"""
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class EarlyWarningSystem:
    """
    V25.0: 보유 종목 선제 교체 신호 감지
    
    4 가지 신호:
        1. 모멘텀 둔화 (3 주 연속 13612W 하락)
        2. RS 역전 (더 강한 종목 출현)
        3. 섹터 로테이션 (자금 이탈)
        4. 거래량 스파이크 (이상 징후)
    
    출력:
        - 긴급도 점수 (0-100)
        - 교체 후보 종목
        - 모멘텀 추이
    """
    
    def __init__(self):
        self.checkpoints = [-1, -6, -11, -16]  # 거래일 기준 (현재, 1 주전, 2 주전, 3 주전)
    
    def _check_momentum_decay(self, ticker: str, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        신호 1: 모멘텀 둔화 감지 (3 주 연속 13612W 모멘텀 하락)
        
        Args:
            ticker: 종목 티커
            df: 가격 데이터 (이미 다운로드된 것 재사용)
        
        Returns:
            Tuple[int, List[str]]: (점수, 모멘텀 추이)
        """
        momentums = []
        
        for cp in self.checkpoints:
            if len(df) < abs(cp):
                momentums.append(0)
                continue
            
            prices = df['Close'][:cp]
            if len(prices) < 21:
                momentums.append(0)
            else:
                ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100
                ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100
                ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100
                ret_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) * 100 if len(prices) >= 252 else 0
                
                # Keller 13612W 공식
                m = (12 * ret_1m + 4 * ret_3m + 2 * ret_6m + ret_12m) / 19
                momentums.append(m)
        
        # 3 주 연속 하락 여부
        decay_weeks = sum(1 for i in range(3) if momentums[i] < momentums[i+1])
        score = decay_weeks * 25  # 최대 75 점
        trend = [f"{m:+.1f}%" for m in momentums]
        
        if score >= 50:
            logger.debug(f"  모멘텀 둔화 감지: {ticker} ({score}점)")
        
        return score, trend
    
    def _check_rs_reversal(self, ticker: str, my_rs_score: int, 
                           all_rs_map: Dict[str, int]) -> Tuple[int, List[str]]:
        """
        신호 2: RS 역전 감지 (내 종목보다 RS 높은 신규 후보 N 개 이상)
        
        Args:
            ticker: 종목 티커
            my_rs_score: 내 종목 RS 점수
            all_rs_map: 전체 종목 RS 점수 dict
        
        Returns:
            Tuple[int, List[str]]: (점수, 교체 후보 목록)
        """
        # 나보다 RS 10 점 이상 높은 종목들
        stronger = [
            t for t, rs in all_rs_map.items()
            if rs > my_rs_score + 10
        ]
        stronger.sort(key=lambda x: all_rs_map[x], reverse=True)
        
        # 공격적: 3 개 이상 출현시 교체 신호
        score = min(100, len(stronger) * 5)
        
        if score >= 15:
            logger.debug(f"  RS 역전 감지: {ticker} ({len(stronger)}개 후보)")
        
        return score, stronger[:5]  # 상위 5 개 후보 반환
    
    def _check_sector_rotation(self, my_sector: str, 
                               sector_momentum_map: Dict[str, float]) -> Tuple[int, int, int]:
        """
        신호 3: 섹터 로테이션 감지 (섹터별 모멘텀 순위)
        
        Args:
            my_sector: 내 섹터
            sector_momentum_map: 섹터별 모멘텀 dict
        
        Returns:
            Tuple[int, int, int]: (점수, 내 순위, 총 섹터 수)
        """
        if not sector_momentum_map or my_sector not in sector_momentum_map:
            return 0, 0, 0
        
        all_sectors = sorted(sector_momentum_map, key=lambda x: sector_momentum_map[x], reverse=True)
        my_rank = all_sectors.index(my_sector) + 1
        rank_pct = my_rank / len(all_sectors)
        
        # 하위 30% 면 섹터 자금 이탈 신호
        score = int(rank_pct * 100)
        
        if score >= 70:
            logger.debug(f"  섹터 로테이션 감지: {my_sector} ({my_rank}위/{len(all_sectors)})")
        
        return score, my_rank, len(all_sectors)
    
    def _check_volume_spike(self, ticker: str, df: pd.DataFrame) -> Tuple[int, float]:
        """
        신호 4: 거래량 스파이크 감지 (이상 징후)
        
        Args:
            ticker: 종목 티커
            df: 가격 데이터 (이미 다운로드된 것 재사용)
        
        Returns:
            Tuple[int, float]: (점수, 거래량 비율)
        """
        if 'Volume' not in df.columns or len(df) < 60:
            return 0, 0.0
        
        # 20 일 평균 거래량
        vol_avg = df['Volume'].iloc[-21:-1].mean()
        current_vol = df['Volume'].iloc[-1]
        
        if vol_avg == 0:
            return 0, 0.0
        
        vol_ratio = current_vol / vol_avg
        
        # 3 배 이상이면 이상 징후
        score = min(100, int((vol_ratio - 1) * 25))
        
        if vol_ratio >= 3.0:
            logger.debug(f"  거래량 스파이크: {ticker} ({vol_ratio:.1f}x)")
        
        return score, vol_ratio
    
    def calculate_urgency(self, ticker: str, holding: Dict, 
                         all_rs_map: Dict[str, int],
                         sector_momentum_map: Dict[str, float],
                         df: pd.DataFrame) -> Dict:
        """
        종합 긴급도 계산 (4 가지 신호 통합)
        
        Args:
            ticker: 종목 티커
            holding: 보유 종목 정보 (rs_score, sector 등 포함)
            all_rs_map: 전체 종목 RS 점수 dict
            sector_momentum_map: 섹터별 모멘텀 dict
            df: 가격 데이터 (이미 다운로드된 것 재사용)
        
        Returns:
            Dict: {
                'urgency': int,  # 종합 긴급도 (0-100)
                'momentum_decay': int,  # 모멘텀 둔화 점수
                'rs_reversal': int,  # RS 역전 점수
                'sector_rotation': int,  # 섹터 로테이션 점수
                'volume_spike': int,  # 거래량 스파이크 점수
                'momentum_trend': List[str],  # 모멘텀 추이
                'replace_candidates': List[str],  # 교체 후보
                'sector_rank': int,  # 섹터 순위
                'vol_ratio': float  # 거래량 비율
            }
        """
        # 1. 모멘텀 둔화
        momentum_score, momentum_trend = self._check_momentum_decay(ticker, df)
        
        # 2. RS 역전
        my_rs_score = holding.get('rs_score', 50)
        rs_score, candidates = self._check_rs_reversal(ticker, my_rs_score, all_rs_map)
        
        # 3. 섹터 로테이션
        my_sector = holding.get('sector', 'Default')
        sector_score, sector_rank, total_sectors = self._check_sector_rotation(
            my_sector, sector_momentum_map
        )
        
        # 4. 거래량 스파이크
        volume_score, vol_ratio = self._check_volume_spike(ticker, df)
        
        # 종합 긴급도 (가중 평균)
        # 모멘텀 40% + RS 30% + 섹터 20% + 거래량 10%
        urgency = int(
            momentum_score * 0.4 +
            rs_score * 0.3 +
            sector_score * 0.2 +
            volume_score * 0.1
        )
        
        # 긴급도 등급
        if urgency >= 70:
            level = "🔴 즉시 교체"
        elif urgency >= 40:
            level = "🟡 교체 검토"
        else:
            level = "🟢 유지"
        
        logger.debug(f"  긴급도: {ticker} = {urgency}점 ({level})")
        
        return {
            'urgency': urgency,
            'momentum_decay': momentum_score,
            'rs_reversal': rs_score,
            'sector_rotation': sector_score,
            'volume_spike': volume_score,
            'momentum_trend': momentum_trend,
            'replace_candidates': candidates,
            'sector_rank': sector_rank,
            'total_sectors': total_sectors,
            'vol_ratio': vol_ratio,
            'level': level
        }
    
    def get_urgency_html(self, ticker: str, urgency_data: Dict) -> str:
        """
        긴급도 HTML 생성
        
        Args:
            ticker: 종목 티커
            urgency_data: calculate_urgency() 결과
        
        Returns:
            str: HTML 문자열
        """
        urgency = urgency_data['urgency']
        level = urgency_data['level']
        
        if urgency >= 70:
            color = "#e74c3c"
            icon = "🔴"
        elif urgency >= 40:
            color = "#f1c40f"
            icon = "🟡"
        else:
            color = "#2ecc71"
            icon = "🟢"
        
        html = f"""
        <div style='background:{color}20;padding:10px;border-radius:6px;margin:5px 0'>
            <div style='font-weight:bold;color:{color}'>{icon} 긴급도: {urgency}점 ({level})</div>
            <div style='font-size:0.85em;margin-top:5px'>
                모멘텀: {urgency_data['momentum_decay']}점 | 
                RS: {urgency_data['rs_reversal']}점 | 
                섹터: {urgency_data['sector_rotation']}점 | 
                거래량: {urgency_data['volume_spike']}점
            </div>
            <div style='font-size:0.85em;margin-top:3px'>
                모멘텀 추이: {' → '.join(urgency_data['momentum_trend'])}
            </div>
            <div style='font-size:0.85em;margin-top:3px'>
                섹터 순위: {urgency_data['sector_rank']}/{urgency_data['total_sectors']} |
                거래량: {urgency_data['vol_ratio']:.1f}x
            </div>
            {'<div style=\"font-size:0.85em;margin-top:3px;color:#e74c3c\">교체 후보: ' + ', '.join(urgency_data['replace_candidates']) + '</div>' if urgency_data['replace_candidates'] else ''}
        </div>
        """
        
        return html.strip()


# 편의 함수
def get_early_warning_system() -> EarlyWarningSystem:
    """EarlyWarningSystem 단일 인스턴스 생성"""
    return EarlyWarningSystem()
