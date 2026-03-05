"""
Expert (우량주) 전략 모듈
- 섹터별 스코어링
- 펀더멘털 + 모멘텀 결합
"""
import logging

logger = logging.getLogger(__name__)


class ExpertStrategy:
    """
    Expert+ 우량주 전략
    
    스코어링 요소:
        - ROE (자본수익률)
        - 성장률 (매출성장률)
        - 배당수익률
        - PBR/PER (밸류에이션)
        - RS (상대강도)
        - 부채비율
    
    섹터별 가중치 적용
    """
    
    def __init__(self, config, sector_mapper):
        self.config = config
        self.sector_mapper = sector_mapper
        self.expert_cutoff = config.account.expert_cutoff if hasattr(config, 'account') else 70
    
    def calculate_score(self, ticker: str, fund_data: dict, price_data: dict) -> tuple:
        """
        섹터별 가중치 스코어링
        
        Args:
            ticker: 종목 티커
            fund_data: 펀더멘털 데이터
                {roe, growth, div, pbr, per, debt_ratio, sector}
            price_data: 가격 데이터
                {rs_score, momentum}
        
        Returns:
            tuple: (점수, 섹터명)
        """
        # 1. 섹터 정규화
        sector = fund_data.get('sector', 'Default')
        sector = self.sector_mapper.normalize_sector(sector)
        
        # SOX 종목은 반도체로 처리
        if hasattr(self, 'sox_tickers') and ticker in self.sox_tickers:
            sector = 'Semiconductors'
        
        # 2. 섹터별 설정 가져오기
        sector_config = self.sector_mapper.get_sector_config(sector)
        weights = sector_config['weights']
        thresh = sector_config['thresh']
        
        # 3. 스코어링
        score = 0
        
        # ROE
        roe = fund_data.get('roe', 0)
        if roe >= thresh.get('roe', 15):
            score += weights.get('roe', 0)
        elif roe > 0:
            score += weights.get('roe', 0) * 0.5
        
        # 성장률
        growth = fund_data.get('growth', 0)
        if 'growth' in weights:
            if growth >= thresh.get('growth', 10):
                score += weights['growth']
            elif growth > 0:
                score += weights['growth'] * 0.5
        
        # 배당
        div = fund_data.get('div', 0)
        if 'div' in weights:
            if div >= thresh.get('div_min', 2.0):
                score += weights['div']
            elif div > 0:
                score += weights['div'] * 0.5
        
        # PBR
        pbr = fund_data.get('pbr', 0)
        if 'pbr' in weights:
            if 0 < pbr <= thresh.get('pbr_max', 3.0):
                score += weights['pbr']
        
        # PER (과고점 감점)
        per = fund_data.get('per', 0)
        if per > thresh.get('per_max', 30) * 1.5:
            score -= 5
        
        # 부채비율 (V24.9 보너스 방식: 정상이면 보너스, 높으면 점수 없음)
        debt = fund_data.get('debt_ratio', 0)
        debt_weight = abs(weights.get('debt', 10))
        if debt <= thresh.get('debt_max', 150):
            # 부채비율 정상: 만점
            score += debt_weight
        elif debt <= thresh.get('debt_max', 150) * 2:
            # 부채비율 약간 높음: 절반 점수
            score += debt_weight * 0.5
        # 부채비율 매우 높음: 점수 없음 (감점)
        
        # RS (상대강도)
        rs_score = price_data.get('rs_score', 50)
        score += weights.get('rs', 20) * (rs_score / 100.0)
        
        # 모멘텀
        if 'momentum' in weights:
            mom = price_data.get('momentum', 0)
            if mom > 10:
                score += weights['momentum']
            elif mom > 0:
                score += weights['momentum'] * 0.5
        
        # 4. 점수 제한 (0-100)
        score = max(0, min(100, score))
        
        return round(score, 1), sector
    
    def get_max_qty(self, current_price: float, total_value: float, 
                    cash_balance: float, max_alloc: float, usd_krw: float) -> int:
        """
        최대 매수 수량 계산
        
        Args:
            current_price: 현재가
            total_value: 총 자산
            cash_balance: 현금 잔고
            max_alloc: 최대 비중
            usd_krw: 환율
        
        Returns:
            int: 최대 수량
        """
        try:
            max_qty = int((total_value * max_alloc) / (current_price * usd_krw))
            cash_qty = int(cash_balance / (current_price * usd_krw))
            return min(max_qty, cash_qty)
        except Exception as e:
            logger.error(f"최대 수량 계산 실패: {e}")
            return 0
