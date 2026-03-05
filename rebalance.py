"""
리밸런싱 관리 모듈
- 매도 후보 선정
- 매수 시나리오 생성
- 섹터 할당 계산
"""
import logging

logger = logging.getLogger(__name__)


class RebalanceManager:
    """
    포트폴리오 리밸런싱 관리자
    
    기능:
        - 매도 권장 종목 선정 (손절/펀더멘털/모멘텀)
        - 매수 후보 계층화 (Tier 1: 교집합, Tier 2: Expert)
        - 시나리오별 매수 계획 (보수/균형/공격)
    """
    
    def __init__(self, holdings_data: list, macro_analyzer, cash_balance: float, 
                 usd_krw: float, sector_mapper, data_dir=None):
        self.holdings = holdings_data
        self.ma = macro_analyzer
        self.cash = cash_balance
        self.usd_krw = usd_krw
        self.sector_mapper = sector_mapper
        self.sell_candidates = []
        self.buy_candidates = []
        self.scenarios = {}
        # V24.9: sell_score_history 지속성
        self.data_dir = data_dir
        self.sell_score_history = self._load_score_history() if data_dir else {}
    
    def get_current_sector_allocation(self, total_asset: float) -> dict:
        """
        현재 섹터 비중 계산
        
        Args:
            total_asset: 총 자산
        
        Returns:
            dict: {섹터명: 비중}
        """
        allocation = {}
        
        for h in self.holdings:
            # sector 키가 없을 경우 fundamentals 에서 조회
            sector = h.get('sector')
            if not sector:
                sector = 'Default'
            sector = self.sector_mapper.normalize_sector(sector)
            val = h['current_price'] * h['qty'] * self.usd_krw
            allocation[sector] = allocation.get(sector, 0) + val
        
        # 비중으로 변환
        for k in allocation:
            allocation[k] = allocation[k] / total_asset
        
        return allocation
    
    def generate_plan(self, expert_results: list, turtle_results: list, 
                      rs_results: list, total_asset_val: float) -> tuple:
        """
        리밸런싱 계획 생성
        
        Args:
            expert_results: Expert 전략 결과
            turtle_results: Turtle 전략 결과
            rs_results: RS 전략 결과
            total_asset_val: 총 자산 가치
        
        Returns:
            tuple: (매도 후보, 시나리오들)
        """
        logger.info("[7/7] 포트폴리오 리밸런싱 계산 중...")
        
        total_sell_proceeds = 0
        current_alloc = self.get_current_sector_allocation(total_asset_val)
        
        # 1. 매도 후보 선정
        for h in self.holdings:
            score, reasons = self.calculate_sell_score(h)
            action = "KEEP"
            
            if score >= 100:
                action = "즉시 매도 (손절)"
            elif score >= 50:
                action = "매도 권장 (펀더멘털)"
            elif score >= 30:
                action = "교체 검토 (모멘텀)"
            
            if action != "KEEP":
                est_value = h['current_price'] * h['qty'] * self.usd_krw
                self.sell_candidates.append({
                    'ticker': h['ticker'],
                    'action': action,
                    'reason': ", ".join(reasons),
                    'est_value': est_value,
                    'qty': h['qty'],
                    'score': score
                })
                total_sell_proceeds += est_value
                # ✅ 매도 후보만 히스토리 기록 (KEEP 은 기록 안함 - 잘못된 점수 급등 방지)
                self.sell_score_history[h['ticker']] = score
        
        # 2. 매수 후보 계층화
        tier1 = []  # Turtle + Expert 교집합
        t_map = {r['ticker']: r for r in turtle_results}
        e_map = {r['ticker']: r for r in expert_results}
        
        for t_ticker, t_data in t_map.items():
            if '매수' in t_data['signal'] and t_ticker in e_map:
                e_data = e_map[t_ticker]
                sector = e_data['sector']
                limit = self.sector_mapper.get_sector_limit(sector)
                curr_exposure = current_alloc.get(sector, 0)
                
                warning_msg = ""
                if curr_exposure >= limit:
                    warning_msg = f" ⚠️비중초과 ({int(curr_exposure*100)}%)"
                
                if e_data['score'] >= 70:
                    tier1.append({
                        'ticker': t_ticker,
                        'tier': 'Tier 1 (교집합)',
                        'price': t_data['price'],
                        'reason': f"교집합{warning_msg}",
                        'score': e_data['score']
                    })
        
        tier1.sort(key=lambda x: x['score'], reverse=True)
        
        # Tier 2: Expert 단독
        tier2 = []
        for e in expert_results:
            sector = e['sector']
            limit = self.sector_mapper.get_sector_limit(sector)
            curr_exposure = current_alloc.get(sector, 0)
            
            warning_msg = ""
            if curr_exposure >= limit:
                warning_msg = f" ⚠️비중초과 ({int(curr_exposure*100)}%)"
            
            if e['score'] >= 80 and e['ticker'] not in [x['ticker'] for x in tier1]:
                tier2.append({
                    'ticker': e['ticker'],
                    'tier': 'Tier 2 (Expert)',
                    'price': e['close'],
                    'reason': f"우량주{warning_msg}",
                    'score': e['score']
                })
        
        tier2.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. 현금 비율에 따른 매수 후보 동적 조정 (V24.9: 상위 20%/50%/80% 방식)
        total_cash = self.cash + total_sell_proceeds
        required_cash_ratio = self.ma.cash_ratio
        all_candidates = tier1 + tier2
        
        # 시나리오별 종목 선정 기준: 상위 20%/50%/80%
        max_a = max(1, int(len(all_candidates) * 0.2))
        buy_candidates_a = all_candidates[:max_a] if max_a > 0 else all_candidates[:1]
        
        max_b = max(1, int(len(all_candidates) * 0.5))
        buy_candidates_b = all_candidates[:max_b]
        
        max_c = max(1, int(len(all_candidates) * 0.8))
        buy_candidates_c = all_candidates[:max_c]
        
        # 4. 시나리오별 매수 계획 (방어모드에서 B/C 강제 제한)
        # 시나리오 A: 보수적 (최대 20% 투자, 나머지 현금)
        alloc_a = total_cash * min(0.2, 1.0 - required_cash_ratio)
        buy_a = self._allocate_smart_sniper(alloc_a, buy_candidates_a)
        
        # 시나리오 B: 균형 (최대 50% 투자) - 흐림 이상에서는 30% 로 제한
        if required_cash_ratio >= 0.3:  # 흐림 (30%) 이상
            # ✅ 흐림/방어모드: 시나리오 B 최대 30% 로 제한
            alloc_b = total_cash * min(0.3, max(0.0, 1.0 - required_cash_ratio))
            logger.warning(f"🚨 흐림/방어모드 (현금 {required_cash_ratio*100:.0f}%) - 시나리오 B {min(0.3, max(0.0, 1.0 - required_cash_ratio))*100:.0f}% 로 제한")
        else:
            alloc_b = total_cash * min(0.5, 1.0 - required_cash_ratio)
        buy_b = self._allocate_smart_sniper(alloc_b, buy_candidates_b)
        
        # 시나리오 C: 공격적 (남은 현금 모두 투자) - 방어모드에서는 차단
        if required_cash_ratio >= 0.5:  # 흐림 (30%) 이상이면 방어모드로 간주
            # ✅ 방어모드: 시나리오 C 강제 차단 (투자 0%)
            alloc_c = 0.0
            buy_c = []
            logger.warning(f"🚨 방어모드 (현금 {required_cash_ratio*100:.0f}%) - 시나리오 C 투자 차단")
        else:
            alloc_c = total_cash * (1.0 - required_cash_ratio)
            buy_c = self._allocate_smart_sniper(alloc_c, buy_candidates_c)
        
        self.scenarios = {
            'A': {
                'name': '보수적 (분산)',
                'buy_list': buy_a,
                'cash_reserve': total_cash - alloc_a,
                'cash_ratio': max(required_cash_ratio, 0.8)  # 최소 80% 현금 보유
            },
            'B': {
                'name': '균형 (분산)' if required_cash_ratio < 0.3 else '제한 (흐림/방어)',
                'buy_list': buy_b,
                'cash_reserve': total_cash - alloc_b,
                'cash_ratio': max(required_cash_ratio, 0.5) if required_cash_ratio < 0.3 else max(required_cash_ratio, 0.7)  # 흐림 이상에서는 70% 현금
            },
            'C': {
                'name': '공격 (분산)' if required_cash_ratio < 0.5 else '차단 (방어모드)',
                'buy_list': buy_c,
                'cash_reserve': total_cash - alloc_c,
                'cash_ratio': required_cash_ratio if required_cash_ratio < 0.5 else 1.0  # 방어모드는 현금 100%
            }
        }
        
        # 히스토리 저장 (한 번만 호출)
        self._save_score_history()
        
        return self.sell_candidates, self.scenarios
    
    def _allocate_smart_sniper(self, budget: float, candidates: list) -> list:
        """
        스마트 스나이퍼 할당
        
        Args:
            budget: 할당 예산
            candidates: 후보 종목
        
        Returns:
            list: 매수 계획
        """
        if not candidates or budget < 10_000_000:  # 최소 1,000 만원
            return []
        
        min_invest = 10_000_000
        target_count = min(len(candidates), int(budget / min_invest))
        
        if target_count < 1:
            target_count = 1
        
        targets = candidates[:target_count]
        
        if not targets:
            return []
        
        per_stock = budget / len(targets)
        plan = []
        
        for c in targets:
            qty = int(per_stock / (c['price'] * self.usd_krw))
            if qty > 0:
                plan.append({
                    'ticker': c['ticker'],
                    'qty': qty,
                    'amount': qty * c['price'] * self.usd_krw
                })
        
        return plan
    
    def _load_score_history(self) -> dict:
        """매도 점수 히스토리 로드 (V24.9)"""
        import os
        import json
        
        history_file = os.path.join(self.data_dir, 'sell_score_history.json')
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"매도 점수 히스토리 로드 실패: {e}")
        return {}
    
    def _save_score_history(self):
        """매도 점수 히스토리 저장 (V24.9)"""
        import os
        import json
        
        if not self.data_dir:
            return
        
        history_file = os.path.join(self.data_dir, 'sell_score_history.json')
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.sell_score_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"매도 점수 히스토리 저장 실패: {e}")
    
    def calculate_sell_score(self, h: dict) -> tuple:
        """
        매도 점수 계산 (V24.9: 히스토리 기반)
        
        Args:
            h: 보유 종목 데이터
        
        Returns:
            tuple: (점수, 사유 리스트)
        """
        score = 0
        reasons = []
        
        # 1. 손절가 이탈 (즉시 매도)
        stop_loss_price = h.get('stop_loss', 0)
        if stop_loss_price > 0 and h['current_price'] < stop_loss_price:
            score += 100
            reasons.append("손절가 이탈")
        
        # 2. Expert 점수 미달
        if h['expert_score'] > 0 and h['expert_score'] < 40:
            score += 50
            reasons.append(f"Expert {h['expert_score']}점 (미달)")
        
        # 3. RS 점수 약세
        if h['rs_score'] > 0 and h['rs_score'] < 40:
            score += 30
            reasons.append(f"RS {h['rs_score']}점 (약세)")
        
        # 4. 하락 추세 전환
        if h.get('trend_score', 0) < 0:
            score += 20
            reasons.append("하락 추세 전환")
        
        # 5. 점수 히스토리 참조 (V24.9) - 기록은 generate_plan 에서 매도 후보만
        ticker = h['ticker']
        prev_score = self.sell_score_history.get(ticker, 0)
        if score > prev_score and score >= 50:
            reasons.append(f"점수 급등 ({prev_score}→{score})")
        # ❌ 여기서 기록 제거 (generate_plan 에서 매도 후보만 기록)
        
        return score, reasons
