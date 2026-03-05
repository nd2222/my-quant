"""
매크로 분석 모듈
- VIX, 금리, 유가, 달러 분석
- 카나리아 신호등 (VWO, VEA, BND, SPY)
- 시장 레짐 판단
"""
import yfinance as yf
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MacroAnalyzer:
    """
    매크로 분석기 (as_of_date 지원 - 백테스트용)
    
    분석 지표:
        - VIX (공포지수)
        - 10 년물 국채금리
        - WTI 원유
        - 달러인덱스
        - S&P500 vs MA200
    
    카나리아 신호등:
        - SPY (미국 주식)
        - VWO (신흥국 주식)
        - VEA (선진국 주식)
        - BND (채권)
    """
    
    def __init__(self, data_dir=None, as_of_date=None):
        """
        Args:
            data_dir: 데이터 디렉토리
            as_of_date: 분석 기준일 (None 이면 오늘, 백테스트용)
        """
        self.indicators = {}
        self.market_score = 50
        self.regime = "Unknown"
        self.cash_ratio = 0.0
        self.warnings = []
        self.canary_signals = {}
        self.canary_negative_count = 0
        self.canary_mode = '공격'
        self.canary_mode_color = '#2ecc71'
        self.canary_mode_icon = '🟢'
        self.data_dir = data_dir
        self.as_of_date = as_of_date  # ✅ 백테스트 기준일
        self.canary_history_file = os.path.join(data_dir, 'canary_history.json') if data_dir else None
    
    def run_deep_dive(self):
        """
        심층 매크로 분석
        """
        logger.info("[2/7] 매크로 딥 다이브 분석 중...")
        try:
            # 1. 주요 지표 수집
            self._collect_indicators()
            
            # 2. 시장 스코어 계산
            self._calculate_market_score()
            
            # 3. 카나리아 신호등 분석
            self._analyze_canary()
            
            # 4. 레짐 판단
            self._determine_regime()
            
            logger.info(f"마켓 스코어: {self.market_score}점 ({self.regime})")
            if self.warnings:
                for w in self.warnings:
                    logger.warning(f"⚠️ {w}")
        
        except Exception as e:
            logger.error(f"매크로 분석 실패: {e}")
            self.regime = "Unknown"
    
    def _collect_indicators(self):
        """주요 지표 수집 (as_of_date 지원)"""
        from datetime import timedelta
        
        try:
            # as_of_date 가 있으면 과거 날짜 기준, 없으면 오늘 기준
            if self.as_of_date:
                end = self.as_of_date
                start_short = end - timedelta(days=10)  # ✅ 5→10: 연휴/주말 여유 확보
                start_long = end - timedelta(days=365)
            else:
                end = None
                start_short = None
                start_long = None
            
            # VIX
            if self.as_of_date:
                vix_df = yf.Ticker("^VIX").history(start=start_short, end=end)
            else:
                vix_df = yf.Ticker("^VIX").history(period="5d")
            vix = vix_df['Close'].iloc[-1]
            self.indicators['VIX'] = round(vix, 2)
            
            # 10 년물 국채금리
            if self.as_of_date:
                tnx_df = yf.Ticker("^TNX").history(start=start_short, end=end)
            else:
                tnx_df = yf.Ticker("^TNX").history(period="5d")
            tnx = tnx_df['Close'].iloc[-1]
            self.indicators['10Y_Yield'] = round(tnx, 2)
            
            # WTI 원유
            if self.as_of_date:
                oil_df = yf.Ticker("CL=F").history(start=start_short, end=end)
            else:
                oil_df = yf.Ticker("CL=F").history(period="5d")
            oil = oil_df['Close'].iloc[-1]
            self.indicators['Oil'] = round(oil, 2)
            
            # 달러인덱스
            if self.as_of_date:
                dxy_df = yf.Ticker("DX-Y.NYB").history(start=start_short, end=end)
            else:
                dxy_df = yf.Ticker("DX-Y.NYB").history(period="5d")
            dxy = dxy_df['Close'].iloc[-1]
            self.indicators['DXY'] = round(dxy, 2)
            
            # S&P500 추세
            if self.as_of_date:
                spy = yf.Ticker("^GSPC").history(start=start_long, end=end)['Close']
            else:
                spy = yf.Ticker("^GSPC").history(period="1y")['Close']
            spy_curr = spy.iloc[-1]
            spy_ma200 = spy.rolling(200).mean().iloc[-1]
            self.indicators['SPY_Trend'] = "Bull" if spy_curr > spy_ma200 else "Bear"
        
        except Exception as e:
            logger.warning(f"지표 수집 실패: {e}")
    
    def _calculate_market_score(self):
        """
        시장 스코어 계산 (0-100)
        
        감점 요인:
            - VIX > 30: -50 점
            - VIX > 20: -20 점
            - VIX > 17: -10 점
            - 금리 > 4.5%: -20 점
            - 금리 > 4.3%: -10 점
            - 유가 > $90: -15 점
            - 유가 > $85: -5 점
            - 강달러: -10 점
            - S&P500 하락추세: -30 점
        """
        score = 100
        vix = self.indicators.get('VIX', 20)
        tnx = self.indicators.get('10Y_Yield', 4.0)
        oil = self.indicators.get('Oil', 80)
        dxy = self.indicators.get('DXY', 100)
        spy_trend = self.indicators.get('SPY_Trend', 'Bull')
        
        # VIX (2024~2025 년 기준 장기평균 19~20 으로 조정)
        if vix > 30:
            score -= 50
        elif vix > 25:
            score -= 20
        elif vix > 20:
            score -= 10
        
        # 금리
        if tnx > 4.5:
            score -= 20
            self.warnings.append("국채금리 급등")
        elif tnx > 4.3:
            score -= 10
            self.warnings.append("국채금리 상승세")
        
        # 유가
        if oil > 90:
            score -= 15
            self.warnings.append("유가 폭등")
        elif oil > 85:
            score -= 5
            self.warnings.append("유가 상승세")
        
        # 달러
        if dxy > 105:
            score -= 10
            self.warnings.append("강달러")
        
        # S&P500 추세
        if spy_trend == "Bear":
            score -= 30
            self.warnings.append("S&P500 하락 추세")
        
        self.market_score = max(0, score)
    
    def _calculate_momentum_13612(self, prices):
        """
        Keller 의 13612W 모멘텀 공식
        
        Args:
            prices: 가격 데이터 (pandas Series)
        
        Returns:
            float: 모멘텀 값 (%)
        """
        if len(prices) < 252:
            if len(prices) >= 63:
                return (prices.iloc[-1] / prices.iloc[-63] - 1) * 100
            return 0.0
        
        ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100
        ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100
        ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100
        ret_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) * 100
        
        momentum = (12 * ret_1m + 4 * ret_3m + 2 * ret_6m + ret_12m) / 19
        return momentum
    
    def _analyze_canary(self):
        """
        카나리아 신호등 분석 (V24.9: momentum_13612 기준, as_of_date 지원)
        
        4 개 ETF 의 13612W 모멘텀 확인:
            - SPY (미국 주식)
            - VWO (신흥국 주식)
            - VEA (선진국 주식)
            - BND (채권)
        
        음수 모멘텀 개수로 공격/방어 결정
        """
        from datetime import timedelta
        
        canary_tickers = {
            'SPY': '미국주식',
            'VWO': '신흥국',
            'VEA': '선진국',
            'BND': '채권'
        }
        
        negative_count = 0
        
        # as_of_date 가 있으면 과거 날짜 기준 (252 일 + 여유 확보)
        if self.as_of_date:
            end = self.as_of_date
            start = end - timedelta(days=400)  # 252 일 + 여유 (주말/휴장)
        else:
            end = None
            start = None
        
        for ticker, name in canary_tickers.items():
            try:
                # 데이터 수집 (최대한 긴 기간)
                if self.as_of_date:
                    df = yf.Ticker(ticker).history(start=start, end=end)
                else:
                    df = yf.Ticker(ticker).history(period="2y")  # 2 년으로 확장
                
                # 데이터 충분성 확인 (13612W 는 252 일 필요)
                if len(df) < 252:
                    logger.warning(f"카나리아 {ticker} 데이터 부족 ({len(df)}일/252 일) - 과거 데이터 확장 시도")
                    # 더 긴 기간으로 재시도
                    if self.as_of_date:
                        start_extra = end - timedelta(days=600)
                        df = yf.Ticker(ticker).history(start=start_extra, end=end)
                    else:
                        df = yf.Ticker(ticker).history(period="3y")
                
                if len(df) >= 252:
                    momentum_13612 = self._calculate_momentum_13612(df['Close'])
                    
                    # NaN/Inf 체크
                    import math
                    if math.isnan(momentum_13612) or math.isinf(momentum_13612):
                        logger.warning(f"카나리아 {ticker} NaN/Inf 감지 - 0 으로 처리")
                        momentum_13612 = 0.0
                    
                    # 1M/12M 수익률도 계산 (표시용)
                    ret_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0
                    ret_12m = (df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else 0
                    
                    self.canary_signals[ticker] = {
                        'name': name,
                        'return': round(momentum_13612, 1),
                        'negative': momentum_13612 < 0,
                        'momentum_13612': round(momentum_13612, 1),
                        'ret_1m': round(ret_1m, 1),
                        'ret_12m': round(ret_12m, 1)
                    }
                    if momentum_13612 < 0:
                        negative_count += 1
                else:
                    # 정말 데이터 없음
                    logger.error(f"카나리아 {ticker} 데이터 확보 실패 ({len(df)}일) - 0 으로 처리")
                    self.canary_signals[ticker] = {
                        'name': name,
                        'return': 0,
                        'negative': False,
                        'momentum_13612': 0,
                        'ret_1m': 0,
                        'ret_12m': 0
                    }
            except Exception as e:
                logger.error(f"카나리아 {ticker} 분석 실패: {e}")
                self.canary_signals[ticker] = {
                    'name': name,
                    'return': 0,
                    'negative': False,
                    'momentum_13612': 0,
                    'ret_1m': 0,
                    'ret_12m': 0
                }
        
        self.canary_negative_count = negative_count
        
        # 모드 결정
        if negative_count == 0:
            self.canary_mode = '공격'
            self.canary_mode_color = '#2ecc71'
            self.canary_mode_icon = '🟢'
        elif negative_count == 1:
            self.canary_mode = '주의'
            self.canary_mode_color = '#f1c40f'
            self.canary_mode_icon = '🟡'
        else:
            self.canary_mode = '방어'
            self.canary_mode_color = '#e74c3c'
            self.canary_mode_icon = '🔴'
        
        logger.info(f"🕊️ 카나리아: {self.canary_mode_icon} {self.canary_mode} 모드 (음수 {negative_count}/4 개)")
        
        # 히스토리 저장
        self._save_canary_history()
    
    def _determine_regime(self):
        """
        시장 레짐 판단
        
        카나리아 신호 우선:
            - 음수 0 개: 쾌청 (Strong Bull)
            - 음수 1 개: 맑음 (Bull)
            - 음수 2 개: 흐림 (Caution)
            - 음수 3 개 이상: 비/태풍 (Correction/Panic)
        """
        canary_neg = getattr(self, 'canary_negative_count', 0)
        
        if canary_neg >= 3:
            self.regime = "태풍 (Panic/Crash)"
            self.cash_ratio = 0.8
        elif canary_neg == 2:
            self.regime = "흐림 (Caution)"
            self.cash_ratio = 0.5
        elif canary_neg == 1:
            self.regime = "맑음 (Bull)"
            self.cash_ratio = 0.2
        else:
            # 카나리아 모두 양수면 시장 스코어 참조
            if self.market_score >= 80:
                self.regime = "쾌청 (Strong Bull)"
                self.cash_ratio = 0.0
            elif self.market_score >= 60:
                self.regime = "맑음 (Bull)"
                self.cash_ratio = 0.0
            elif self.market_score >= 40:
                self.regime = "흐림 (Caution)"
                self.cash_ratio = 0.3
            elif self.market_score >= 20:
                self.regime = "비 (Correction)"
                self.cash_ratio = 0.5
            else:
                self.regime = "태풍 (Panic/Crash)"
                self.cash_ratio = 0.8
    
    def _save_canary_history(self):
        """
        카나리아 히스토리 저장 (V24.9)
        """
        if not self.canary_history_file:
            return
        
        try:
            # 기존 히스토리 로드
            if os.path.exists(self.canary_history_file):
                with open(self.canary_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = {'records': [], 'current_mode': '공격', 'consecutive_days': 0}
            
            # 모드 변경 감지 (하루 1 회만 기록)
            today = datetime.now().strftime('%Y-%m-%d')
            last_record_date = history['records'][-1]['date'] if history['records'] else None
            prev_mode = history.get('current_mode', '공격')
            
            # 오늘 이미 기록했으면 스킵 (하루 1 회 제한)
            if last_record_date == today:
                history['consecutive_days'] += 1
            elif prev_mode != self.canary_mode:
                # 모드 변경 시에만 기록
                history['records'].append({
                    'date': today,
                    'from': prev_mode,
                    'to': self.canary_mode,
                    'negative_count': self.canary_negative_count,
                    'reason': self._get_mode_change_reason(prev_mode)
                })
                history['consecutive_days'] = 0
                history['current_mode'] = self.canary_mode
                logger.info(f"📅 카나리아 신호 변경: {prev_mode} → {self.canary_mode} ({history['consecutive_days']}일째)")
            else:
                history['consecutive_days'] += 1
            
            # 최근 50 개 기록만 유지
            history['records'] = history['records'][-50:]
            
            # 저장
            with open(self.canary_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.warning(f"카나리아 히스토리 저장 실패: {e}")
    
    def _get_mode_change_reason(self, prev_mode):
        """모드 변경 사유 반환"""
        neg = self.canary_negative_count
        signals = self.canary_signals
        
        if prev_mode == '공격' and neg >= 1:
            fallen = [t for t, d in signals.items() if d.get('negative', False)]
            return f"{', '.join(fallen)} 이탈"
        elif prev_mode == '주의' and neg >= 2:
            fallen = [t for t, d in signals.items() if d.get('negative', False)]
            return f"{', '.join(fallen)} 추가 이탈"
        elif prev_mode == '방어' and neg <= 1:
            recovered = [t for t, d in signals.items() if not d.get('negative', False) and d.get('return', 0) > 0]
            return f"{', '.join(recovered)} 회복"
        return ''
    
    def get_canary_html(self) -> str:
        """
        카나리아 신호등 HTML 생성
        
        Returns:
            str: HTML 문자열
        """
        if not self.canary_signals:
            return ""
        
        html = "<div style='margin-bottom:20px;background:#fff;padding:15px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.05)'>"
        html += "<h3>🕊️ 카나리아 신호등</h3>"
        html += "<div style='display:flex;gap:15px;flex-wrap:wrap;margin-top:10px'>"
        
        for ticker, data in self.canary_signals.items():
            ret = data['return']
            is_neg = data['negative']
            color = "#e74c3c" if is_neg else ("#f1c40f" if ret < 2 else "#2ecc71")
            icon = "🔴" if is_neg else ("🟡" if ret < 2 else "🟢")
            
            html += f"""
            <div style='flex:1;min-width:120px;background:{color}20;padding:10px;border-radius:6px;text-align:center'>
                <div style='font-weight:bold;font-size:1.1em'>{icon} {ticker}</div>
                <div style='font-size:0.9em;color:#666'>{data['name']}</div>
                <div style='font-size:1.2em;font-weight:bold;color:{color};margin-top:5px'>{ret:+.1f}%</div>
            </div>
            """
        
        # 종합 판단
        neg_count = getattr(self, 'canary_negative_count', 0)
        if neg_count == 0:
            recommendation = "✅ 공격 자산 100%"
            rec_color = "#2ecc71"
        elif neg_count == 1:
            recommendation = "⚖️ 공격 50% + 방어 50%"
            rec_color = "#f1c40f"
        else:
            recommendation = "⚠️ 방어 자산 100% (현금/채권)"
            rec_color = "#e74c3c"
        
        html += f"""
        <div style='width:100%;margin-top:15px;padding:10px;background:{rec_color}20;border-radius:6px;text-align:center;font-weight:bold;color:{rec_color}'>
            {recommendation}
        </div>
        """
        
        html += "</div></div>"
        return html
