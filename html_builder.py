"""
리포트 생성 모듈 - HTML 빌더
"""
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class HTMLReportBuilder:
    """HTML 리포트 빌더"""
    
    def __init__(self, holdings_data, macro_data, turtle_data, expert_data, rs_data, scenarios, total_asset, cash_balance, usd_krw, macro_analyzer=None, sector_mapper=None):
        self.holdings = holdings_data or []
        self.macro = macro_data or {}
        self.turtle = turtle_data or {}
        self.expert = expert_data or {}
        self.rs = rs_data or {}
        self.scenarios = scenarios or {}
        self.total_asset = total_asset
        self.cash_balance = cash_balance
        self.usd_krw = usd_krw
        self.ma = macro_analyzer
        self.sector_mapper = sector_mapper
    
    def build(self) -> str:
        """최종 HTML 조립 (V24.9)"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><style>{self._build_css()}</style></head>
<body>
    <div class="container">
        <div style="text-align:right;font-size:0.85em;color:#666;margin-bottom:10px">Report Generated: {now}</div>
        {self._build_header()}
        {self._build_canary_section()}
        {self._build_macro_section()}
        {self._build_asset_dashboard()}
        {self._build_holdings_section()}
        {self._build_intersection_section()}
        {self._build_rebalance_section()}
        {self._build_strategy_tabs()}
    </div>
    <script>{self._build_js()}</script>
</body>
</html>"""
    
    def _build_css(self) -> str:
        return """body{background:#f0f2f5;font-family:'Malgun Gothic',sans-serif;padding:20px;color:#333}
.container{max-width:1400px;margin:auto}
h1{text-align:center;color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:15px}
.market-alert{background:#2ecc71;color:white;padding:15px;border-radius:8px;margin-bottom:20px;text-align:center;font-weight:700;font-size:1.2em}
.asset-dashboard{display:flex;justify-content:space-around;background:linear-gradient(135deg,#2c3e50,#34495e);color:#fff;padding:15px;border-radius:12px;margin-bottom:25px}
.asset-item{text-align:center}
.asset-label{font-size:.9em;opacity:.8;margin-bottom:5px}
.asset-value{font-size:1.4em;font-weight:700}
.card{background:#fff;padding:25px;border-radius:12px;box-shadow:0 3px 10px rgba(0,0,0,.08);margin-bottom:25px}
h2{border-left:6px solid #2c3e50;padding-left:15px;color:#2c3e50;margin-top:0}
.detail-table{width:100%;border-collapse:collapse;font-size:14px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1);border:1px solid #eee}
.detail-table th{background:#ecf0f1;color:#2c3e50;padding:12px;text-align:center;font-weight:700;border-bottom:2px solid #bdc3c7}
.detail-table td{border-bottom:1px solid #eee;padding:12px;text-align:center;color:#333}
.detail-table tr:nth-child(even){background-color:#fcfcfc}
.up{color:#e74c3c;font-weight:700}
.down{color:#3498db;font-weight:700}"""
    
    def _build_header(self) -> str:
        market_status = self.ma.regime if self.ma else "Unknown"
        market_score = self.ma.market_score if self.ma else 50
        ind_summary = f"VIX: {self.ma.indicators.get('VIX', 20)} | 10Y: {self.ma.indicators.get('10Y_Yield', 4.0)}% | Oil: ${self.ma.indicators.get('Oil', 80)}" if self.ma else ""
        return f"""<h1>Quant System V24.9 Report</h1>
        <div class="market-alert">{market_status} (Score: {market_score})<br><span style="font-size:0.8em">{ind_summary}</span></div>"""
    
    def _build_canary_section(self) -> str:
        """카나리아 3 레이어 (V24.9)"""
        if not self.ma or not hasattr(self.ma, 'canary_signals'):
            return ""
        
        mode = getattr(self.ma, 'canary_mode', '공격')
        mode_color = getattr(self.ma, 'canary_mode_color', '#2ecc71')
        mode_icon = getattr(self.ma, 'canary_mode_icon', '🟢')
        neg_count = getattr(self.ma, 'canary_negative_count', 0)
        
        # 레이어 1: 최상단 — 즉각적인 결론
        if mode == '공격':
            mode_desc = "카나리아 이상 없음"
            action_desc = "공격 모드 (풀 투자)"
        elif mode == '주의':
            mode_desc = "1 개 종목 이탈"
            action_desc = "주의 모드 (50% 방어)"
        else:
            mode_desc = f"{neg_count}개 종목 이탈"
            action_desc = "방어 모드 (현금/채권)"
        
        html = f"""<div class="card" style="border-left: 6px solid {mode_color}">
            <h2>🕊️ 카나리아 현황판</h2>
            <div style="background:{mode_color};color:white;padding:20px;border-radius:8px;margin-bottom:20px;text-align:center">
                <div style="font-size:1.5em;font-weight:bold;margin-bottom:5px">{mode_icon} {mode} 모드</div>
                <div style="font-size:1.1em;opacity:0.95">{mode_desc}</div>
                <div style="font-size:0.95em;opacity:0.9;margin-top:5px">{action_desc}</div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px">"""
        
        for ticker, data in self.ma.canary_signals.items():
            ret = data.get('return', 0)
            is_neg = data.get('negative', False)
            color = "#e74c3c" if is_neg else "#2ecc71"
            icon = "🔴" if is_neg else "🟢"
            momentum = data.get('momentum_13612', ret)
            
            html += f"""<div style="background:{color}10;border:2px solid {color};border-radius:8px;padding:15px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <div style="font-weight:bold;font-size:1.2em">{ticker}</div>
                    <div style="font-size:1.5em">{icon}</div>
                </div>
                <div style="font-size:0.85em;color:#666;margin-bottom:8px">{data.get('name','')}</div>
                <div style="font-size:1.3em;font-weight:bold;color:{color};margin-bottom:10px">{ret:+.1f}%</div>
                <div style="font-size:0.8em;color:#555;line-height:1.6">
                    <div>13612W: <b>{momentum:+.1f}%</b></div>
                </div>
            </div>"""
        
        html += """</div>"""
        
        # 레이어 3: 히스토리
        if self.ma.canary_history_file and os.path.exists(self.ma.canary_history_file):
            import json
            try:
                with open(self.ma.canary_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                records = history.get('records', [])[-5:]
                consecutive = history.get('consecutive_days', 0)
                
                if records or consecutive > 0:
                    html += f"""<div style="background:#f8f9fa;border-radius:8px;padding:15px">
                        <div style="font-weight:bold;color:#2c3e50;margin-bottom:10px">📅 신호 변화 이력</div>
                        <div style="font-size:0.9em;color:#666;margin-bottom:10px">현재 {mode} 모드 지속: <b style="color:{mode_color}">{consecutive}일째</b></div>
                        <div style="font-size:0.85em;line-height:1.8">"""
                    for rec in reversed(records):
                        from_color = "#2ecc71" if rec['from'] == '공격' else ("#f1c40f" if rec['from'] == '주의' else "#e74c3c")
                        to_color = "#2ecc71" if rec['to'] == '공격' else ("#f1c40f" if rec['to'] == '주의' else "#e74c3c")
                        html += f"""<div>{rec['date']} <b style="color:{from_color}">{rec['from']}</b> → <b style="color:{to_color}">{rec['to']}</b> ({rec.get('reason','')})</div>"""
                    html += """</div></div>"""
            except:
                pass
        
        html += "</div>"
        return html
    
    def _build_macro_section(self) -> str:
        """매크로 섹션 (indicators dict 구조에 맞게 수정)"""
        if not self.macro:
            return '<div class="card"><h2>Market Data</h2><p>No data</p></div>'
        
        rows = ""
        for key, val in self.macro.items():
            # indicators 는 {'VIX': 20.5, '10Y_Yield': 4.2, ...} 형태
            rows += f"""<tr><td>{key}</td><td>{val}</td></tr>"""
        
        return f"""<div class="card"><h2>Market Indicators</h2>
            <table class='detail-table'>
                <tr><th>Indicator</th><th>Value</th></tr>
                {rows}
            </table>
        </div>"""
    
    def _build_asset_dashboard(self) -> str:
        stock_val = sum([p.get('current_price',0)*p.get('qty',0)*self.usd_krw for p in self.holdings])
        return f"""<div class="asset-dashboard">
            <div class="asset-item"><div class="asset-label">Total Asset</div><div class="asset-value">{self.total_asset/10000:,.0f} 만원</div></div>
            <div class="asset-item"><div class="asset-label">Stock Value</div><div class="asset-value">{stock_val/10000:,.0f} 만원</div></div>
            <div class="asset-item"><div class="asset-label">Cash</div><div class="asset-value">{self.cash_balance/10000:,.0f} 만원</div></div>
            <div class="asset-item"><div class="asset-label">Exchange Rate</div><div class="asset-value">{self.usd_krw:,.2f} 원</div></div>
        </div>"""
    
    def _build_holdings_section(self) -> str:
        if not self.holdings:
            return '<div class="card"><h2>Holdings</h2><p>No holdings</p></div>'
        rows = ""
        for p in self.holdings:
            ticker = p.get('ticker','')
            price = p.get('current_price',0)
            entry = p.get('entry_price', price)
            profit = ((price/entry)-1)*100 if entry > 0 else 0
            rows += f"""<tr><td style="text-align:left"><b>{ticker}</b></td><td>${entry:.2f}</td><td>${price:.2f}</td><td class="{'up' if profit>0 else 'down'}">{profit:+.1f}%</td></tr>"""
        return f"""<div class="card"><h2>Holdings ({len(self.holdings)} stocks)</h2><table class='detail-table'><tr><th>Ticker</th><th>Entry</th><th>Current</th><th>Profit</th></tr>{rows}</table></div>"""
    
    def _build_intersection_section(self) -> str:
        """교집합 테이블 (Turtle + Expert)"""
        turtle_results = self.turtle.get('results', []) if isinstance(self.turtle, dict) else []
        expert_results = self.expert.get('results', []) if isinstance(self.expert, dict) else []
        
        t_map = {r['ticker']: r for r in turtle_results}
        e_map = {r['ticker']: r for r in expert_results}
        
        intersection = []
        for t_ticker, t_data in t_map.items():
            if '매수' in t_data.get('signal', '') and t_ticker in e_map:
                e_data = e_map[t_ticker]
                intersection.append({
                    'ticker': t_ticker,
                    'signal': t_data.get('signal', ''),
                    't_score': t_data.get('trend_score', 0),
                    'e_score': e_data.get('score', 0),
                    'rs_score': e_data.get('rs_score', 0),
                    'price': e_data.get('close', t_data.get('price', 0))
                })
        
        if not intersection:
            return ""
        
        rows = ""
        for item in intersection:
            rows += f"""<tr><td><b>{item['ticker']}</b></td><td>{item['signal']}</td><td>{item['t_score']:.0f}</td><td class="up">{item['e_score']}</td><td>{item['rs_score']:.0f}</td><td>${item['price']:.2f}</td></tr>"""
        
        return f"""<div class="card section-intersection" style="border-left:6px solid #e67e22">
            <h2>🎯 교집합 (Turtle + Expert)</h2>
            <table class='detail-table'>
                <tr><th>Ticker</th><th>Signal</th><th>T Score</th><th>E Score</th><th>RS</th><th>Price</th></tr>
                {rows}
            </table>
        </div>"""
    
    def _build_rebalance_section(self) -> str:
        if not self.scenarios:
            return '<div class="card"><h2>Rebalance</h2><p>No scenarios</p></div>'
        
        html = '<div class="card"><h2>📊 Rebalance Scenarios</h2><div style="display:flex;gap:10px">'
        for key, sc in self.scenarios.items():
            name = sc.get('name', key)
            cash = sc.get('cash_reserve', 0)
            buy_list = sc.get('buy_list', [])
            
            html += f"""<div style="flex:1;background:#f8f9fa;padding:10px;border-radius:5px;border:1px solid #ddd">
                <h4>{key}. {name}</h4>
                <p>Cash: {cash/10000:,.0f} 만원</p>
                <ul style="padding-left:20px">"""
            
            if buy_list:
                for b in buy_list:
                    html += f"""<li><b>{b.get('ticker','')}</b>: {b.get('qty',0)} 주</li>"""
            else:
                html += "<li>No recommendations</li>"
            
            html += """</ul></div>"""
        
        html += "</div></div>"
        return html
    
    def _build_strategy_tabs(self) -> str:
        """전략 탭 (Turtle/Expert/RS)"""
        turtle_results = self.turtle.get('results', []) if isinstance(self.turtle, dict) else []
        expert_results = self.expert.get('results', []) if isinstance(self.expert, dict) else []
        rs_results = self.rs.get('results', []) if isinstance(self.rs, dict) else []
        
        html = """<div class="strategy-tabs" style="display:flex;margin-bottom:20px">
            <div class="strategy-tab active" onclick="showStrategy('turtle')" style="padding:12px 25px;cursor:pointer;background:#ecf0f1;border:1px solid #bdc3c7;font-weight:700">Turtle</div>
            <div class="strategy-tab" onclick="showStrategy('expert')" style="padding:12px 25px;cursor:pointer;background:#ecf0f1;border:1px solid #bdc3c7;font-weight:700">Expert</div>
            <div class="strategy-tab" onclick="showStrategy('rs')" style="padding:12px 25px;cursor:pointer;background:#ecf0f1;border:1px solid #bdc3c7;font-weight:700">RS</div>
        </div>"""
        
        # Turtle
        html += f"""<div class="strategy-content active" id="strategy-turtle" style="display:block">
            <div class="card"><h2>Turtle Results ({len(turtle_results)} stocks)</h2>"""
        if turtle_results:
            html += "<table class='detail-table'><tr><th>Ticker</th><th>Signal</th><th>Score</th></tr>"
            for r in turtle_results[:20]:
                html += f"""<tr><td><b>{r.get('ticker','')}</b></td><td>{r.get('signal','')}</td><td>{r.get('trend_score',0):.0f}</td></tr>"""
            html += "</table>"
        html += "</div></div>"
        
        # Expert
        html += f"""<div class="strategy-content" id="strategy-expert" style="display:none">
            <div class="card"><h2>Expert Results ({len(expert_results)} stocks)</h2>"""
        if expert_results:
            html += "<table class='detail-table'><tr><th>Ticker</th><th>Score</th><th>RS</th></tr>"
            for r in expert_results[:20]:
                html += f"""<tr><td><b>{r.get('ticker','')}</b></td><td>{r.get('score',0)}</td><td>{r.get('rs_score',0):.0f}</td></tr>"""
            html += "</table>"
        html += "</div></div>"
        
        # RS
        html += f"""<div class="strategy-content" id="strategy-rs" style="display:none">
            <div class="card"><h2>RS Results ({len(rs_results)} stocks)</h2>"""
        if rs_results:
            html += "<table class='detail-table'><tr><th>Ticker</th><th>RS Score</th><th>Momentum</th></tr>"
            for r in rs_results[:20]:
                html += f"""<tr><td><b>{r.get('ticker','')}</b></td><td>{r.get('rs_score',0):.1f}</td><td>{r.get('momentum_13612',0):+.1f}%</td></tr>"""
            html += "</table>"
        html += "</div></div>"
        
        return html
    
    def _build_js(self) -> str:
        return """function showStrategy(type) {
    document.querySelectorAll('.strategy-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.strategy-content').forEach(c => c.style.display = 'none');
    event.target.classList.add('active');
    document.getElementById('strategy-' + type).style.display = 'block';
}"""
