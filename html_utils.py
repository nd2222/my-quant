"""
HTML 유틸리티 모듈
- XSS 방지 (HTML escape)
- 안전한 HTML 생성
"""
import html
import re
from typing import Optional


def escape_html(text: str) -> str:
    """
    HTML 이스케이프 (XSS 방지)
    
    Args:
        text: 이스케이프할 텍스트
    
    Returns:
        str: 이스케이프된 텍스트
    
    예시:
        >>> escape_html("<script>alert('XSS')</script>")
        '&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;'
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def escape_ticker(ticker: str) -> str:
    """
    티커 HTML 이스케이프 (전용 함수)
    
    Args:
        ticker: 종목 티커
    
    Returns:
        str: 안전한 티커 문자열
    """
    if ticker is None:
        return ""
    
    # 티커는 대문자 영문 + 숫자 + 점만 허용
    # 그 외 문자는 제거
    safe_ticker = re.sub(r'[^A-Z0-9.\-]', '', str(ticker).upper())
    
    # 추가로 HTML 이스케이프
    return html.escape(safe_ticker, quote=True)


def safe_table_row(cells: list, headers: bool = False) -> str:
    """
    안전한 테이블 행 생성
    
    Args:
        cells: 셀 값 목록
        headers: 헤더 행 여부
    
    Returns:
        str: HTML 테이블 행
    """
    tag = 'th' if headers else 'td'
    escaped_cells = [f"<{tag}>{escape_html(str(cell))}</{tag}>" for cell in cells]
    return f"<tr>{''.join(escaped_cells)}</tr>"


def format_number(value: float, decimals: int = 2) -> str:
    """
    숫자 포맷팅 (천 단위 구분)
    
    Args:
        value: 숫자 값
        decimals: 소수점 자리수
    
    Returns:
        str: 포맷팅된 숫자 문자열
    """
    if value is None:
        return "N/A"
    
    try:
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: float, show_sign: bool = True) -> str:
    """
    퍼센트 포맷팅
    
    Args:
        value: 퍼센트 값 (0-100)
        show_sign: 부호 표시 여부
    
    Returns:
        str: 포맷팅된 퍼센트 문자열
    """
    if value is None:
        return "N/A"
    
    try:
        if show_sign:
            return f"{value:+.2f}%"
        return f"{value:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def colorize_by_value(value: float, thresholds: dict = None) -> str:
    """
    값에 따른 색상 반환
    
    Args:
        value: 비교할 값
        thresholds: 임계값 dict ({'positive': '#2ecc71', 'negative': '#e74c3c'})
    
    Returns:
        str: HTML 색상 코드
    """
    if thresholds is None:
        thresholds = {
            'positive': '#2ecc71',
            'neutral': '#f1c40f',
            'negative': '#e74c3c'
        }
    
    if value > 2:
        return thresholds['positive']
    elif value < -2:
        return thresholds['negative']
    else:
        return thresholds['neutral']


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    안전한 나눗셈 (0 division 방지)
    
    Args:
        numerator: 분자
        denominator: 분모
        default: 분모가 0 일 때 반환값
    
    Returns:
        float: 나눗셈 결과
    """
    if denominator == 0:
        return default
    return numerator / denominator


# HTML 템플릿 상수
TABLE_STYLE = """
<style>
    .detail-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    .detail-table th {
        background: #ecf0f1;
        color: #2c3e50;
        padding: 12px;
        text-align: center;
        font-weight: 700;
        border-bottom: 2px solid #bdc3c7;
    }
    .detail-table td {
        border-bottom: 1px solid #eee;
        padding: 12px;
        text-align: center;
        color: #333;
    }
    .detail-table tr:nth-child(even) {
        background-color: #fcfcfc;
    }
    .up { color: #e74c3c; font-weight: 700; }
    .down { color: #3498db; font-weight: 700; }
</style>
"""

CARD_STYLE = """
<style>
    .card {
        background: #fff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    .card h2 {
        border-left: 6px solid #2c3e50;
        padding-left: 15px;
        color: #2c3e50;
        margin-top: 0;
    }
</style>
"""
