"""
주문 이력 관리 모듈 (P2-14)
- 주문 이력 DB 저장
- 감사 추적
- 일일 최대 주문량 제한
"""
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OrderHistory:
    """주문 이력 데이터베이스"""
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: SQLite 데이터베이스 경로
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_date TEXT NOT NULL,
                    order_time TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    memo TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_limits (
                    date TEXT PRIMARY KEY,
                    total_amount REAL DEFAULT 0,
                    order_count INTEGER DEFAULT 0,
                    max_amount REAL DEFAULT 100000000,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
        
        logger.debug(f"주문 이력 DB 초기화: {self.db_path}")
    
    def add_order(self, ticker: str, side: str, qty: int, price: float, 
                  memo: str = None) -> bool:
        """
        주문 이력 추가
        
        Args:
            ticker: 종목 티커
            side: 'BUY' or 'SELL'
            qty: 수량
            price: 가격
            memo: 메모
        
        Returns:
            bool: 성공 여부
        """
        try:
            now = datetime.now()
            order_date = now.strftime('%Y-%m-%d')
            order_time = now.strftime('%H:%M:%S')
            amount = qty * price
            
            # 일일 한도 확인
            if not self._check_daily_limit(order_date, amount):
                logger.warning(f"일일 한도 초과: {ticker} {side} {qty}주 ({amount:,.0f}원)")
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO orders 
                    (order_date, order_time, ticker, side, qty, price, amount, memo)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (order_date, order_time, ticker, side, qty, price, amount, memo))
                
                # 일일 집계 업데이트
                conn.execute('''
                    INSERT OR REPLACE INTO daily_limits 
                    (date, total_amount, order_count, updated_at)
                    VALUES (
                        ?,
                        COALESCE((SELECT total_amount FROM daily_limits WHERE date = ?), 0) + ?,
                        COALESCE((SELECT order_count FROM daily_limits WHERE date = ?), 0) + 1,
                        CURRENT_TIMESTAMP
                    )
                ''', (order_date, order_date, amount, order_date))
                
                conn.commit()
            
            logger.info(f"주문 이력 추가: {ticker} {side} {qty}주 @ {price:,.0f}원")
            return True
        
        except Exception as e:
            logger.error(f"주문 이력 추가 실패: {e}")
            return False
    
    def _check_daily_limit(self, date: str, amount: float) -> bool:
        """
        일일 한도 확인
        
        Args:
            date: 날짜
            amount: 주문 금액
        
        Returns:
            bool: 한도 이내면 True
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT total_amount, max_amount FROM daily_limits
                    WHERE date = ?
                ''', (date,))
                
                row = cursor.fetchone()
                if row:
                    current_total, max_amount = row
                    return (current_total + amount) <= max_amount
                else:
                    return amount <= 100000000  # 기본 1 억
        
        except Exception as e:
            logger.error(f"일일 한도 확인 실패: {e}")
            return True  # 오류 시 허용 (안전장치)
    
    def get_orders(self, date: str = None, limit: int = 100) -> List[Dict]:
        """
        주문 이력 조회
        
        Args:
            date: 날짜 (기본: 오늘)
            limit: 조회 수
        
        Returns:
            List[Dict]: 주문 이력 목록
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM orders
                    WHERE order_date = ?
                    ORDER BY order_time DESC
                    LIMIT ?
                ''', (date, limit))
                
                results = []
                for row in cursor:
                    results.append({
                        'id': row['id'],
                        'order_date': row['order_date'],
                        'order_time': row['order_time'],
                        'ticker': row['ticker'],
                        'side': row['side'],
                        'qty': row['qty'],
                        'price': row['price'],
                        'amount': row['amount'],
                        'status': row['status'],
                        'memo': row['memo']
                    })
                
                return results
        
        except Exception as e:
            logger.error(f"주문 이력 조회 실패: {e}")
            return []
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """
        일일 주문 요약
        
        Args:
            date: 날짜 (기본: 오늘)
        
        Returns:
            Dict: 요약 정보
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT total_amount, order_count, max_amount
                    FROM daily_limits
                    WHERE date = ?
                ''', (date,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'date': date,
                        'total_amount': row[0],
                        'order_count': row[1],
                        'max_amount': row[2],
                        'remaining': row[2] - row[0]
                    }
                else:
                    return {
                        'date': date,
                        'total_amount': 0,
                        'order_count': 0,
                        'max_amount': 100000000,
                        'remaining': 100000000
                    }
        
        except Exception as e:
            logger.error(f"일일 요약 조회 실패: {e}")
            return {}
    
    def set_daily_limit(self, max_amount: float, date: str = None):
        """
        일일 한도 설정
        
        Args:
            max_amount: 최대 금액
            date: 날짜 (기본: 오늘)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO daily_limits 
                    (date, max_amount, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (date, max_amount))
                conn.commit()
            
            logger.info(f"일일 한도 설정: {max_amount:,.0f}원")
        
        except Exception as e:
            logger.error(f"일일 한도 설정 실패: {e}")


# 편의 함수
def get_order_history(data_dir: str = None) -> OrderHistory:
    """주문 이력 인스턴스 생성"""
    if data_dir is None:
        data_dir = r"C:\Quant\Data"
    
    db_path = os.path.join(data_dir, "orders.db")
    return OrderHistory(db_path)
