"""
히스토리 저장 모듈 (SQLite 기반)
- Race Condition 해결
- 파일 띅 불필요
- 쿼리 가능
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class HistoryStore:
    """SQLite 기반 히스토리 저장소"""
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: SQLite 데이터베이스 경로
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화 (테이블 생성)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS canary_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    negative_count INTEGER,
                    market_score INTEGER,
                    signals TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sell_score_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_canary_date 
                ON canary_history(date)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_sell_ticker 
                ON sell_score_history(ticker)
            ''')
            
            conn.commit()
        
        logger.debug(f"히스토리 데이터베이스 초기화: {self.db_path}")
    
    def save_canary(self, mode: str, negative_count: int, market_score: int, 
                    signals: Dict, date: str = None):
        """
        카나리아 히스토리 저장
        
        Args:
            mode: 공격/주의/방어
            negative_count: 음수 신호 개수
            market_score: 마켓 스코어
            signals: 카나리아 신호 dict
            date: 날짜 (기본: 오늘)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO canary_history 
                    (date, mode, negative_count, market_score, signals)
                    VALUES (?, ?, ?, ?, ?)
                ''', (date, mode, negative_count, market_score, json.dumps(signals)))
                conn.commit()
            
            logger.debug(f"카나리아 히스토리 저장: {mode} ({negative_count}개)")
        except Exception as e:
            logger.error(f"카나리아 히스토리 저장 실패: {e}")
    
    def get_canary_history(self, limit: int = 50) -> List[Dict]:
        """
        카나리아 히스토리 조회
        
        Args:
            limit: 조회할 기록 수
        
        Returns:
            List[Dict]: 히스토리 목록
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT date, mode, negative_count, market_score, signals
                    FROM canary_history
                    ORDER BY date DESC, created_at DESC
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor:
                    results.append({
                        'date': row['date'],
                        'mode': row['mode'],
                        'negative_count': row['negative_count'],
                        'market_score': row['market_score'],
                        'signals': json.loads(row['signals']) if row['signals'] else {}
                    })
                
                return results
        except Exception as e:
            logger.error(f"카나리아 히스토리 조회 실패: {e}")
            return []
    
    def save_sell_score(self, ticker: str, score: int, date: str = None):
        """
        매도 점수 히스토리 저장
        
        Args:
            ticker: 종목 티커
            score: 매도 점수
            date: 날짜 (기본: 오늘)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO sell_score_history 
                    (date, ticker, score)
                    VALUES (?, ?, ?)
                ''', (date, ticker, score))
                conn.commit()
            
            logger.debug(f"매도 점수 저장: {ticker} = {score}")
        except Exception as e:
            logger.error(f"매도 점수 저장 실패: {e}")
    
    def get_sell_score(self, ticker: str, date: str = None) -> int:
        """
        매도 점수 조회
        
        Args:
            ticker: 종목 티커
            date: 날짜 (기본: 오늘)
        
        Returns:
            int: 매도 점수 (없으면 0)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT score FROM sell_score_history
                    WHERE ticker = ? AND date = ?
                ''', (ticker, date))
                
                row = cursor.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"매도 점수 조회 실패: {e}")
            return 0
    
    def get_sell_score_trend(self, ticker: str, days: int = 7) -> List[int]:
        """
        매도 점수 추이 조회
        
        Args:
            ticker: 종목 티커
            days: 조회 일수
        
        Returns:
            List[int]: 점수 목록 (과거→현재)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT score, date FROM sell_score_history
                    WHERE ticker = ?
                    ORDER BY date DESC
                    LIMIT ?
                ''', (ticker, days))
                
                results = [row[0] for row in cursor.fetchall()]
                return list(reversed(results))  # 과거→현재
        except Exception as e:
            logger.error(f"매도 점수 추이 조회 실패: {e}")
            return []
    
    def cleanup(self, keep_days: int = 90):
        """
        오래된 히스토리 정리
        
        Args:
            keep_days: 유지할 일수
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    DELETE FROM canary_history WHERE date < ?
                ''', (cutoff_date,))
                conn.execute('''
                    DELETE FROM sell_score_history WHERE date < ?
                ''', (cutoff_date,))
                conn.commit()
            
            logger.info(f"히스토리 정리 완료 ({keep_days}일 이전 삭제)")
        except Exception as e:
            logger.error(f"히스토리 정리 실패: {e}")


# 편의 함수
def get_history_store(data_dir: str = None) -> HistoryStore:
    """
    히스토리 저장소 생성 (단일 인스턴스 권장)
    
    Args:
        data_dir: 데이터 디렉토리
    
    Returns:
        HistoryStore: 히스토리 저장소 인스턴스
    """
    if data_dir is None:
        data_dir = r"C:\Quant\Data"
    
    db_path = os.path.join(data_dir, "history.db")
    return HistoryStore(db_path)
