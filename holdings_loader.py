"""
holdings.csv 로더 (증권사별 자동 감지)
"""
import pandas as pd
import os
import logging
import chardet

logger = logging.getLogger(__name__)

# 증권사별 컬럼명 매핑
BROKER_COLUMN_MAP = {
    # 키움증권
    '종목코드': 'ticker',
    '종목명': 'name', 
    '보유수량': 'qty',
    '현재가': 'current_price',
    '평균단가': 'avg_price',
    '손절가': 'stop_loss',
    '코드': 'ticker',
    '종목': 'ticker',
    '보유량': 'qty',
    '매입가': 'avg_price',
    '매입금액': 'total_price',
    # 미래에셋
    '종목번호': 'ticker',
    '수량': 'qty',
    '현재가격': 'current_price',
    '매입단가': 'avg_price',
    # 삼성증권
    '티커': 'ticker',
    '잔고수량': 'qty',
    # HTS 일반
    '평가금액': 'eval_amount',
    '평가손익': 'profit_loss',
    '평가수익률': 'profit_rate',
}

def load_holdings_csv(csv_path: str) -> list:
    """
    증권사 CSV → holdings 리스트 변환
    
    Args:
        csv_path: CSV 파일 경로
    
    Returns:
        list of dict: rebalance.py 가 기대하는 형식
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV 파일 없음: {csv_path}")
        return []
    
    try:
        # 인코딩 자동 감지
        with open(csv_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding'] or 'cp949'
        
        logger.info(f"감지된 인코딩: {detected_encoding}")
        
        # 인코딩 시도 (여러 인코딩 시도)
        df = None
        for encoding in [detected_encoding, 'utf-8', 'euc-kr', 'cp949', 'utf-8-sig']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding, skiprows=1)
                logger.info(f"인코딩 성공: {encoding}")
                break
            except UnicodeDecodeError:
                logger.debug(f"인코딩 실패: {encoding}")
                continue
        
        if df is None:
            logger.error("모든 인코딩 시도 실패")
            return []
        
        # 컬럼명 공백 제거
        df.columns = df.columns.str.strip()
        logger.info(f"CSV 컬럼: {list(df.columns)}")  # ← 이거 로그로 확인하세요
        
        # 컬럼명 정규화 (매핑 적용)
        df = df.rename(columns=BROKER_COLUMN_MAP)
        logger.info(f"정규화 후 컬럼: {list(df.columns)}")
        
        # 필수 컬럼 체크
        required = ['ticker', 'qty', 'current_price']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"필수 컬럼 없음: {missing}")
            logger.error(f"실제 컬럼: {list(df.columns)}")
            return []
        
        # 숫자 정제 (쉼표 제거 등)
        for col in ['qty', 'current_price', 'avg_price', 'stop_loss']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('"', ''),
                    errors='coerce'
                ).fillna(0)
        
        # ticker 정제 (공백, 소문자 → 대문자)
        df['ticker'] = df['ticker'].astype(str).str.strip().str.upper().str.replace("'", "")
        
        # 빈 행 제거
        df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan') & (df['qty'] > 0)]
        
        # total_price 가 있으면 avg_price 계산 (숫자 변환 후)
        if 'total_price' in df.columns and 'avg_price' not in df.columns:
            try:
                df['total_price'] = pd.to_numeric(
                    df['total_price'].astype(str).str.replace(',', '').str.replace('"', ''),
                    errors='coerce'
                ).fillna(0)
                df['avg_price'] = df.apply(
                    lambda row: float(row['total_price']) / float(row['qty']) if row['qty'] > 0 else 0,
                    axis=1
                )
            except Exception as e:
                logger.warning(f"avg_price 계산 실패: {e}")
        
        holdings = []
        for _, row in df.iterrows():
            holdings.append({
                'ticker': row['ticker'],
                'qty': int(row['qty']),
                'price': float(row.get('avg_price', row['current_price'])),  # P7: price 키 추가 (하위호환)
                'current_price': float(row['current_price']),
                'avg_price': float(row.get('avg_price', 0)),
                'stop_loss': float(row.get('stop_loss', 0)),
                'expert_score': 0,  # 나중에 채워짐
                'rs_score': 0,
                'trend_score': 0,
                'sector': '',
            })
            logger.info(f"  ✅ {row['ticker']}: {int(row['qty'])}주 @ ${float(row['current_price']):.2f}")
        
        logger.info(f"보유종목 {len(holdings)}개 로드 완료")
        return holdings
    
    except Exception as e:
        logger.error(f"CSV 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    csv_path = r'C:\Quant\holdings.csv'
    holdings = load_holdings_csv(csv_path)
    print(f"\n로드된 보유종목: {len(holdings)}개")
    for h in holdings:
        print(f"  {h['ticker']}: {h['qty']}주 @ ${h['current_price']:.2f}")
