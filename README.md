# 🚀 통합 퀀트 시스템 V24.9

> 13612W 모멘텀 + 카나리아 신호등 + Turtle Trading

[![Version](https://img.shields.io/badge/version-24.9-blue.svg)](https://github.com/yourusername/my-quant)
[![Python](https://img.shields.io/badge/python-3.11-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-gray.svg)](LICENSE)

---

## 📋 특징

### 🎯 핵심 기능
- **13612W 모멘텀**: Keller 의 가중치 모멘텀 (12×1M + 4×3M + 2×6M + 1×12M) / 19
- **카나리아 신호등**: 3 레이어 시장 분석 (공격/주의/방어 모드)
- **Turtle Trading**: Donchian Channel 기반 추세 추종
- **Expert 전략**: 펀더멘털 우량주 선별
- **RS 전략**: 상대강도 주도주 선별

### 📊 전략 구성
| 전략 | 설명 | 선정비율 |
|------|------|----------|
| **Turtle** | 20 일 최고가 돌파 + ATR 포지셔닝 | 2-5% |
| **Expert** | ROE, 성장률, 부채비율 종합평가 | 15-25% |
| **RS** | 63 일 상대강도 모멘텀 | 20-35% |

### 🔍 카나리아 신호등
```
🟢 공격 모드: 음수 0/4 개 (SPY, VWO, VEA, BND 모두 정상)
🟡 주의 모드: 음수 1-2 개
🔴 방어 모드: 음수 3-4 개 (현금 비중 확대)
```

---

## 📁 프로젝트 구조

```
my-quant/
├── quant_run.py              # 메인 실행 파일
├── config_manager.py         # 설정 관리
├── holdings_loader.py        # 보유 종목 로드
├── backtester.py             # 백테스터
│
├── core/                     # 핵심 모듈
│   ├── config.py
│   ├── data_loader.py
│   ├── fundamentals.py
│   ├── history_store.py      # SQLite 히스토리
│   ├── macro.py              # 카나리아 분석
│   ├── order_history.py
│   ├── quant_system.py
│   ├── rebalance.py
│   ├── universe_expander.py
│   └── strategies/
│       ├── expert.py
│       ├── rs.py
│       └── turtle.py
│
├── utils/                    # 유틸리티
│   ├── momentum.py           # 13612W 계산
│   ├── html_utils.py
│   └── sector_mapper.py
│
├── reports/                  # 리포트
│   └── html_builder.py
│
├── warning/                  # 경고 시스템
│   └── early_warning.py
│
├── Data/                     # 데이터 (gitignore)
├── Reports/                  # 리포트 (gitignore)
├── Charts/                   # 차트 (gitignore)
│
├── config.json.example       # 설정 예시
├── requirements.txt          # 의존성
└── README.md                 # 이 파일
```

---

## 🚀 빠른 시작

### 1. 설치
```bash
# Python 3.11+ 필요
git clone https://github.com/yourusername/my-quant.git
cd my-quant
pip install -r requirements.txt
```

### 2. 설정
```bash
# config.json.example 복사
cp config.json.example config.json

# API 키 설정 (선택사항)
# - Polygon.io: 러셀 2000 동적 수집
# - FMP: 재무제표 데이터
```

### 3. 실행
```bash
# 전체 시스템 실행
python quant_run.py

# 백테스트 실행
python backtester.py
```

### 4. 결과 확인
```
Reports/report_YYYY-MM-DD.html  # HTML 리포트
system.log                       # 실행 로그
```

---

## 📊 사용 예시

### 카나리아 신호 확인
```python
from core.macro import CanaryAnalyzer

canary = CanaryAnalyzer()
mode, score = canary.analyze()

print(f"마켓 모드: {mode}")  # 🟢 공격 / 🟡 주의 / 🔴 방어
print(f"마켓 스코어: {score}")  # 0-100
```

### Turtle 진입 신호
```python
from core.strategies.turtle import TurtleStrategy

turtle = TurtleStrategy()
signal = turtle.check_entry(df, current_price)

if signal['signal']:
    print(f"진입 신호: {signal['reason']}")
    print(f"ATR: {signal['atr']}")
```

### 13612W 모멘텀
```python
from utils.momentum import calculate_momentum_13612

prices = [100, 102, 105, ...]  # 종가 목록
momentum = calculate_momentum_13612(prices)

print(f"13612W 모멘텀: {momentum:.2f}%")
```

---

## ⚙️ 설정

### config.json
```json
{
    "account": {
        "total_krw": 76826207,
        "risk_ratio": 0.01,
        "max_alloc_per_stock": 0.20,
        "expert_cutoff": 70,
        "min_invest_per_stock": 10000000,
        "half_kelly_ratio": 0.08,
        "min_cash_buffer": 0.05
    },
    "data": {
        "chunk_size": 50,
        "delay_between_chunks": 2.0,
        "download_timeout": 30,
        "download_period": "1y",
        "macro_period": "6mo",
        "rs_period_days": 63,
        "volume_avg_days": 21
    },
    "exchange": {
        "default_rate": 1450.0,
        "cache_max_age_hours": 24
    },
    "api_keys": {
        "polygon_api_key": "your_key_here",
        "fmp_api_key": "your_key_here"
    }
}
```

---

## 📈 전략 상세

### Turtle S1 진입
1. **20 일 최고가 돌파** (Donchian Upper)
2. **40 일 필터** (이전 저가돌파 이력 확인)
3. **ATR 기반 포지셔닝** (1% 리스크)
4. **피라미딩** (0.5 ATR 간격, 최대 4 Unit)
5. **손절가** (2 ATR 아래)

### Expert 점수 계산
```
점수 = ROE(30%) + 성장률(20%) + RS(25%) + 배당(10%) + PBR(15%)

섹터별 가중치 차등:
- Technology: 성장률 30%, ROE 15%
- Financial: ROE 35%, PBR 20%
- Energy: 배당 30%, ROE 20%
```

### RS (상대강도)
- **기간**: 63 일 (3 개월)
- **비교**: S&P500 대비 초과수익률
- **랭킹**: 0-100 백분위
- **13612W**: 가중치 모멘텀 병합

---

## 🔧 의존성

```txt
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.28
requests>=2.31.0
matplotlib>=3.7.0
pydantic>=2.0.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
```

---

## 📝 라이선스

MIT License - [LICENSE](LICENSE) 파일 참조

---

## 🤝 기여

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 연락처

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다.  
투자 책임은 전적으로 사용자에게 있으며, 개발자는 어떠한 손실도 책임지지 않습니다.

---

**마지막 업데이트**: 2026-03-05  
**버전**: V24.9 (13612W + Canary)
