import time
from pywinauto import Application
import sys
import io

# 한글 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

def auto_order():
    print("\n>>> [주문 엔진 가동] target.txt 파일을 확인합니다...")
    
    # 1. 메모장에서 매수할 종목 읽어오기
    try:
        with open("target.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            print(">>> [알림] 매수 대상이 없습니다. (target.txt 비어있음)")
            return

        # 콤마(,)로 구분된 종목과 수량을 분리
        ticker, qty = content.split(',')
        qty = int(qty)
        print(f">>> [타겟 확인] 종목: {ticker} / 수량: {qty}주 -> 주문 준비!")

    except FileNotFoundError:
        print(">>> [오류] target.txt 파일이 없습니다. ro.py가 먼저 실행되어야 합니다.")
        return
    except ValueError:
        print(">>> [오류] target.txt 내용 형식이 잘못되었습니다. (예: PM,35)")
        return

    # 2. 키움증권 HTS 찾아서 주문 넣기
    try:
        print(">>> [HTS 연결] 영웅문 Global을 찾는 중...")
        app = Application().connect(title_re=".*영웅문Global.*")
        win = app.window(title_re=".*영웅문Global.*")
        win.set_focus() # 창을 맨 앞으로
        
        # [4989] 미니주문 창이 켜져 있다고 가정
        print(f">>> [입력 시작] {ticker} 입력 중...")
        
        # (1) 종목코드 입력
        win.type_keys(ticker) 
        win.type_keys("{ENTER}")
        time.sleep(1.5) # 로딩 대기
        
        # (2) 수량 입력 (탭 2번 -> 환경에 따라 탭 횟수 조절 필요할 수 있음)
        win.type_keys("{TAB 2}")
        win.type_keys(str(qty))
        
        # (3) 매수 (F9)
        time.sleep(0.5)
        print(">>> [주문 전송] F9 키 입력!")
        win.type_keys("{F9}")
        
        print(f">>> ✅ {ticker} {qty}주 매수 주문 전송 완료!")
        
    except Exception as e:
        print(f">>> ❌ 주문 실패: {e}")
        print("    (영웅문 Global이 켜져 있는지, [4989]창이 열려 있는지 확인하세요)")

if __name__ == "__main__":
    auto_order()