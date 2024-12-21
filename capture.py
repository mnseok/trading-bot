import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

def take_fullpage_screenshot(url: str, screenshot_name: str = "full_screenshot.png"):
    # Chromedriver 경로 설정
    chrome_driver_path = "/opt/homebrew/bin/chromedriver"
    service = Service(executable_path=chrome_driver_path)
    
    # Chrome 옵션 설정 (필요에 따라 headless 등 추가)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    # 드라이버 실행
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 10)

    try:
        # 해당 URL 접속
        driver.get(url)
        
        # 페이지 로딩 대기(필요시 조정)
        time.sleep(1)

        # --------------------------------------------------
        # 1. 버튼(메뉴) 클릭
        #    /html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]
        # --------------------------------------------------
        menu_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]')
        menu_button.click()
        time.sleep(0.5)

        # --------------------------------------------------
        # 2. '1시간' 텍스트 선택
        #    일반적으로 xpath에서 텍스트 포함 여부로 클릭하는 것이 유연할 수 있음
        #    예: //span[contains(text(), "1시간")]
        # --------------------------------------------------
        one_hour_button = wait.until(EC.presence_of_element_located(
            (By.XPATH, '//cq-item[@stxtap="Layout.setPeriodicity(1,60,\'minute\')"]')
        ))
        one_hour_button.click()
        
        # --------------------------------------------------
        # 3. 지표 버튼 클릭
        # --------------------------------------------------
        indicator_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]')
        indicator_button.click()
        
        # --------------------------------------------------
        # 4. '볼린저 밴드' 텍스트 포함 요소 클릭
        # --------------------------------------------------
        bollinger_band_button = wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR,
            "#fullChartiq > div > div > div.ciq-nav > div > div > cq-menu.ciq-menu.ciq-studies.collapse.stxMenuActive > cq-menu-dropdown > cq-scroll > cq-studies > cq-studies-content > cq-item:nth-child(15)"
        )))
        bollinger_band_button.click()

        # --------------------------------------------------
        # 3. 화면 캡처
        # --------------------------------------------------
        # 전체 화면으로 전환
        driver.fullscreen_window()
        time.sleep(1)
        
        # 스크린샷 촬영
        driver.save_screenshot(screenshot_name)
        print(f"전체화면 캡처 저장 완료: {os.path.abspath(screenshot_name)}")
        
    except Exception as e:
        print("에러가 발생했습니다:", e)
    finally:
        # 드라이버 종료
        driver.quit()

if __name__ == "__main__":
    url = "https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC"
    take_fullpage_screenshot(url, "upbit_full_screenshot.png")
