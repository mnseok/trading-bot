# AI 비트코인 트레이딩 봇

AI를 활용한 비트코인 자동매매 봇입니다. Upbit API를 사용하여 비트코인 거래를 수행하고, Gemini AI를 활용하여 매매 의사결정을 합니다.

## 주요 기능

- **자동 매매**: AI가 시장 데이터를 분석하여 매수/매도/홀딩 결정
- **기술적 분석**: 다양한 기술적 지표(MACD, RSI, 볼린저밴드 등) 활용
- **뉴스 분석**: 최신 BTC 관련 뉴스 수집 및 분석 
- **차트 분석**: Upbit 차트 스크린샷 자동 캡처
- **거래 기록**: SQLite DB에 모든 거래 내역 저장
- **거래 반성**: AI가 각 거래에 대한 반성문 작성
- **웹 인터페이스**: Flask 기반 거래 내역 조회 페이지

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/mnseok/trading-bot.git
cd trading-bot
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경변수 설정
`.env` 파일을 생성하고 다음 내용을 입력:
```
UPBIT_ACCESS_KEY=<your-upbit-access-key>
UPBIT_SECRET_KEY=<your-upbit-secret-key>
SERPAPI_KEY=<your-serpapi-key>
GEMINI_API_KEY=<your-gemini-api-key>
```

## 실행 방법

1. 트레이딩 봇 실행
```bash
python mvp.py
```

2. 웹 인터페이스 실행 (별도 터미널)
```bash
python app.py
```
웹 브라우저에서 `http://127.0.0.1:5000` 접속

## 주요 구성 요소

- `mvp.py`: 메인 트레이딩 봇 로직
  - 시장 데이터 수집 및 분석
  - AI 기반 매매 의사결정
  - 거래 실행 및 기록
  - 거래 반성문 생성
- `app.py`: Flask 웹 서버
  - 거래 내역 조회 페이지 제공
  - 실시간 거래 현황 모니터링
- `trading_records.db`: SQLite 데이터베이스
  - 거래 기록 저장
  - 반성문 및 개선점 저장
- `templates/`: 웹 페이지 템플릿

## 작동 방식

1. 시장 데이터 수집
   - Upbit API를 통한 OHLCV 데이터 수집
   - 오더북 데이터 분석
   - 기술적 지표 계산 (MACD, RSI, 볼린저밴드 등)
   - BTC 관련 뉴스 수집
   - Upbit 차트 스크린샷 캡처

2. AI 분석 및 의사결정
   - Gemini AI를 활용한 데이터 분석
   - 매수/매도/홀딩 결정
   - 거래 비율 결정

3. 거래 실행
   - Upbit API를 통한 실제 거래 실행
   - 거래 내역 DB 저장
   - AI 기반 거래 반성문 작성

4. 모니터링
   - Flask 웹 인터페이스를 통한 실시간 모니터링
   - 거래 내역 및 반성문 조회

## 주요 기술적 지표

- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- 볼린저 밴드 (Bollinger Bands)
- 켈트너 채널 (Keltner Channel)
- 스토캐스틱 (Stochastic)
- ADX (Average Directional Index)
- OBV (On Balance Volume)
- VWAP (Volume Weighted Average Price)

## 주의사항

- 실제 자금을 거래하므로 신중하게 사용
- API 키는 절대 공개하지 말 것
- 테스트 환경에서 충분히 검증 후 실제 거래 시작 권장
- 거래 금액과 빈도는 본인의 위험 감수 수준에 맞게 설정

## 개선 계획

- [ ] 다양한 암호화폐 동시 거래 지원
- [ ] 더 많은 기술적 지표 추가
- [ ] 백테스팅 기능 구현
- [ ] 텔레그램 알림 기능 추가
- [ ] AI 모델 성능 개선

## 라이선스

MIT License

## 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 문의사항

이슈를 통해 문의해주세요.
