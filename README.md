# 📌 프로젝트 개요
이 프로젝트는 고려대학교 세종캠퍼스 **디지털경영전공 학부생**을 위한 정보 제공 챗봇입니다.  
LangChain 기반 RAG(Retrieval-Augmented Generation) 방식을 사용하여 학과 안내 PDF에서 필요한 정보를 검색·요약하고 HTML 형식으로 제공하며, Django 웹 인터페이스를 통해 실시간 질의응답이 가능합니다.

---

# 📂 프로젝트 구조
DPT-chatbot/
├─ chatbot/ # Django 앱 (UI 및 RAG 로직)
│ ├─ pycache/ # Python 캐시 파일
│ ├─ static/chatbot/ # 정적 리소스 (CSS, JS, 이미지)
│ │ └─ style.css # 챗봇 UI 스타일 시트
│ ├─ templates/chatbot/ # HTML 템플릿
│ │ └─ chatbot.html # 메인 챗봇 화면
│ ├─ apps.py # Django 앱 설정
│ ├─ urls.py # URL 라우팅 설정
│ ├─ utils.py # PDF 처리, 인덱싱, RAG 시스템 구현
│ └─ views.py # 사용자 요청 처리 및 응답 반환
│
├─ data/ # 학과 안내 PDF 보관 폴더
│
├─ dpt_env/ # 환경설정(로컬/개발용, 배포 제외)
│
├─ dpt_project/ # Django 프로젝트 설정
│ ├─ pycache/
│ ├─ init.py
│ ├─ asgi.py # ASGI 설정
│ ├─ settings.py # Django 설정 파일
│ ├─ urls.py # 전역 URL 매핑
│ └─ wsgi.py # WSGI 설정 (배포용)
│
├─ faiss_index_internal/ # FAISS 인덱스 저장 폴더
│
├─ venv/ # Python 가상환경
│
├─ .DS_Store # macOS 시스템 파일
├─ db.sqlite3 # SQLite 데이터베이스
├─ manage.py # Django 관리 스크립트


---

# 📝 주요 모듈 설명
- **chatbot/utils.py**
  - `PDFProcessor`: PDF → Document 로드, 청킹 처리
  - `generate_faiss_index()`: PDF 데이터로 FAISS 벡터 인덱스 생성
  - `RAGSystem`:  
    - LLM 및 메모리 초기화  
    - Retriever & PromptTemplate 기반 RAG 체인 구성  
    - `process_question()`로 질문 처리 및 HTML 응답 생성
- **data/**
  - 학과 안내 PDF 원본 저장
- **faiss_index_internal/**
  - 인덱스 생성 후 저장되는 벡터 데이터
- **templates/**
  - 웹 UI HTML 템플릿
- **static/**
  - CSS, JS, 이미지 등 정적 리소스
- **views.py**
  - 사용자 요청 처리, 챗봇 응답 반환
- **urls.py**
  - API 엔드포인트 라우팅

---

# ⚙️ 동작 흐름
1. **PDF 준비** → `data/` 폴더에 PDF 저장  
2. **인덱스 생성** → `generate_faiss_index()` 실행, `faiss_index_internal/`에 저장  
3. **사용자 질문 입력** → Django View에서 `RAGSystem.process_question()` 호출  
4. **문서 검색** → FAISS Retriever로 관련 Chunk 검색  
5. **응답 생성** → PromptTemplate + LLM을 통해 HTML 형식 응답 생성  
6. **웹 UI 출력** → chatbot.html에서 실시간 응답 표시
