# 📌 프로젝트 개요
이 프로젝트는 고려대학교 세종캠퍼스 **디지털경영전공 학부생**을 위한 정보 제공 챗봇입니다. <br>
LangChain 기반 RAG(Retrieval-Augmented Generation) 방식을 사용하여 학과 안내 PDF에서 필요한 정보를 검색·요약하고 HTML 형식으로 제공하며, Django 웹 인터페이스를 통해 실시간 질의응답이 가능합니다.

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
