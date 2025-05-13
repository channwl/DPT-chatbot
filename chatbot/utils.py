from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from typing import List
from django.conf import settings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationSummaryMemory
from langsmith import Client
import traceback
import os
from dotenv import load_dotenv


class PDFProcessor:
    @staticmethod
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        for d in documents:
            d.metadata['file_path'] = pdf_path
        return documents

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)


def generate_faiss_index():
    pdf_dir = "data/"
    all_documents = []

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print("✅ data/ 폴더 생성 완료 (PDF 파일 추가 필요)")
        return

    pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ data/ 폴더에 PDF 파일이 없습니다.")
        return

    for file_name in pdf_files:
        docs = PDFProcessor.pdf_to_documents(os.path.join(pdf_dir, file_name))
        all_documents.extend(docs)

    chunks = PDFProcessor.chunk_documents(all_documents)
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY
)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_internal")
    print(f"✅ {len(pdf_files)}개의 PDF 파일로 인덱스 생성 완료!")


class RAGSystem:
    def __init__(self):
        self.langsmith_client = Client()

        #Thinking 토큰 수 제한 설정 포함
        self.llm = ChatOpenAI(
            model_name="gpt-4.1-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
)

        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            return_messages=True,
            max_token_limit=300,
            memory_key="history",
            input_key="input",
            output_key="output"
        )

        self.rag_chain = self.get_rag_chain()

    def get_vector_db(self):
        if not os.path.exists("faiss_index_internal"):
            print("⚡ faiss_index_internal 폴더 없음, 인덱스 자동 생성 시작")
            generate_faiss_index()

        embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 또는 "text-embedding-ada-002"
        openai_api_key=settings.OPENAI_API_KEY
)
        return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)

    def get_rag_chain(self):
        template = """
        당신은 전문 챗봇입니다. 사용자의 질문에 대해 정확하고 간결하며 HTML로 출력 가능한 답변을 작성하세요.

###역할
- 사용자의 질문에 대해 <strong>정확하고 유익한 정보</strong>를 제공합니다.
- <strong>짧은 문장</strong>과 <strong>불릿포인트</strong> 형식을 사용해 시각적으로 정리합니다.
- HTML 환경에 적합한 마크업을 포함해야 합니다.

###입력 형식
- 질문 (예: 사용자의 궁금증 또는 정보 요청)
- 관련 컨텍스트 (예: PDF에서 추출된 학과 정보)
- 이전 대화 요약

###응답 구성 규칙
- 첫 문장은 요약 개요입니다.  
- 이후 불릿포인트 형식으로 핵심 내용을 정리합니다.  
- 각 항목 앞에 `- `를 붙이고, <strong>태그</strong>로 키워드를 강조합니다.  
- 응답은 HTML로 출력되는 환경을 고려하여 마크업을 포함해야 합니다.
- 세부 항목이 있는 경우, `&nbsp;&nbsp;- 항목 내용` 형식으로 **들여쓰기**하세요.
- 각 단락은 <br> 또는 줄바꿈을 명확하게 삽입하세요.
- 사용자가 질문한 내용이 이전 대화에서 다룬 사항과 연결될 경우, 이전에 제공된 정보를 반영하여 답변해주세요.

###제약조건
- 내용이 너무 길어지지 않도록 간결함을 유지하세요.
- 질문이 졸업요건 관련일 경우, "전공 필수 과목"을 반드시 포함하세요.
- 모델은 PDF 내에서 정보를 찾아 누락 없이 반영해야 합니다.
- PDF에 영어로 된 부분이 있다면 한글로 해석해서 답변해주세요.

###예시
질문: 23학번 졸업요건이 뭐야?  
컨텍스트: (PDF 문서에서 추출된 졸업요건 관련 내용)  
출력 예시:

답변:  
23학번의 졸업요건은 다음과 같습니다.  
- <strong>전공 필수 과목</strong>은 다음과 같습니다: 경영통계, 회계원리, 데이터사이언스입문  
- 총 이수학점은 <strong>130학점</strong>이며, 전공 60학점 이상이 요구됩니다.  
- <strong>졸업논문 또는 대체과목</strong>도 이수 조건에 포함됩니다.  
- 추가로 궁금한 점이 있다면 언제든지 말씀해주세요.

##세부제약조건
- 교수진에 대한 정보를 물어보면 "조직도"의 PDF를 통해서만 답변해주세요

---

###프롬프트 구조
입력 형식:  
- 이전 대화 요약: {history}  
- 컨텍스트: {context}  
- 질문: {question}  

출력 형식:  
- 위 규칙을 따릅니다.  
        """
        prompt = PromptTemplate.from_template(template)

        def retrieve_context(inputs: dict):
            vector_db = self.get_vector_db()
            retriever = vector_db.as_retriever(search_kwargs={"k": 5})
            return retriever.invoke(inputs["question"])

        return RunnableMap({
            "question": lambda x: x["question"],
            "context": RunnableLambda(retrieve_context),
            "history": lambda x: x["history"]
        }) | prompt | self.llm | StrOutputParser()

    def process_question(self, question: str) -> str:
        try:
            vector_db = self.get_vector_db()
            retriever = vector_db.as_retriever(search_kwargs={"k": 7})
            docs = retriever.invoke(question)
            history_summary = self.memory.load_memory_variables({})['history']

            answer = self.rag_chain.invoke({
                "question": question,
                "context": docs,
                "history": history_summary
            })

            self.memory.save_context({"input": question}, {"output": answer})
            return answer

        except Exception as e:
            print("오류 발생!")
            print(f"오류 종류: {type(e).__name__}")
            print(f"오류 메시지: {e}")
            traceback.print_exc()
            return "질문 처리 중 오류가 발생했습니다."
