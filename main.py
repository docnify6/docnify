import os
import io
import json
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Request, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
import PyPDF2
import google.genai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.auth.transport.requests import Request as GoogleRequest
import firebase_admin
from firebase_admin import credentials, auth, firestore
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

load_dotenv()

# Initialize Firebase Admin SDK
service_account_info = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

if not firebase_admin._apps:
    try:
        if service_account_info:
            if service_account_info.startswith('{'):
                # It's JSON content (for production deployment)
                cred = credentials.Certificate(json.loads(service_account_info))
                print("✅ Firebase initialized with service account JSON")
            elif os.path.exists(service_account_info):
                # It's a file path (for local development)
                cred = credentials.Certificate(service_account_info)
                print("✅ Firebase initialized with service account file")
            else:
                raise FileNotFoundError("Service account file not found")
            firebase_admin.initialize_app(cred)
        else:
            raise FileNotFoundError("Service account info not found")
    except Exception as e:
        print(f"❌ Failed to initialize Firebase with service account: {e}")
        print("Please ensure the FIREBASE_SERVICE_ACCOUNT_JSON environment variable is set correctly")
        raise e

# Initialize Firestore
db = firestore.client()

app = FastAPI(title="Docnify API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://docnifyai.web.app", "https://docnify-e0j1.onrender.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Docnify API is running" }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}



# API Key Pool Management
def get_gemini_clients():
    """Create list of Gemini clients from available API keys"""
    clients = []
    for i in range(1, 21):  # Try keys 1-20
        api_key = os.getenv(f"GEMINI_API_KEY_{i}")
        if api_key and api_key != f"YOUR_API_KEY_{i}":  # Skip placeholder keys
            try:
                clients.append(genai.Client(api_key=api_key))
            except Exception as e:
                print(f"Failed to initialize client for key {i}: {e}")
    return clients

clients = get_gemini_clients()

def call_gemini_with_fallback(prompt: str, model: str = 'gemini-2.5-flash'):
    """Call Gemini API with automatic fallback to next available key"""
    last_error = None

    for i, client in enumerate(clients):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response
        except Exception as e:
            error_str = str(e).lower()
            # Check for quota/rate limit errors
            if '429' in error_str or 'resource_exhausted' in error_str or 'quota' in error_str:
                print(f"API key {i+1} quota exhausted, trying next key...")
                last_error = e
                continue
            else:
                # For other errors (auth, network, etc.), fail immediately
                raise e

    # If all keys failed due to quota, raise the last error
    if last_error:
        raise last_error

    # If no clients available
    raise HTTPException(status_code=500, detail="No valid Gemini API keys configured")

# In-memory storage for temporary data (documents and sessions)
document_store = {}
user_sessions = {}  # Keep sessions in memory for simplicity
ip_usage = {}  # ip -> {count, last_reset}

# Google Drive OAuth setup
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

def store_oauth_state(state: str, user_id: str):
    db.collection("oauth_states").document(state).set({
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc)
    })

def pop_oauth_state(state: str) -> str | None:
    ref = db.collection("oauth_states").document(state)
    doc = ref.get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    created_at = data.get("created_at")
    if datetime.now(timezone.utc) - created_at > timedelta(minutes=10):
        ref.delete()
        return None
    user_id = data.get("user_id")
    ref.delete()
    return user_id

class DocumentExplanation(BaseModel):
    summary: str
    action_required: str
    deadline: str
    risk_if_ignored: str
    highlights: List[str]
    document_id: Optional[str] = None
    requires_auth: bool = False
    requires_verification: bool = False

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    source_section: str

class AuthRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user_id: str

UNIVERSAL_DOCNIFY_PROMPT = """You are Docnify, a document explanation engine.
The document content may come from text, images, or scanned pages.
Treat all provided content as the full document.
Explain everything in very simple English.
Do not assume the document is complete or perfect.
Explain what the document is trying to say overall.
Convert important information into clear actions if mentioned.
If deadlines are written anywhere, explain them clearly.
If risks or consequences are written, explain them simply.
If something is unclear or missing, say it is not mentioned.
Do not add outside knowledge or guess anything."""

SIMPLE_PROMPT = f"""{UNIVERSAL_DOCNIFY_PROMPT}

SIMPLE MODE - Think fast, extract only the most important information.

OUTPUT LIMITS:
- What is this about? → max 3 sentences
- What should you do? → 3 to 4 steps
- Any deadline? → 1 sentence
- What if you ignore it? → 1 sentence
- Important highlights → max 5 bullet points

Respond with ONLY a JSON object:
{{
  "summary": "What this document is about in max 3 simple sentences",
  "action_required": "Numbered steps of what to do, or 'No action required' if none needed",
  "deadline": "Exact dates if present, or 'No deadline mentioned' if none",
  "risk_if_ignored": "Real consequences if mentioned, or 'No risk mentioned' if none",
  "highlights": ["Important fact 1", "Important fact 2", "Important fact 3"]
}}"""

DETAILED_PROMPT = f"""{UNIVERSAL_DOCNIFY_PROMPT}

DETAILED MODE - Think carefully, provide deeper understanding.

OUTPUT EXPECTATIONS:
- What is this about? → 4 to 6 sentences
- What should you do? → 6 to 10 clear steps
- Any deadline? → explain what the deadline applies to
- What if you ignore it? → 2 to 3 sentences
- Important highlights → 8 to 12 bullet points

Respond with ONLY a JSON object:
{{
  "summary": "What this document is about in 4-6 simple sentences with more explanation",
  "action_required": "Numbered steps of what to do with explanations, or 'No action required' if none needed",
  "deadline": "Exact dates with explanation of what the deadline applies to, or 'No deadline mentioned' if none",
  "risk_if_ignored": "Real consequences explained in 2-3 sentences, or 'No risk mentioned' if none",
  "highlights": ["Important fact 1", "Important fact 2", "Important fact 3", "Important fact 4", "Important fact 5", "Important fact 6", "Important fact 7", "Important fact 8"]
}}"""

QA_SYSTEM_PROMPT = f"""{UNIVERSAL_DOCNIFY_PROMPT}

QUESTION MODE - Answer questions ONLY using the provided document text.

STRICT RULES:
- Answer only what the document contains
- If the answer is not in the document, say: "This document does not mention this."
- Use very simple English
- Short and direct answers
- No legal or technical jargon
- No assumptions or guesses

Respond with ONLY a JSON object:
{{
  "answer": "Clear explanation in simple words",
  "source_section": "Quote or describe the section briefly"
}}"""

def check_ip_usage(client_ip: str) -> bool:
    """Check if IP has exceeded usage limit"""
    today = datetime.now().date()
    
    if client_ip not in ip_usage:
        ip_usage[client_ip] = {"count": 0, "last_reset": today}
        return True
    
    usage = ip_usage[client_ip]
    
    # Reset count if new day
    if usage["last_reset"] != today:
        usage["count"] = 0
        usage["last_reset"] = today
    
    return usage["count"] < 5

def increment_ip_usage(client_ip: str):
    """Increment IP usage count"""
    today = datetime.now().date()
    
    if client_ip not in ip_usage:
        ip_usage[client_ip] = {"count": 1, "last_reset": today}
    else:
        ip_usage[client_ip]["count"] += 1

def get_current_user(authorization: str = Header(None)):
    """Auth check using Firebase ID token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")

    token = authorization.split(" ")[1]
    try:
        # Verify Firebase ID token
        decoded_token = auth.verify_id_token(token)
        return {
            "user_id": decoded_token["uid"],
            "email": decoded_token["email"]
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

def store_document_text(user_id: str, text: str, pdf_content: bytes = None, explanation: dict = None) -> str:
    """Store document text and optionally PDF"""
    doc_id = str(uuid.uuid4())
    # For authenticated users, don't set expiry (permanent storage)
    # For temp users, expire after 24 hours
    expires_at = None if user_id != "temp" else datetime.now() + timedelta(hours=24)
    document_store[doc_id] = {
        "text": text,
        "user_id": user_id,
        "pdf_content": pdf_content,
        "explanation": explanation,
        "expires_at": expires_at,
        "drive_file_id": None
    }
    return doc_id

def get_document_text(doc_id: str, user_id: str) -> str:
    """Retrieve document text for user"""
    if doc_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = document_store[doc_id]
    if doc["user_id"] != user_id and doc["user_id"] != "temp":
        raise HTTPException(status_code=403, detail="Access denied")

    if doc["expires_at"] and datetime.now() > doc["expires_at"]:
        del document_store[doc_id]
        raise HTTPException(status_code=404, detail="Document expired")

    return doc["text"]

def get_document_pdf(doc_id: str, user_id: str) -> bytes:
    """Retrieve document PDF for user"""
    if doc_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = document_store[doc_id]
    if doc["user_id"] != user_id and doc["user_id"] != "temp":
        raise HTTPException(status_code=403, detail="Access denied")

    if doc["expires_at"] and datetime.now() > doc["expires_at"]:
        del document_store[doc_id]
        raise HTTPException(status_code=404, detail="Document expired")

    if "pdf_content" not in doc or not doc["pdf_content"]:
        raise HTTPException(status_code=404, detail="PDF content not available")

    return doc["pdf_content"]

def get_google_drive_service(user_id: str):
    """Get Google Drive service for authenticated user"""
    # Get user document from Firestore
    user_doc = db.collection('users').document(user_id).get()
    if not user_doc.exists:
        raise HTTPException(status_code=400, detail="User not found")

    user_data = user_doc.to_dict()
    if "google_drive_token" not in user_data:
        raise HTTPException(status_code=400, detail="Google Drive not connected")

    token_data = user_data["google_drive_token"]
    credentials = Credentials(
        token=token_data.get("access_token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri"),
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes")
    )

    if credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleRequest())
        db.collection("users").document(user_id).update({
            "google_drive_token.access_token": credentials.token
        })

    return build('drive', 'v3', credentials=credentials)

def upload_to_google_drive(user_id: str, pdf_content: bytes, filename: str) -> str:
    """Upload PDF to user's Google Drive"""
    try:
        service = get_google_drive_service(user_id)

        file_metadata = {
            'name': filename,
            'mimeType': 'application/pdf'
        }

        media = MediaIoBaseUpload(io.BytesIO(pdf_content), mimetype='application/pdf')

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        return file.get('id')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Drive upload failed: {str(e)}")

def generate_explanation_pdf(explanation: dict, doc_id_prefix: str) -> bytes:
    """Generate PDF from explanation data with better formatting"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=28,
        spaceAfter=40,
        alignment=1,  # Center
        textColor='#667eea',
        fontName='Helvetica-Bold'
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=10,
        spaceBefore=20,
        textColor='#333333',
        fontName='Helvetica-Bold'
    )

    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,
        spaceAfter=12
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        leftIndent=30,
        spaceAfter=8
    )

    story = []

    # Title
    story.append(Paragraph("Docnify Analysis", title_style))
    story.append(Paragraph(f"Document ID: {doc_id_prefix}", ParagraphStyle('DocId', parent=styles['Normal'], fontSize=10, alignment=1, textColor='#666666')))
    story.append(Spacer(1, 0.3*inch))

    # Summary
    story.append(Paragraph("📋 What is this about?", section_style))
    summary = explanation.get('summary', 'N/A')
    story.append(Paragraph(summary, content_style))

    # Action Required
    story.append(Paragraph("⚡ What should you do?", section_style))
    action_text = explanation.get('action_required', 'N/A')
    if isinstance(action_text, str):
        # Split by newlines and format as numbered list
        actions = [line.strip() for line in action_text.split('\n') if line.strip()]
        for i, action in enumerate(actions, 1):
            story.append(Paragraph(f"{i}. {action}", content_style))
    else:
        story.append(Paragraph(str(action_text), content_style))

    # Deadline
    story.append(Paragraph("⏰ Any deadline?", section_style))
    deadline = explanation.get('deadline', 'N/A')
    story.append(Paragraph(deadline, content_style))

    # Risk
    story.append(Paragraph("⚠️ What if you ignore it?", section_style))
    risk = explanation.get('risk_if_ignored', 'N/A')
    story.append(Paragraph(risk, content_style))

    # Highlights
    story.append(Paragraph("✨ Important highlights", section_style))
    highlights = explanation.get('highlights', [])
    if isinstance(highlights, list):
        for highlight in highlights:
            story.append(Paragraph(f"• {highlight}", bullet_style))
    else:
        story.append(Paragraph(str(highlights), content_style))

    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Docnify", ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, alignment=1, textColor='#999999')))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def get_or_create_docnify_folder(service) -> str:
    """Get or create Docnify folder in Google Drive"""
    # Check if Docnify folder exists
    query = "name = 'Docnify' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields='files(id, name)').execute()
    files = results.get('files', [])

    if files:
        return files[0]['id']
    else:
        # Create Docnify folder
        file_metadata = {
            'name': 'Docnify',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

def upload_pdf_to_drive(user_id: str, pdf_content: bytes, filename: str) -> str:
    """Upload PDF to user's Google Drive Docnify folder, checking for duplicates"""
    try:
        service = get_google_drive_service(user_id)

        # Get or create Docnify folder
        folder_id = get_or_create_docnify_folder(service)

        # Check if file with same name already exists in the folder
        query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
        existing_files = service.files().list(q=query, fields='files(id, name)').execute()

        if existing_files.get('files'):
            raise HTTPException(status_code=409, detail=f"File '{filename}' already exists in Docnify folder")

        file_metadata = {
            'name': filename,
            'mimeType': 'application/pdf',
            'parents': [folder_id]
        }

        media = MediaIoBaseUpload(io.BytesIO(pdf_content), mimetype='application/pdf')

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        return file.get('id')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Drive upload failed: {str(e)}")

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text from PDF file with OCR fallback for images"""
    try:
        pdf_file.file.seek(0)
        pdf_content = pdf_file.file.read()
        
        if not pdf_content:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        # First try regular text extraction
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # If we got meaningful text, return it
            if text.strip() and len(text.strip()) > 50:
                return text.strip()
        except Exception as e:
            print(f"Regular PDF text extraction failed: {e}")
        
        # If regular extraction failed or returned minimal text, try OCR
        print("Attempting OCR extraction...")
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_content, dpi=300)
            ocr_text = ""
            
            for i, image in enumerate(images):
                # Extract text from each page image using OCR
                page_text = pytesseract.image_to_string(image, lang='eng')
                if page_text.strip():
                    ocr_text += f"Page {i+1}:\n{page_text}\n\n"
            
            if ocr_text.strip():
                return ocr_text.strip()
            else:
                raise HTTPException(status_code=400, detail="Could not extract any text from PDF. The document may be empty or contain unsupported content.")
                
        except Exception as ocr_error:
            print(f"OCR extraction failed: {ocr_error}")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. This may be a scanned document or contain images that couldn't be processed.")
        
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")
    except HTTPException:
        raise
    except Exception as e:
        print(f"PDF processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def explain_document_with_gemini(text: str, mode: str = "simple") -> Dict:
    """Send document text to Gemini and get structured explanation"""
    try:
        if len(text) > 10000:
            text = text[:10000] + "..."

        system_prompt = SIMPLE_PROMPT if mode == "simple" else DETAILED_PROMPT
        prompt = f"{system_prompt}\n\nDocument text:\n{text}"

        response = call_gemini_with_fallback(prompt)
        
        response_text = response.text.strip()
        
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        
        if '{' in response_text and '}' in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
        else:
            json_str = response_text
        
        explanation = json.loads(json_str)
        
        required_fields = ["summary", "action_required", "deadline", "risk_if_ignored", "highlights"]
        for field in required_fields:
            if field not in explanation:
                explanation[field] = "Information not available in document"
        
        if isinstance(explanation.get("action_required"), list):
            explanation["action_required"] = "\n".join(f"{i+1}. {item}" for i, item in enumerate(explanation["action_required"]))
        
        if isinstance(explanation.get("summary"), list):
            explanation["summary"] = " ".join(explanation["summary"])
        
        if isinstance(explanation.get("deadline"), list):
            explanation["deadline"] = " ".join(explanation["deadline"])
        
        if isinstance(explanation.get("risk_if_ignored"), list):
            explanation["risk_if_ignored"] = " ".join(explanation["risk_if_ignored"])
        
        if not isinstance(explanation["highlights"], list):
            explanation["highlights"] = [str(explanation["highlights"])]
        
        return explanation
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="AI response format error")
    except Exception as e:
        print(f"AI processing error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

def answer_question_with_gemini(document_text: str, question: str) -> Dict:
    """Answer specific question about document"""
    try:
        prompt = f"{QA_SYSTEM_PROMPT}\n\nDOCUMENT TEXT:\n{document_text}\n\nUSER QUESTION:\n{question}"

        response = call_gemini_with_fallback(prompt)
        
        response_text = response.text.strip()
        
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        
        if '{' in response_text and '}' in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
        else:
            json_str = response_text
        
        answer = json.loads(json_str)
        
        if "answer" not in answer:
            answer["answer"] = "Could not find answer in document"
        if "source_section" not in answer:
            answer["source_section"] = "Not specified"
        
        return answer
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="AI response format error")
    except Exception as e:
        print(f"AI processing error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

# Authentication is now handled by Firebase on the frontend
# These endpoints are kept for backward compatibility but Firebase Auth is preferred

@app.post("/explain", response_model=DocumentExplanation)
async def explain_document(file: UploadFile = File(...), mode: str = Form("simple"), request: Request = None, authorization: str = Header(None)):
    """Process PDF document and return simple explanation"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 100MB")
    
    # Get client IP
    client_ip = request.client.host if request else "unknown"
    
    # Check if user is authenticated
    is_authenticated = False
    user = None
    if authorization and authorization.startswith("Bearer "):
        try:
            user = get_current_user(authorization)
            is_authenticated = True
        except:
            pass
    
    requires_auth = False
    requires_verification = False
    
    if is_authenticated:
        # Get Firebase user to check email verification
        try:
            firebase_user = auth.get_user(user["user_id"])
            email_verified = firebase_user.email_verified
        except:
            email_verified = False
            
        # Check user's document count
        user_doc = db.collection('users').document(user["user_id"]).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            doc_count = user_data.get('document_count', 0)
            
            # Check if user needs email verification
            if not email_verified:
                requires_verification = True
            # Check document limit for signed-in users (2 documents)
            elif doc_count >= 2:
                raise HTTPException(status_code=429, detail="Document limit reached. Signed-in users can process 2 documents. Please verify your email to continue.")
        else:
            # New user, create document
            db.collection('users').document(user["user_id"]).set({
                'email': user["email"],
                'created_at': datetime.now(),
                'google_drive_connected': False,
                'email_verified': email_verified,
                'document_count': 0
            })
            if not email_verified:
                requires_verification = True
    else:
        # Check IP usage limit for non-authenticated users
        if not check_ip_usage(client_ip):
            requires_auth = True
        else:
            increment_ip_usage(client_ip)
    
    try:
        # Read PDF content
        file.file.seek(0)
        pdf_content = file.file.read()
        file.file.seek(0)

        document_text = extract_text_from_pdf(file)
        explanation = explain_document_with_gemini(document_text, mode)

        # Store documents and increment count for authenticated users
        doc_id = None
        if is_authenticated and not requires_verification:
            # Increment document count
            user_ref = db.collection('users').document(user["user_id"])
            user_ref.update({'document_count': firestore.Increment(1)})
            
            # Check if user has Google Drive connected
            user_doc = db.collection('users').document(user["user_id"]).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                if user_data.get('google_drive_connected', False):
                    doc_id = store_document_text(user["user_id"], document_text, pdf_content, explanation)

        explanation["document_id"] = doc_id
        explanation["requires_auth"] = requires_auth
        explanation["requires_verification"] = requires_verification

        return DocumentExplanation(**explanation)
    finally:
        file.file.close()

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(question_request: QuestionRequest, user: dict = Depends(get_current_user)):
    """Ask specific question about uploaded document"""
    document_text = get_document_text(question_request.document_id, user["user_id"])
    answer = answer_question_with_gemini(document_text, question_request.question)
    return QuestionResponse(**answer)

@app.get("/auth/google")
async def google_auth(user: dict = Depends(get_current_user)):
    """Initiate Google OAuth flow"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI")],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
        redirect_uri=os.getenv("GOOGLE_REDIRECT_URI")
    )
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes=False,
        prompt="consent"
    )
    store_oauth_state(state, user["user_id"])
    return {"auth_url": authorization_url, "state": state}

@app.get("/auth/google/callback")
async def google_auth_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
):
    """Handle Google OAuth callback"""
    if not code or not state:
        return RedirectResponse(url=f"{os.getenv('FRONTEND_URL')}/?google_error=true&reason=missing_parameters")

    user_id = pop_oauth_state(state)
    if not user_id:
        return RedirectResponse(url=f"{os.getenv('FRONTEND_URL')}/?google_error=true&reason=invalid_state")


    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI")],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            },
            scopes=SCOPES,
            redirect_uri=os.getenv("GOOGLE_REDIRECT_URI")
        )
        flow.fetch_token(code=code)

        credentials = flow.credentials
        token_data = {
            'access_token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }

        # Store in Firestore
        user_ref = db.collection('users').document(user_id)
        user_ref.set({
            'google_drive_token': token_data,
            'google_drive_connected': True,
            'updated_at': datetime.now()
        }, merge=True)

        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:5173')
        
        # Return HTML page that shows success and closes tab after 2 seconds
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Google Drive Connected</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-align: center;
                }}
                .container {{
                    max-width: 400px;
                    padding: 40px;
                }}
                .success-icon {{
                    font-size: 4rem;
                    margin-bottom: 20px;
                }}
                h1 {{
                    font-size: 2rem;
                    margin-bottom: 10px;
                }}
                p {{
                    font-size: 1.1rem;
                    opacity: 0.9;
                    margin-bottom: 20px;
                }}
                .countdown {{
                    font-size: 1.2rem;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✅</div>
                <h1>Google Drive Connected!</h1>
                <p>Your Google account has been successfully linked to Docnify.</p>
                <p class="countdown">This tab will close in <span id="timer">2</span> seconds...</p>
            </div>
            <script>
                let countdown = 2;
                const timer = document.getElementById('timer');
                
                const interval = setInterval(() => {{
                    countdown--;
                    timer.textContent = countdown;
                    
                    if (countdown <= 0) {{
                        clearInterval(interval);
                        window.close();
                    }}
                }}, 1000);
                
                // Also try to notify parent window if it exists
                if (window.opener) {{
                    window.opener.postMessage({{
                        type: 'google-drive-connected',
                        success: true
                    }}, '{frontend_url}');
                }}
            </script>
        </body>
        </html>
        """
        
        return Response(content=html_content, media_type="text/html")

    except Exception as e:
        print(f"Google OAuth callback error: {e}")
        return RedirectResponse(url=f"{os.getenv('FRONTEND_URL')}/?google_error=true")

@app.get("/drive/files")
async def get_drive_files(user: dict = Depends(get_current_user)):
    """Get all files from Google Drive Docnify folder"""
    try:
        service = get_google_drive_service(user["user_id"])
        folder_id = get_or_create_docnify_folder(service)
        
        # Get all files in Docnify folder
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(
            q=query,
            fields='files(id, name, createdTime, size, mimeType)',
            orderBy='createdTime desc'
        ).execute()
        
        files = results.get('files', [])
        drive_files = []
        
        for file in files:
            drive_files.append({
                "id": file['id'],
                "name": file['name'],
                "created_at": file['createdTime'],
                "size": file.get('size', '0'),
                "mime_type": file.get('mimeType', 'unknown'),
                "source": "google_drive"
            })
        
        return {"files": drive_files}
    except Exception as e:
        print(f"Error getting drive files: {e}")
        return {"files": []}

@app.get("/temp/files")
async def get_temp_files():
    """Get temporary files for non-authenticated users"""
    temp_files = []
    for doc_id, doc in document_store.items():
        if doc["user_id"] == "temp":
            temp_files.append({
                "id": doc_id,
                "filename": f"document_{doc_id[:8]}.pdf",
                "expires_at": doc["expires_at"].isoformat()
            })
    return {"files": temp_files}

@app.get("/download/{doc_id}")
async def download_file(doc_id: str, authorization: str = Header(None)):
    """Download PDF file"""
    # Check if user is authenticated
    user_id = None
    if authorization and authorization.startswith("Bearer "):
        try:
            user = get_current_user(authorization)
            user_id = user["user_id"]
        except:
            pass

    # If not authenticated, try to access as temp user
    if not user_id:
        user_id = "temp"

    pdf_content = get_document_pdf(doc_id, user_id)
    return Response(content=pdf_content, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=document_{doc_id[:8]}.pdf"})

@app.delete("/drive/files/{file_id}")
async def delete_drive_file(file_id: str, user: dict = Depends(get_current_user)):
    """Delete file from Google Drive"""
    try:
        service = get_google_drive_service(user["user_id"])
        service.files().delete(fileId=file_id).execute()
        return {"message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.post("/save-to-drive/{doc_id}")
async def save_to_drive(doc_id: str, user: dict = Depends(get_current_user)):
    """Save explanation to Google Drive as PDF"""
    doc = document_store.get(doc_id)
    if not doc or doc["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Document not found")

    explanation = doc.get("explanation")
    if not explanation:
        raise HTTPException(status_code=400, detail="Explanation not available")

    # Generate PDF from explanation
    pdf_content = generate_explanation_pdf(explanation, doc_id[:8])

    filename = f"docnify_explanation_{doc_id[:8]}.pdf"

    drive_file_id = upload_pdf_to_drive(user["user_id"], pdf_content, filename)

    # Update the document store with drive file ID
    if doc_id in document_store:
        document_store[doc_id]["drive_file_id"] = drive_file_id

    return {"drive_file_id": drive_file_id, "message": "Explanation saved to Google Drive"}

@app.get("/drive/status")
async def drive_status(user: dict = Depends(get_current_user)):
    """Get Google Drive connection status and email"""
    try:
        # Get user document from Firestore
        user_doc = db.collection('users').document(user["user_id"]).get()
        if not user_doc.exists:
            return {"connected": False}

        user_data = user_doc.to_dict()
        if not user_data.get('google_drive_connected', False):
            return {"connected": False}

        token_data = user_data.get("google_drive_token")
        if not token_data:
            return {"connected": False}

        # Get user email from Google API
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            # If no refresh token, we can't create credentials properly
            # Return connected but without email
            return {"connected": True, "email": None}

        credentials = Credentials(
            token=token_data.get("access_token"),
            refresh_token=refresh_token,
            token_uri=token_data.get("token_uri"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes")
        )

        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        email = user_info.get('email')

        return {"connected": True, "email": email}
    except Exception as e:
        print(f"Error getting drive status: {e}")
        return {"connected": False}

@app.post("/drive/disconnect")
async def disconnect_drive(user: dict = Depends(get_current_user)):
    """Disconnect Google Drive"""
    try:
        user_ref = db.collection('users').document(user["user_id"])
        user_ref.update({
            'google_drive_connected': False,
            'google_drive_token': None,
            'updated_at': datetime.now()
        })
        return {"message": "Google Drive disconnected successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disconnect: {str(e)}")

@app.post("/send-verification")
async def send_verification_email(user: dict = Depends(get_current_user)):
    """Send email verification to user"""
    try:
        # Get Firebase user and send verification email
        firebase_user = auth.get_user(user["user_id"])
        
        # Generate verification link
        link = auth.generate_email_verification_link(firebase_user.email)
        
        # In a real app, you'd send this via email service
        # For now, we'll return the link (in production, use SendGrid, etc.)
        return {"verification_link": link, "message": "Verification email sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send verification: {str(e)}")

@app.post("/verify-email")
async def verify_email(token: str, user: dict = Depends(get_current_user)):
    """Verify email with token"""
    try:
        # In Firebase, email verification is handled client-side
        # This endpoint updates our Firestore record
        user_ref = db.collection('users').document(user["user_id"])
        user_ref.update({
            'email_verified': True,
            'verified_at': datetime.now()
        })
        return {"message": "Email verified successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify email: {str(e)}")

@app.get("/user/status")
async def get_user_status(user: dict = Depends(get_current_user)):
    """Get user status including document count and verification status"""
    try:
        # Get Firebase user to check email verification
        try:
            firebase_user = auth.get_user(user["user_id"])
            email_verified = firebase_user.email_verified
        except:
            email_verified = False
            
        user_doc = db.collection('users').document(user["user_id"]).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return {
                "email": user["email"],
                "email_verified": email_verified,
                "document_count": user_data.get('document_count', 0),
                "documents_remaining": max(0, 2 - user_data.get('document_count', 0)) if not email_verified else "unlimited"
            }
        else:
            return {
                "email": user["email"],
                "email_verified": email_verified,
                "document_count": 0,
                "documents_remaining": 2 if not email_verified else "unlimited"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user status: {str(e)}")

@app.post("/check-email")
async def check_email_exists(email: str = Form(...)):
    """Check if email exists in Firebase"""
    try:
        auth.get_user_by_email(email)
        return {"exists": True}
    except auth.UserNotFoundError:
        return {"exists": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking email: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
