from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from collections import defaultdict
import os
import hmac
import hashlib

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    ForeignKey, UniqueConstraint, func, Boolean, DateTime, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import razorpay

# ======================
# CONFIG
# ======================

SECRET_KEY = "supersecretkey_change_this"  # <-- change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# 🔑 Replace with your Google OAuth client ID
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "938605090290-spglalmtou8cnn22j6j7he1q82foeho8.apps.googleusercontent.com")

# 💳 Razorpay keys (replace with your keys)
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_RoDquDjwxWmSjV")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "NdTxn1AR30Q36ahtKvV3kZJ2")

# Each payment grants 1 credit costing ₹50
CREDIT_PRICE_PAISE = 5000  # 50 INR in paise

SQLALCHEMY_DATABASE_URL = "sqlite:///./timetable.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/google")
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

app = FastAPI(title="College Timetable Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # loosen for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# DB MODELS
# ======================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    google_sub = Column(String, unique=True, index=True, nullable=True)

    credits = Column(Integer, default=0)
    has_used_trial = Column(Boolean, default=False)

    # lock a Google account to a single device
    current_device_id = Column(String, nullable=True)

    reviews = relationship("Review", back_populates="user")



class Faculty(Base):
    __tablename__ = "faculties"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    reviews = relationship("Review", back_populates="faculty")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    faculty_id = Column(Integer, ForeignKey("faculties.id"))
    rating = Column(Float)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # New: course context
    course_code = Column(String, nullable=True)
    course_title = Column(String, nullable=True)

    user = relationship("User", back_populates="reviews")
    faculty = relationship("Faculty", back_populates="reviews")

    __table_args__ = (
        UniqueConstraint("user_id", "faculty_id", name="uix_user_faculty"),
    )



Base.metadata.create_all(bind=engine)

# ======================
# AUTH MODELS & HELPERS
# ======================

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: EmailStr
    credits: int
    has_used_trial: bool

    class Config:
        from_attributes = True  # Pydantic v2


class GoogleAuthIn(BaseModel):
    id_token: str
    device_id: Optional[str] = None



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_google_sub(db: Session, sub: str) -> Optional[User]:
    return db.query(User).filter(User.google_sub == sub).first()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate login. Please sign in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = get_user_by_email(db, token_data.email)
    if user is None:
        raise credentials_exception
    return user
def normalize_faculty_name(name: str) -> str:
    return name.strip().upper()
# ======================
# TIMETABLE CONSTANTS
# ======================

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# 1h slots -> 4 big periods
PERIODS = {
    ("08:00", "09:00"): "P1",
    ("09:00", "10:00"): "P1",
    ("10:00", "11:00"): "P2",
    ("11:00", "12:00"): "P2",
    ("13:00", "14:00"): "P3",
    ("14:00", "15:00"): "P3",
    ("15:00", "16:00"): "P4",
    ("16:00", "17:00"): "P4",
}

# ======================
# API MODELS (TIMETABLE)
# ======================

class Section(BaseModel):
    section_code: str
    course_name: str
    faculty_name: str
    time_slots: Dict[str, List[str]]
    faculty_rating: Optional[float] = None


class Timetable(BaseModel):
    sections: List[Section]


class Preferences(BaseModel):
    dislike_early: bool = False
    dislike_midmorning: bool = False
    dislike_afternoon: bool = False
    dislike_evening: bool = False

    prefer_weekend_off: bool = True

    preferred_faculty: List[str] = []
    avoid_faculty: List[str] = []

    faculty_weight: float = 0.6
    free_days_weight: float = 0.2
    timing_weight: float = 0.2


class GenerateRequest(BaseModel):
    raw_text: str
    chosen_courses: List[str]
    preferences: Preferences = Preferences()
    top_k: int = 5


class ReviewIn(BaseModel):
    faculty_name: str
    rating: float  # 1–5
    comment: Optional[str] = None
    course_code: Optional[str] = None
    course_title: Optional[str] = None



class CoursesRequest(BaseModel):
    raw_text: str


class SectionSummary(BaseModel):
    section_code: str
    course_name: str
    faculty_name: str


class CourseSummary(BaseModel):
    course_name: str
    sections: List[SectionSummary]


class GenerateResponseItem(BaseModel):
    score: float
    timetable: Timetable
    grid: Dict[str, Dict[str, List[Dict[str, str]]]]


class OrderCreateOut(BaseModel):
    order_id: str
    amount: int
    currency: str
    key_id: str


class PaymentVerifyIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

class ReviewOut(BaseModel):
    rating: float
    comment: Optional[str]
    created_at: datetime
    course_code: Optional[str] = None
    course_title: Optional[str] = None



class FacultySummaryOut(BaseModel):
    faculty_name: str
    avg_rating: float
    count: int
    summary: str
    breakdown: Dict[int, int]
    reviews: List[ReviewOut]

# ======================
# TIMETABLE PARSER
# ======================

def parse_sections(raw_text: str) -> List[Section]:
    import re
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    sections: List[Section] = []
    current_course: Optional[str] = None
    current_block: List[str] = []

    GENERIC_HEADINGS = [
    "course overview",
    "open elective",
    "professional elective",
    "professional core",
    "basic sciences",
    "engineering sciences",
    "humanities and sciences",
    "employability enhancement courses",
    "full registration",
    "phase-i",
    "phase ii"
]


    DEPARTMENT_KEYWORDS = [
        "aids", "aiml", "cse", "ece", "eee", "mech", "civil", "ai",
        "ai & ds", "ai & ml", "ped", "english", "maths"
    ]

    # ----------------------------------------------------------------------
    # COURSE TITLE DETECTOR
    # ----------------------------------------------------------------------
    def is_course_title(line: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9]{4,}\s*\[", line))

    # ----------------------------------------------------------------------
    # SECTION HEADER DETECTOR
    # EX: "4K1-3, AI - Kumaravelu R"
    # ----------------------------------------------------------------------
    def is_section_header(line: str) -> bool:
        return bool(re.match(r"^([0-9A-Z-]+),\s*(.*?)\s*-\s*(.+)$", line))

    # ----------------------------------------------------------------------
    # Extract TRUE course title below headings
    # ----------------------------------------------------------------------
    def extract_course_title(i):
        for j in range(i + 1, len(lines)):
            t = lines[j].strip()
            low = t.lower()

            # Skip useless headings
            if any(low.startswith(h) for h in GENERIC_HEADINGS):
                continue

            # Skip department lines
            if " - " in t:
                left, right = t.split(" - ", 1)
                if left.lower() in DEPARTMENT_KEYWORDS or right.lower() in DEPARTMENT_KEYWORDS:
                    continue

            # Skip date lines and times
            if t.startswith("Date:"):
                continue
            if re.search(r"\d{2}:\d{2}", t):
                continue

            # Skip section header
            if is_section_header(t):
                continue

            return t  # valid title

        return None

    # ----------------------------------------------------------------------
    # Validate FACULTY name (to prevent "AIDS & AIML" from becoming faculty)
    # ----------------------------------------------------------------------
    def is_real_faculty(name: str) -> bool:
        name_low = name.lower()

        # Remove departments
        if any(dept in name_low for dept in DEPARTMENT_KEYWORDS):
            return False

        # Faculty should contain at least one space (e.g., "Kumaravelu R")
        if len(name.split()) < 2:
            return False

        # Must contain letters
        if not re.search(r"[A-Za-z]", name):
            return False

        return True

    # ----------------------------------------------------------------------
    # Process each SECTION BLOCK
    # ----------------------------------------------------------------------
    def process_block(block, course):
        if not block or not course:
            return None

        header = block[0]
        m = re.match(r"^([0-9A-Z-]+),\s*(.*?)\s*-\s*(.+)$", header)
        if not m:
            return None

        section_code, subj, faculty = m.groups()
        faculty = faculty.strip()

        # ❌ Ignore wrong faculty like AIDS & AIML
        if not is_real_faculty(faculty):
            return None

        time_slots = {day: [] for day in DAYS}

        for line in block[1:]:
            if line.startswith("Date:"):
                continue

            for day in DAYS:
                if line.startswith(day + ":"):

                    # All time ranges on that line
                    ranges = re.findall(r"(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})", line)

                    for start, end in ranges:
                        # ❌ ignore classes < 30 minutes  
                        h1, m1 = map(int, start.split(":"))
                        h2, m2 = map(int, end.split(":"))
                        duration = (h2 * 60 + m2) - (h1 * 60 + m1)
                        if duration < 30:
                            continue

                        # convert to period
                        if (start, end) in PERIODS:
                            p = PERIODS[(start, end)]
                            if p not in time_slots[day]:
                                time_slots[day].append(p)

                    break

        # ❌ If no valid time slots → ignore section
        if all(len(v) == 0 for v in time_slots.values()):
            return None

        return Section(
            section_code=section_code,
            course_name=course,
            faculty_name=faculty,
            time_slots=time_slots,
            faculty_rating=None
        )

    # ----------------------------------------------------------------------
    # MAIN PARSE LOOP
    # ----------------------------------------------------------------------
    for i, line in enumerate(lines):

        # COURSE TITLE
        if is_course_title(line):
            if current_block:
                sec = process_block(current_block, current_course)
                if sec:
                    sections.append(sec)

            base = line
            extra = extract_course_title(i)
            if extra:
                current_course = f"{base} - {extra}"
            else:
                current_course = base

            current_block = []
            continue

        # SECTION HEADER
        if is_section_header(line):
            if current_block:
                sec = process_block(current_block, current_course)
                if sec:
                    sections.append(sec)

            current_block = [line]
            continue

        current_block.append(line)

    # Last section
    if current_block:
        sec = process_block(current_block, current_course)
        if sec:
            sections.append(sec)

    return sections





# ======================
# SCORING HELPERS
# ======================

def clashes_with_current(current_sections: List[Section], new_section: Section) -> bool:
    occupied = set()
    for s in current_sections:
        for day, periods in s.time_slots.items():
            for p in periods:
                occupied.add((day, p))
    for day, periods in new_section.time_slots.items():
        for p in periods:
            if (day, p) in occupied:
                return True
    return False


def occupied_slots(sections: List[Section]) -> Dict[tuple, Section]:
    occ: Dict[tuple, Section] = {}
    for s in sections:
        for day, periods in s.time_slots.items():
            for p in periods:
                key = (day, p)
                if key in occ:
                    raise ValueError("Clash detected")
                occ[key] = s
    return occ


def free_days_score(occ: Dict[tuple, Section]) -> float:
    score = 0.0
    for day in DAYS:
        if not any(k[0] == day for k in occ.keys()):
            score += 1.0
            if day == "Saturday":
                score += 1.0
    return score


def timing_penalty(occ: Dict[tuple, Section], prefs: Preferences) -> float:
    penalty = 0.0
    for (_day, period) in occ.keys():
        if prefs.dislike_early and period == "P1":
            penalty -= 1.0
        if prefs.dislike_midmorning and period == "P2":
            penalty -= 1.0
        if prefs.dislike_afternoon and period == "P3":
            penalty -= 1.0
        if prefs.dislike_evening and period == "P4":
            penalty -= 1.0
    return penalty


def get_faculty_rating_db(db: Session, faculty_name: str) -> float:
    faculty_name = normalize_faculty_name(faculty_name)
    faculty = db.query(Faculty).filter(Faculty.name == faculty_name).first()
    if not faculty:
        return 3.5
    avg = db.query(func.avg(Review.rating)).filter(Review.faculty_id == faculty.id).scalar()
    if avg is None:
        return 3.5
    return float(avg)


def faculty_preference_score(sections: List[Section], prefs: Preferences) -> float:
    score = 0.0
    for sec in sections:
        if sec.faculty_name in prefs.preferred_faculty:
            score += 2.0
        if sec.faculty_name in prefs.avoid_faculty:
            score -= 3.0
    return score


def score_timetable(sections: List[Section], prefs: Preferences) -> float:
    try:
        occ = occupied_slots(sections)
    except ValueError:
        return -1e9

    ratings = [s.faculty_rating or 3.5 for s in sections]
    faculty_score = sum(ratings) / len(ratings) if ratings else 0.0
    free_score = free_days_score(occ) if prefs.prefer_weekend_off else 0.0
    time_pen = timing_penalty(occ, prefs)
    faculty_pref = faculty_preference_score(sections, prefs)

    return (
        prefs.faculty_weight * faculty_score +
        prefs.free_days_weight * free_score +
        prefs.timing_weight * time_pen +
        faculty_pref
    )


def group_by_course(sections: List[Section]) -> Dict[str, List[Section]]:
    by_course: Dict[str, List[Section]] = defaultdict(list)
    for s in sections:
        by_course[s.course_name].append(s)
    return by_course


def build_grid(sections: List[Section]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    grid: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        day: {p: [] for p in ["P1", "P2", "P3", "P4"]} for day in DAYS
    }
    for sec in sections:
        for day, periods in sec.time_slots.items():
            for p in periods:
                grid[day][p].append({
                    "course": sec.course_name,
                    "faculty": sec.faculty_name,
                    "section": sec.section_code,
                })
    return grid


def build_best_timetables(
    sections: List[Section],
    chosen_courses: List[str],
    prefs: Preferences,
    top_k: int = 5,
) -> List[GenerateResponseItem]:
    by_course = group_by_course(sections)
    filtered_courses = [c for c in chosen_courses if c in by_course]

    best: List[tuple[float, List[Section]]] = []

    def backtrack(i: int, current_sections: List[Section]):
        nonlocal best
        if i == len(filtered_courses):
            sc = score_timetable(current_sections, prefs)
            best.append((sc, list(current_sections)))
            best.sort(key=lambda x: x[0], reverse=True)
            if len(best) > top_k:
                best[:] = best[:top_k]
            return

        course = filtered_courses[i]
        for sec in by_course[course]:
            if clashes_with_current(current_sections, sec):
                continue
            current_sections.append(sec)
            backtrack(i + 1, current_sections)
            current_sections.pop()

    backtrack(0, [])

    results: List[GenerateResponseItem] = []
    for score, secs in best:
        results.append(GenerateResponseItem(
            score=score,
            timetable=Timetable(sections=secs),
            grid=build_grid(secs),
        ))
    return results

# ======================
# FACULTY REVIEW SUMMARY (Amazon-style)
# ======================

def build_faculty_summary(faculty_name: str, reviews: List[Review]) -> FacultySummaryOut:
    if not reviews:
        return FacultySummaryOut(
            faculty_name=faculty_name,
            avg_rating=0.0,
            count=0,
            summary="No student reviews yet. Be the first to share your experience.",
            breakdown={i: 0 for i in range(1, 6)},
            reviews=[],
        )

    ratings = [int(round(r.rating)) for r in reviews]
    avg = sum(ratings) / len(ratings)

    breakdown = {i: 0 for i in range(1, 6)}
    for r in ratings:
        if 1 <= r <= 5:
            breakdown[r] += 1

    # Amazon-like summary text
    if avg >= 4.5:
        tone = "Students consistently rate this faculty as excellent with highly positive feedback."
    elif avg >= 4.0:
        tone = "Students generally have a very good experience with this faculty."
    elif avg >= 3.0:
        tone = "Feedback is mixed: some students are satisfied, while others see room for improvement."
    elif avg > 0:
        tone = "Students often find this faculty challenging, with several critical comments."
    else:
        tone = "No clear trend from reviews yet."

    out_reviews = [
        ReviewOut(
            rating=r.rating,
            comment=r.comment,
            created_at=r.created_at,
            course_code=r.course_code,
            course_title=r.course_title,
        )
        for r in sorted(reviews, key=lambda x: x.created_at, reverse=True)
    ]

    return FacultySummaryOut(
        faculty_name=faculty_name,
        avg_rating=avg,
        count=len(reviews),
        summary=tone,
        breakdown=breakdown,
        reviews=out_reviews,
    )

# ======================
# AUTH ENDPOINTS (GOOGLE LOGIN ONLY)
# ======================

@app.post("/auth/google", response_model=Token)
def google_login(body: GoogleAuthIn, db: Session = Depends(get_db)):
    token = body.id_token
    device_id = body.device_id or ""

    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            grequests.Request(),
            GOOGLE_CLIENT_ID
        )
        email = idinfo["email"]
        sub = idinfo["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token. One device should have one account")

    # ✅ Enforce: ONE DEVICE → ONE GOOGLE ACCOUNT
    if device_id:
        existing_user = (
            db.query(User)
            .filter(User.current_device_id == device_id)
            .first()
        )
        if existing_user and existing_user.google_sub != sub:
            raise HTTPException(
                status_code=403,
                detail="This device is already linked to another Google account."
            )

    # Find or create user
    user = get_user_by_google_sub(db, sub)
    if not user:
        user = get_user_by_email(db, email)
        if user and not user.google_sub:
            user.google_sub = sub
        elif not user:
            user = User(
                email=email,
                google_sub=sub,
                credits=0,
                has_used_trial=False,
            )
            db.add(user)

        db.commit()
        db.refresh(user)

    # ✅ Bind device to user (only once)
    if device_id and not user.current_device_id:
        user.current_device_id = device_id
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}




@app.get("/auth/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# ======================
# PAYMENT / CREDIT ENDPOINTS (RAZORPAY)
# ======================

@app.post("/payments/create-order", response_model=OrderCreateOut)
def create_order(current_user: User = Depends(get_current_user)):
    order = razorpay_client.order.create(dict(
        amount=CREDIT_PRICE_PAISE,
        currency="INR",
        payment_capture=1
    ))
    return OrderCreateOut(
        order_id=order["id"],
        amount=CREDIT_PRICE_PAISE,
        currency="INR",
        key_id=RAZORPAY_KEY_ID
    )


@app.post("/payments/verify")
def verify_payment(
    data: PaymentVerifyIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    body = data.razorpay_order_id + "|" + data.razorpay_payment_id
    expected_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        body.encode(),
        hashlib.sha256
    ).hexdigest()

    if expected_signature != data.razorpay_signature:
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    current_user.credits += 1
    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return {"message": "Payment verified, 1 credit added", "credits": current_user.credits}

# ======================
# TIMETABLE API ENDPOINTS
# ======================

@app.post("/courses", response_model=List[CourseSummary])
def get_courses(req: CoursesRequest):
    sections = parse_sections(req.raw_text)
    by_course = group_by_course(sections)
    result: List[CourseSummary] = []
    for course_name, secs in by_course.items():
        result.append(CourseSummary(
            course_name=course_name,
            sections=[
                SectionSummary(
                    section_code=s.section_code,
                    course_name=s.course_name,
                    faculty_name=s.faculty_name,
                )
                for s in secs
            ],
        ))
    return result


def charge_credit_if_needed(db: Session, user: User):
    """
    First generate is free (trial).
    After that, every /generate consumes 1 credit.
    """
    if not user.has_used_trial:
        user.has_used_trial = True
    else:
        if user.credits <= 0:
            raise HTTPException(
                status_code=402,
                detail="You have no credits left. Please purchase a credit to generate a new timetable."
            )
        user.credits -= 1
    db.add(user)
    db.commit()
    db.refresh(user)


@app.post("/generate", response_model=List[GenerateResponseItem])
def generate(
    req: GenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    charge_credit_if_needed(db, current_user)

    sections = parse_sections(req.raw_text)
    for s in sections:
        s.faculty_rating = get_faculty_rating_db(db, s.faculty_name)
    return build_best_timetables(sections, req.chosen_courses, req.preferences, req.top_k)

# ======================
# FACULTY REVIEWS API
# ======================

@app.post("/review")
def submit_review(
    review: ReviewIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if review.rating < 1 or review.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    faculty_name = normalize_faculty_name(review.faculty_name)
    faculty = db.query(Faculty).filter(Faculty.name == review.faculty_name).first()
    if not faculty:
        faculty = Faculty(name=review.faculty_name)
        db.add(faculty)
        db.commit()
        db.refresh(faculty)

    existing = (
        db.query(Review)
        .filter(Review.user_id == current_user.id, Review.faculty_id == faculty.id)
        .first()
    )

    if existing:
        existing.rating = review.rating
        existing.comment = review.comment
        existing.course_code = review.course_code
        existing.course_title = review.course_title
        existing.created_at = datetime.utcnow()
    else:
        db.add(Review(
            user_id=current_user.id,
            faculty_id=faculty.id,
            rating=review.rating,
            comment=review.comment,
            course_code=review.course_code,
            course_title=review.course_title,
        ))

    db.commit()

    avg = get_faculty_rating_db(db, faculty.name)
    return {
        "message": "Review recorded (anonymous)",
        "faculty_name": faculty.name,
        "avg_rating": avg,
    }


@app.get("/faculty/{faculty_name}/reviews", response_model=FacultySummaryOut)
def get_faculty_reviews(faculty_name: str, db: Session = Depends(get_db)):
    faculty_name = normalize_faculty_name(faculty_name)
    faculty = db.query(Faculty).filter(Faculty.name == faculty_name).first()
    if not faculty:
        return FacultySummaryOut(
            faculty_name=faculty_name,
            avg_rating=0.0,
            count=0,
            summary="No student reviews yet for this faculty.",
            breakdown={i: 0 for i in range(1, 6)},
            reviews=[],
        )

    reviews = db.query(Review).filter(Review.faculty_id == faculty.id).all()
    return build_faculty_summary(faculty_name, reviews)

# ======================
# FRONTEND ROUTE
# ======================

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FileResponse("index.html")



