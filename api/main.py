
from fastapi import FastAPI, Request, HTTPException, Form,WebSocket, WebSocketDisconnect, Query
from fastapi import Depends, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import asyncio
from datetime import datetime, timedelta
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic import EmailStr
from typing import Optional
import hmac
import hashlib
import time
import requests
import pandas as pd
from threading import Thread, Lock, Event


from datetime import datetime
from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator
)
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator, IchimokuIndicator
)
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator

import numpy as np
import logging


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# import spacy
import os
import io
import re
from dotenv import load_dotenv
import boto3
from openai import OpenAI
from pydantic import BaseModel
# from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr

from api.database import get_db_connection
from passlib.hash import bcrypt
from passlib.context import CryptContext


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if config.env exists (for local testing)
if os.path.exists("config.env"):
    load_dotenv("config.env")
    print("Loaded environment variables from config.env (local testing).")
else:
    print("Using Vercel environment variables.")
#print(os.environ)

CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
SECRET = os.getenv("PAYPAL_SECRET_KEY")
RAZORPAY_KEY_ID = os.getenv('razorpay_key_id')
print("nasty",os.getenv('razorpay_key_id'))
RAZORPAY_SECRET_KEY = os.getenv('razorpay_secret_key')
print(RAZORPAY_SECRET_KEY)    

# Load environment variables
# load_dotenv("config.env")
OPENAI_API_KEY = os.getenv("open_ai_key")
print(OPENAI_API_KEY)

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Check your config.env file.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
# Initialize FastAPI app
app = FastAPI()

# JWT Configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
import uuid

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")
# FastAPI Router
router = APIRouter()
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com"  # Use live URL in production

@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    """Serve the homepage."""
    return templates.TemplateResponse("index.html", {"request": request})
# # Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
allow_origins=["http://127.0.0.1:8000", "https://crypto-ai-pi.vercel.app","https://cryptoai.drofn.com","https://api.razorpay.com"]
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

import razorpay
from fastapi import HTTPException

client = razorpay.Client(auth=("rzp_live_oeyqQxUaW48JyP", "razorpay_secret_key"))

def verify_payment(payment_id: str, order_id: str):
    try:
        payment = client.payment.fetch(payment_id)
        if payment["order_id"] != order_id:
            raise HTTPException(status_code=422, detail="Order ID mismatch")
        if payment["status"] != "captured":
            raise HTTPException(status_code=422, detail="Payment not captured")
        return payment
    except razorpay.errors.RazorpayError as e:
        raise HTTPException(status_code=422, detail=f"Razorpay verification failed: {str(e)}")



async def send_reset_email(email: str, reset_link: str):
    try:
        # Email server configuration
        print(email)
        print(reset_link)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email =  os.getenv("MAIL_FROM")
        print(sender_email)
        sender_password = os.getenv("MAIL_PASSWORD")
        print(sender_password)

        # Create email message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Password Reset Request"
        message["From"] = sender_email
        message["To"] = email

        # Email body
        text = f"""\
        Hi,
        
        Click the link below to reset your password:
        {reset_link}
        
        If you did not request this, please ignore this email.
        """
        message.attach(MIMEText(text, "plain"))

        # Connect to the email server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, message.as_string())
        server.quit()

        print(f"Password reset email sent successfully to {email}.")
    except Exception as e:
        print(f"Failed to send password reset email: {str(e)}")
        raise


fake_users_db = {
    "test@example.com": {
        "username": "test",
        "hashed_password": "password123",  # Example hashed password
        "disabled": False,
    }
}

# Helper Functions
# def verify_password(plain_password, hashed_password):
#     return plain_password == hashed_password  # Replace with hashing logic

def get_user(username: str):
    return fake_users_db.get(username)

FREE_TRIAL_DAYS = 3
MONTHLY_PLAN_COST = 10  # USD
ANNUAL_PLAN_COST = 100  # USD

def get_paypal_access_token():
    response = requests.post(
        f"{PAYPAL_API_BASE}/v1/oauth2/token",
        auth=(CLIENT_ID, SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials"},
    )
    response_data = response.json()
    print(response_data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get PayPal access token")
    return response_data["access_token"]
@app.post("/webhooks/paypal")
async def paypal_webhook(request: Request):
    payload = await request.json()
    event_type = payload.get("event_type")

    # Add webhook signature validation here
    headers = request.headers
    if not validate_paypal_webhook(payload, headers):
        raise HTTPException(status_code=400, detail="Invalid PayPal webhook signature")

    if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
        subscription_id = payload["resource"]["id"]
        user_id = payload["resource"]["custom_id"]  # If you passed it in metadata
        # Update user subscription status in the database
        print(f"Subscription activated for user {user_id}: {subscription_id}")

    return {"status": "success"}

def validate_paypal_webhook(payload, headers):
    """
    Validate the PayPal webhook signature.
    """
    access_token = get_paypal_access_token()
    response = requests.post(
        f"{PAYPAL_API_BASE}/v1/notifications/verify-webhook-signature",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "auth_algo": headers.get("paypal-auth-algo"),
            "cert_url": headers.get("paypal-cert-url"),
            "transmission_id": headers.get("paypal-transmission-id"),
            "transmission_sig": headers.get("paypal-transmission-sig"),
            "transmission_time": headers.get("paypal-transmission-time"),
            "webhook_id": "YOUR_WEBHOOK_ID",  # Replace with your webhook ID
            "webhook_event": payload,
        },
    )
    return response.json().get("verification_status") == "SUCCESS"


    return {"status": "success"}
async def cancel_subscription_paypal(subscription_id: str):
    access_token = get_paypal_access_token()  # Dynamically fetch token
    url = f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}/cancel"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    data = {"reason": "User requested cancellation"}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 204:
        return {"message": "Subscription canceled successfully."}
    else:
        return {"error": response.json()}

# Helper function to fetch subscription status
# async def fetch_subscription_status(user_id: int, db) -> dict:
#     cursor = db.cursor()
#     cursor.execute("""
#         SELECT subscription_start_date, subscription_end_date
#         FROM users
#         WHERE id = %s
#     """, (user_id,))
#     result = cursor.fetchone()

#     if not result or not result[0]:
#         return {"is_trial_active": False, "is_subscribed": False}

#     start_date, end_date = result

#     if not end_date or datetime.utcnow() > end_date:
#         return {"is_trial_active": False, "is_subscribed": False}

#     if datetime.utcnow() < start_date + timedelta(days=FREE_TRIAL_DAYS):
#         days_left = (start_date + timedelta(days=FREE_TRIAL_DAYS) - datetime.utcnow()).days
#         print(f"Trial active for {days_left} days")
#         return {"is_trial_active": True, "days_left": days_left}

#     return {"is_trial_active": False, "is_subscribed": True}
@app.get("/api/check-access")
async def check_access(user_id: int, db=Depends(get_db_connection)):
    status = await fetch_subscription_status(user_id, db)
    if status["redirect"] == "subscription":
        return {
            "redirect_to": "/subscription",
            "message": status["message"],
            "subscription_type": None,
            "subscription_status": "inactive",
        }
    return {
        "redirect_to": "/html3",
        "message": f"Access granted. Subscription type: {status['subscription_type']}",
        "subscription_type": status["subscription_type"],
        "subscription_status": "active",
    }
def update_subscription(user_id,db):
    try: 
        subscription_status="one_time"
        

        cursor = db.cursor()
        cursor.execute("""
            UPDATE users
            SET subscription_status = %s,
                    last_payment_date=NOW()
            WHERE id = %s
        """, (subscription_status, user_id))
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        db.close()    

async def fetch_subscription_status(user_id: int, db) -> dict:
    cursor = db.cursor()
    cursor.execute("""
        SELECT subscription_start_date, subscription_end_date, one_time_payment_start_date,subscription_status
        FROM users
        WHERE id = %s
    """, (user_id,))
    result = cursor.fetchone()

    if not result:
        return {"redirect": "subscription", "message": "No active access found"}

    subscription_start_date, subscription_end_date, one_time_payment_start_date,subscription_status = result

    # Initialize status flags
    is_subscribed = False
    has_one_time_access = False
    is_trial_active = False
    one_time_payment_end_date=None
    # Check if subscription is active
    if subscription_status=="inactive":
        if subscription_end_date and datetime.utcnow() <= subscription_end_date:
            is_subscribed = True

        # Check if free trial is active
        if subscription_start_date and datetime.utcnow() < subscription_start_date + timedelta(days=FREE_TRIAL_DAYS):
            is_trial_active = True

        # Check if one-time payment is active
        if one_time_payment_start_date and datetime.utcnow() <= one_time_payment_start_date + timedelta(days=30):
            has_one_time_access = True
            one_time_payment_end_date=one_time_payment_start_date + timedelta(days=30)

    # Update the subscription status in the database
    # subscription_status = "inactive"
    if is_subscribed:
        subscription_status = "active"
    elif is_trial_active:
        subscription_status = "active"
    elif has_one_time_access:
        subscription_status = "active"

    # cursor.execute("""
    #     UPDATE users
    #     SET subscription_status = %s
        
    #     WHERE id = %s
    # """, (subscription_status, user_id))
    # db.commit()
    print(subscription_status)
    # Redirect based on the access status
    if subscription_status == "inactive":
        return {"redirect": "subscription", "message": "No active access. Redirecting to subscription page.", "subscription_type":"inactive"}
    else:
        return {"redirect": "html3", "message": "Access granted. Redirecting to the main page.", "subscription_type":"active"}
@router.post("/api/subscribe")
async def subscribe_user(user_id: int, plan_type: str, paypal_subscription_id: str, db=Depends(get_db_connection)):
    try:
        cursor = db.cursor()
        # Update user's subscription in the database
        cursor.execute("""
            UPDATE users 
            SET subscription_start_date = NOW(),
                subscription_plan = %s,
                paypal_subscription_id = %s
            WHERE id = %s
        """, (plan_type, paypal_subscription_id, user_id))
        db.commit()
        return {"message": "Subscription successful"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Subscription error: {str(e)}")
    finally:
        cursor.close()
        db.close()

@router.post("/api/cancel-subscription")
async def cancel_subscription(user_id: int, db=Depends(get_db_connection)):
    try:
        cursor = db.cursor()
        # Fetch PayPal subscription ID
        cursor.execute("SELECT paypal_subscription_id FROM users WHERE id = %s", (user_id,))
        subscription_id = cursor.fetchone()[0]

        if not subscription_id:
            raise HTTPException(status_code=404, detail="No active subscription found")

        # Cancel PayPal subscription
        cancel_subscription_paypal(subscription_id)

        # Update user's subscription status
        cursor.execute("""
            UPDATE users 
            SET subscription_plan = NULL,
                subscription_end_date = NOW()
                       subscription_status=inactive
            WHERE id = %s
        """, (user_id,))
        db.commit()

        return {"message": "Subscription canceled successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cancellation error: {str(e)}")
    finally:
        cursor.close()
        db.close()


@router.get("/api/subscription-status")
async def subscription_status(user_id: int, db=Depends(get_db_connection)):
    """
    Check the subscription status of a user.
    Args:
        user_id (int): User ID.
    Returns:
        dict: Subscription status.
    """
    try:
        status = await fetch_subscription_status(user_id, db)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching subscription status: {str(e)}")
@app.get("/subscription", response_class=HTMLResponse)
async def subscription_page(request: Request):
    """
    Serve the subscription page.
    """
    return templates.TemplateResponse("subscription.html", {"request": request})

@router.get("/api/protected")
async def protected_endpoint(user_id: int, db=Depends(get_db_connection)):
    """
    Endpoint protected by subscription or free trial.
    Args:
        user_id (int): User ID.
    """
    try:
        status = await fetch_subscription_status(user_id, db)

        # Check access permission
        if not status.get("is_subscribed") and not status.get("is_trial_active"):
            raise HTTPException(status_code=403, detail="Subscription required.")

        return {"message": "Access granted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking subscription status: {str(e)}")
    
@app.post("/api/paypal-payment")
async def paypal_payment(data: dict):
    subscription_id = data.get("subscription_id")
    user_id = data.get("user_id")

    if not subscription_id or not user_id:
        raise HTTPException(status_code=400, detail="Missing subscription_id or user_id.")

    # Perform backend actions like saving subscription details
    try:
        # Example database logic
        # db.execute("INSERT INTO subscriptions (user_id, subscription_id) VALUES (%s, %s)", (user_id, subscription_id))
        print(f"Subscription {subscription_id} processed for user {user_id}.")
        return {"success": True, "message": "Subscription processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing subscription: {str(e)}")    
    
# @app.post("/api/razorpay-payment")
# async def process_razorpay_payment(user_id: int,request: Request):
#     data = await request.json()
#     payment_id = data.get("payment_id")

#     if not payment_id:
#         raise HTTPException(status_code=400, detail="Payment ID is required.")

#     # Verify payment with Razorpay (optional step, Razorpay webhook is more robust)
#     # Call Razorpay API here if necessary to verify the payment

#     # Grant dashboard access
#     # user_id = "user_id"  # Replace with the actual user ID logic
#     users_db[user_id]["access_granted"] = True

#     return JSONResponse({"success": True, "message": "Payment successful, access granted."})

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """Hash a plain text password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=15)) -> str:
    """Generate a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")



@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve the login page."""
    with open("templates/login.html", "r") as file:
        return HTMLResponse(content=file.read())
    

# Define the request schema
class CreateOrderRequest(BaseModel):
    amount: int  # Amount in INR
    user_id: str  # User ID

@app.post("/api/create-order")

async def create_order(data: dict):
    # Validate input
    amount = data.get("amount",1)
    amount = 1
    print(data) 
    if not amount:
        raise HTTPException(status_code=400, detail="Amount is required")

    url = "https://api.razorpay.com/v1/orders"
    auth = (RAZORPAY_KEY_ID, RAZORPAY_SECRET_KEY)
    # payload = {
    #     "amount": amount * 100,  # Convert to paise
    #     "currency": "INR",
    #     "receipt": "txn_12345",  # Optional but recommended
    #     "payment_capture": 1,    # Auto-capture payment
    # }
    payload = {
        "amount": amount * 100,  # Convert to paise
        "currency": "INR",
        "receipt": f"receipt_{int(time.time())}",  # Unique receipt ID
        "payment_capture": 1,    # Auto-capture payment
        "notes": {
            "user_id": data.get("user_id", "Unknown User")
        }
    }


    response = requests.post(url, auth=auth, json=payload)
    print("response,",response.ok)
    if response.status_code != 200:
        print(f"Error from Razorpay: {response.json()}")
        raise HTTPException(status_code=response.status_code, detail="Failed to create Razorpay order")

    return response.json()


# class PaymentVerificationRequest(BaseModel):
#     razorpay_payment_id: str
#     razorpay_order_id: str
#     razorpay_signature: str


# @app.post("/api/razorpay-payment")
# async def verify_payment(data: PaymentVerificationRequest):
#     """
#     Verifies Razorpay payment signature and redirects to html3.
#     """
#     try:
#         # Generate the HMAC signature for validation
#         payload = f"{data.razorpay_order_id}|{data.razorpay_payment_id}"
#         expected_signature = hmac.new(
#             RAZORPAY_SECRET_KEY.encode(),
#             payload.encode(),
#             hashlib.sha256
#         ).hexdigest()

#         # Validate the signature
#         if expected_signature != data.razorpay_signature:
#             raise HTTPException(status_code=400, detail="Invalid payment signature")

#         # Payment verification successful
#         return {
#             "success": True,
#             "message": "Payment verified successfully.",
#             "redirect_url": "/html3.html"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))    
    
class PaymentVerificationRequest(BaseModel):
    razorpay_payment_id: str
    razorpay_order_id: str
    razorpay_signature: str
    user_id: int  # Include user ID to identify the user in the database

@app.post("/api/razorpay-payment")
async def verify_payment(data: PaymentVerificationRequest, db=Depends(get_db_connection)):
    """
    Verifies Razorpay payment signature, updates subscription status, 
    and last payment date in the database, then redirects to html3.
    """
    try:
        # Generate the HMAC signature for validation
        payload = f"{data.razorpay_order_id}|{data.razorpay_payment_id}"
        expected_signature = hmac.new(
            RAZORPAY_SECRET_KEY.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # Validate the signature
        if expected_signature != data.razorpay_signature:
            raise HTTPException(status_code=400, detail="Invalid payment signature")

        # Update subscription status and last payment date in the database
        cursor = db.cursor()

        # Update the user's subscription status and payment details
        # subscription_end_date = datetime.utcnow() + timedelta(days=30)  # Assuming 1-month subscription
        one_time_payment_start_date=datetime.utcnow()
        one_time_payment_end_date=one_time_payment_start_date + timedelta(days=30)
        # last_payment_date = datetime.utcnow()
        subscription_status = "active"

        cursor.execute(
            """
            UPDATE users
            SET subscription_status = %s, 
                one_time_payment_start_date=%s,
                one_time_payment_end_date = %s 
                
            WHERE id = %s
            """,
            (subscription_status,one_time_payment_start_date , one_time_payment_end_date, data.user_id)
        )

        db.commit()

        # Payment verification and database update successful
        return {
            "success": True,
            "message": "Payment verified successfully and subscription updated.",
            "redirect_url": "/index3.html"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        db.close()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.get("/dashboard2", response_class=HTMLResponse)
async def dashboard2(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

@app.get("/dashboard3", response_class=HTMLResponse)
async def dashboard3(request: Request):
    return templates.TemplateResponse("index3.html", {"request": request})
@app.get("/subs", response_class=HTMLResponse)
async def subs(request: Request):
    return templates.TemplateResponse("subscription.html", {"request": request})
@app.post("/api/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...),
):
    """
    Log in the user and validate credentials.
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Fetch the user from the database
        cursor.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            # return {"error": "Invalid email or user does not exist"}
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid email or user does not exist"}
            )

        user_id, hashed_password = user

        # Verify the password
        if not pwd_context.verify(password, hashed_password):
            # return {"error": "Invalid password"}
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid password"}
            )


        # Generate an access token
        access_token = create_access_token(data={"sub": str(user_id)})

        # Include success message
      
        
        return {
            "message": "Login successful!",
            "redirect_url": "/dashboard2",
            "access_token": access_token,
            "user_id": user_id  # Include user ID in response
        }

    except Exception as e:
        print(f"Error during login: {str(e)}")
        # return {"error": "An unexpected error occurred. Please try again later."}
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred. Please try again later."}
        )

    finally:
        cursor.close()
        connection.close()


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response




@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and generate JWT token."""
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        # raise HTTPException(status_code=400, detail="Invalid username or password")
        return {"error": "Invalid username or password"}
    
    token = create_access_token(data={"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/app")
async def main_app(current_user: str = Depends(get_current_user)):
    """Main application logic, protected by login."""
    return {"message": f"Welcome to the app, {current_user}!"}

users_db={}

@app.get("/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    """
    Serve the registration page.
    """
    return templates.TemplateResponse("register.html", {"request": request})


# Handle registration form submission
# @app.post("/api/register")
# async def register_user(
#     username: str = Form(...),
#     email: EmailStr = Form(...),
#     password: str = Form(...)
# ):
#     """
#     Handle the user registration process.
#     """
#     # Check if the user already exists
#     if email in users_db:
#         raise HTTPException(status_code=400, detail="User already exists")

#     # Save the user to the "database" (in-memory for this example)
#     users_db[email] = {"username": username, "password": password}

#     return {"message": "Account created successfully!", "user": {"username": username, "email": email}}
sessions = {}
@app.post("/api/register")
async def register_user(
    username: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...)
):
    """
    Register a new user in the PostgreSQL database.
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Check if the email already exists
        cursor.execute("SELECT 1 FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return {"error": "User already exists"}
  
        # Hash the password
        password_hash = pwd_context.hash(password)
        status="inactive"
        # Insert the user into the database
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash,subscription_status)
            VALUES (%s, %s, %s,%s)
            RETURNING id
            """,
            (username, email, password_hash,status)
        )
        connection.commit()

        # Fetch the inserted user ID
        user_id = cursor.fetchone()[0]
        print(f"New user created with ID: {user_id}")

        return {"message": "Account created successfully!", "user_id": user_id}
    except Exception as e:
        connection.rollback()
        print(f"Database error: {e}")
        return {"error": "An unexpected error occurred. Please try again later."}
    finally:
        cursor.close()
        connection.close()

users_db = {
    "user@example.com": {"username": "user", "password": "hashed_password"},
    # Add other users as needed
}

import uuid

import uuid
from fastapi import HTTPException, Depends
from pydantic import BaseModel, EmailStr

# async def send_reset_email(email: str, reset_link: str):
#     # Your email sending logic here
#     print(f"Email sent to {email} with link: {reset_link}")

# Define the request model
class ForgetPasswordRequest(BaseModel):
    email: EmailStr
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")  
@app.post("/api/forget-password")
async def forget_password(
    request: ForgetPasswordRequest,  # Use Pydantic model
    db=Depends(get_db_connection)
):
    try:
        cursor = db.cursor()

        # Access the email from the request model
        email = request.email

        # Check if email exists in the database
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            return {"error": "Email not found"}

        # Generate a unique reset token
        reset_token = str(uuid.uuid4())
        # reset_link = f"http://127.0.0.1:8000/reset-password?token={reset_token}"
        reset_link = f"{BASE_URL}/reset-password?token={reset_token}"
        print("Reset Link:", reset_link)

        # Save the reset token in the database
        cursor.execute(
            "UPDATE users SET reset_token = %s WHERE email = %s",
            (reset_token, email),
        )
        db.commit()
        print(email)
        print("Reset token saved in the database.")
        # Send the email with the reset link
      
        await send_reset_email(email, reset_link)

        return {"message": "Password reset link has been sent to your email."}
    except Exception as e:
        db.rollback()
        return {"error": f"Database error: {str(e)}"}
    finally:
        cursor.close()
        db.close()

# @app.get("/forget-password", response_class=HTMLResponse)
# async def get_forget_password_page():
#     with open("forget-password.html", "r") as file:
#         return HTMLResponse(content=file.read())
@app.get("/forget-password", response_class=HTMLResponse)
async def get_forget_password_page(request: Request):
    return templates.TemplateResponse("forget-password.html", {"request": request})    

@app.get("/reset-password")
async def reset_password_page(request: Request, token: str):
    # Validate the token and render the reset password page
    return templates.TemplateResponse("reset-password.html", {"request": request, "token": token})    
@app.post("/api/reset-password")
async def reset_password(new_password: str = Form(...), token: str = Form(...), db=Depends(get_db_connection)):
    try:
        cursor = db.cursor()

        # Check if the token exists and fetch the user
        cursor.execute("SELECT email FROM users WHERE reset_token = %s", (token,))
        user = cursor.fetchone()

        if not user:
            return {"error": "Invalid or expired token"}

        email = user[0]

        # Update the user's password and clear the token
        hashed_password = pwd_context.hash(new_password)
        cursor.execute(
            "UPDATE users SET password_hash = %s, reset_token = NULL WHERE email = %s",
            (hashed_password, email),
        )
        db.commit()

        return {"message": "Password reset successfully!"}
    except Exception as e:
        db.rollback()
        return {"error": f"Database error: {str(e)}"}
    finally:
        cursor.close()
        db.close()
@app.get("/terms", response_class=HTMLResponse)
async def terms_of_service(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})


import requests
import pandas as pd
from datetime import datetime
from binance.client import Client
# Routes
@app.get("/", response_class=HTMLResponse)
async def disclaimer():
    """Serve the homepage with a disclaimer modal."""
    with open("templates/index.html", "r") as file:
        return HTMLResponse(content=file.read())
polly_client = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("my_AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("my_AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)
# import os
# sts_client = boto3.client("sts")
# role_arn = "arn:aws:iam::888577066858:role/lambda_er"
# response = sts_client.assume_role(
#     RoleArn=role_arn,
#     RoleSessionName="lambdaERSession"
# )


# credentials = response["Credentials"]
# print(credentials)
# AWS_REGION = os.getenv("AWS_REGION", "us-east-1") 
# # Initialize AWS Polly client
# polly_client = boto3.client(
#     "polly",
#     aws_access_key_id=credentials["AccessKeyId"],
#     aws_secret_access_key=credentials["SecretAccessKey"],
#     aws_session_token=credentials["SessionToken"],region_name=AWS_REGION
# )
API_KEY =  os.getenv("binance_api_key")
API_SECRET = os.getenv("binance_secret_key")
# gemini_api_key = os.getenv("gemini_api2")

client = Client(API_KEY, API_SECRET)
# genaiclient = genai.Client(
#             api_key=gemini_api_key
#         )
import requests
import pandas as pd
from datetime import datetime, timedelta


def get_historical_data_extended(symbol, interval, start_date, end_date):
    """
    Fetch extended historical cryptocurrency data from Binance API.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
        interval (str): Data interval (e.g., '1h', '1d').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Historical price data.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000  # Maximum data points per request
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for bad HTTP response
        
        data = response.json()
        if not data:
            break  # Exit if no data is returned
        
        all_data.extend(data)
        current_start = int(data[-1][6]) + 1  # Use the last 'close_time' + 1ms for the next request
    
    # Convert data to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # Process and clean data
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    return df[["open_time", "open", "high", "low", "close", "volume","close_time"]]    


def add_technical_indicators(df):
    """
    Add a comprehensive set of technical indicators to the DataFrame.
    """
    # Make a copy of the DataFrame
    df = df.copy()

    # Momentum Indicators
    df.loc[:, "rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    df.loc[:, "stoch_k"] = StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).stoch()
    df.loc[:, "williams_r"] = WilliamsRIndicator(
        high=df["high"], low=df["low"], close=df["close"], lbp=14
    ).williams_r()
    df.loc[:, "roc"] = ROCIndicator(close=df["close"], window=12).roc()
    df.loc[:, "tsi"] = TSIIndicator(close=df["close"], window_slow=25, window_fast=13).tsi()

    # Trend Indicators
    df.loc[:, "sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df.loc[:, "ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df.loc[:, "sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()
    df.loc[:, "ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df.loc[:, "sma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()
    df.loc[:, "ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
    df.loc[:, "adx"] = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()
    df.loc[:, "cci"] = CCIIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).cci()
    ichimoku = IchimokuIndicator(
        high=df["high"], low=df["low"], window1=9, window2=26, window3=52
    )
    df.loc[:, "ichimoku_a"] = ichimoku.ichimoku_a()
    df.loc[:, "ichimoku_b"] = ichimoku.ichimoku_b()

    # Volatility Indicators
    bollinger = BollingerBands(close=df["close"], window=20, window_dev=2)
    df.loc[:, "bollinger_hband"] = bollinger.bollinger_hband()
    df.loc[:, "bollinger_lband"] = bollinger.bollinger_lband()
    df.loc[:, "atr"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    keltner = KeltnerChannel(
        high=df["high"], low=df["low"], close=df["close"], window=20
    )
    df.loc[:, "keltner_hband"] = keltner.keltner_channel_hband()
    df.loc[:, "keltner_lband"] = keltner.keltner_channel_lband()

    # Volume Indicators
    df.loc[:, "obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    df.loc[:, "mfi"] = calculate_mfi(df)
    df.loc[:, "cmf"] = ChaikinMoneyFlowIndicator(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20
    ).chaikin_money_flow()
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df.loc[:, "macd"] = macd.macd()
    df.loc[:, "macd_signal"] = macd.macd_signal()

    return df

def calculate_mfi(df, window=14):
    """
    Calculate Money Flow Index (MFI) manually.
    
    Args:
        df (pd.DataFrame): DataFrame with columns: 'high', 'low', 'close', 'volume'.
        window (int): Lookback window for MFI calculation.
    
    Returns:
        pd.Series: MFI values.
    """
    # Step 1: Calculate Typical Price (TP)
    df.loc[:,'typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Step 2: Calculate Raw Money Flow (RMF)
    df.loc[:,'money_flow'] = df['typical_price'] * df['volume']
    
    # Step 3: Positive and Negative Money Flow (Vectorized)
    df.loc[:,'positive_flow'] = (df['typical_price'] > df['typical_price'].shift(1)) * df['money_flow']
    df.loc[:,'negative_flow'] = (df['typical_price'] < df['typical_price'].shift(1)) * df['money_flow']
    
    # Step 4: Money Flow Ratio (MFR)
    positive_flow_sum = df['positive_flow'].rolling(window=window).sum()
    negative_flow_sum = df['negative_flow'].rolling(window=window).sum()
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    
    # Step 5: Money Flow Index (MFI)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi

# Global control for fetch thread
fetch_running = Event()
fetch_thread = None
all_data = pd.DataFrame()

def fetch_data_continuously(symbol, delay_between_fetches):
    """Continuously fetch data while the flag is set."""
    global all_data
    output_file = "temp.csv"

    while fetch_running.is_set():  # Check if the fetch_running flag is set
        try:
            # Fetch new data
            order_book = client.get_order_book(symbol=symbol)
            order_book_timestamp = pd.Timestamp.now(tz='UTC').floor("s").tz_localize(None)
            bid_price = float(order_book['bids'][0][0])
            bid_qty = float(order_book['bids'][0][1])
            ask_price = float(order_book['asks'][0][0])
            ask_qty = float(order_book['asks'][0][1])

            # Create DataFrame for new data
            orderbook_df = pd.DataFrame([{
                'timestamp': order_book_timestamp,
                'bid_price': bid_price,
                'bid_qty': bid_qty,
                'ask_price': ask_price,
                'ask_qty': ask_qty
            }])

            # Append new data to all_data
            if all_data.empty:
                all_data = orderbook_df
            else:
                all_data = pd.concat([all_data, orderbook_df], ignore_index=True)

            # Save to CSV
            if not os.path.isfile(output_file):
                all_data.to_csv(output_file, index=False)
            else:
                orderbook_df.to_csv(output_file, mode="a", header=False, index=False)

            time.sleep(delay_between_fetches)  # Wait before fetching again
        except Exception as e:
            print(f"Error in fetch_data_continuously: {e}")
            break

symbol=""
@app.websocket("/ws/orderbook")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await websocket.accept()
    try:
        while True:
            if not symbol:
                raise ValueError("Symbol is required for the order book.")

            # Fetch live order book data
            order_book = client.get_order_book(symbol=symbol)
            order_book_timestamp = pd.Timestamp.now(tz='UTC').floor("s").tz_localize(None)
            bid_price = float(order_book['bids'][0][0])
            bid_qty = float(order_book['bids'][0][1])
            ask_price = float(order_book['asks'][0][0])
            ask_qty = float(order_book['asks'][0][1])

            # Prepare the data
            data = {
                "timestamp": order_book_timestamp.isoformat(),
                "bid_price": bid_price,
                "bid_qty": bid_qty,
                "ask_price": ask_price,
                "ask_qty": ask_qty,
            }

            # Send data to the client
            await websocket.send_json(data)
            await asyncio.sleep(1)  # Adjust update frequency
    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
     

class ChatRequest(BaseModel):
    user_id: str
    message: str
    crypto: str  # Add the crypto field

@app.post("/begin_conversation")
async def begin_conversation(request: ChatRequest):
    """Start the conversation and fetch process."""
    global symbol  # Declare 'symbol' as global to modify it inside the function
    logger.info(f"Received request: {request}")
   
    global fetch_thread

    # Parse the cryptocurrency from the request
    selected_crypto = request.crypto
    symbol+=selected_crypto
    print(f"Received request to begin conversation with crypto: {selected_crypto}")

    # Start fetching data for the selected cryptocurrency
    if not fetch_running.is_set():
        fetch_running.set()  # Set the flag
        fetch_thread = Thread(target=fetch_data_continuously, args=(selected_crypto, 1), daemon=True)
        fetch_thread.start()
        print(f"Started fetch thread for {selected_crypto}.")
    else:
        print("Fetch thread is already running.")

    return {"message": f"Conversation started and data fetching initiated for {selected_crypto}."}

@app.post("/end_conversation")
async def end_conversation(user_id: str = "default"):
    """End the conversation and stop fetch process."""
    global fetch_thread

    # Stop fetching data
    if fetch_running.is_set():
        fetch_running.clear()  # Clear the flag
        fetch_thread.join()  # Wait for the thread to finish
        print("Fetch thread stopped.")

    # Reset session
    if user_id in sessions:
        sessions.pop(user_id)
    return {"message": "Conversation ended and data fetching stopped."}




def add_volatility_features(df):
    """
    Add volatility-related features to the dataset.
    """
    # Ensure necessary columns are present
    required_columns = ["high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the dataset.")

    # Add features using .loc to avoid SettingWithCopyWarning
    df.loc[:, "rolling_std_10"] = df["close"].rolling(window=10).std()
    df.loc[:, "rolling_std_20"] = df["close"].rolling(window=20).std()

    # Bollinger Band Width
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df.loc[:, "bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()

    # Average True Range (ATR)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df.loc[:, "atr"] = atr.average_true_range()

    # Price Rate of Change (ROC)
    df.loc[:, "roc"] = df["close"].pct_change(periods=10)

    # EMA Difference
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df.loc[:, "ema_diff"] = short_ema - long_ema

    # Rolling Mean and Std
    df.loc[:, "rolling_mean_5"] = df["close"].rolling(window=5).mean()
    df.loc[:, "rolling_std_5"] = df["close"].rolling(window=5).std()

    # Add time index and future target
    df.loc[:, "time_index"] = range(len(df))
    df.loc[:, "close_next"] = df["close"].shift(-1)

    return df





# Helper Functions
def clean_text(text: str) -> str:
    """Remove emojis and special characters."""
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text


@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    """Serve the homepage."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/speak")
async def speak(request: Request):
    """Generate speech using AWS Polly."""
    data = await request.json()
    text = data.get("text", "").strip()
    print(text)
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        sanitized_text = clean_text(text)
        response = polly_client.synthesize_speech(
            Text=sanitized_text,
            OutputFormat="mp3",
            Engine="generative",
            VoiceId="Ruth",  # Replace with the desired voice
            TextType="text",
        )

        audio_stream = io.BytesIO(response["AudioStream"].read())
        audio_stream.seek(0)

        # Return audio stream
        return StreamingResponse(audio_stream, media_type="audio/mpeg")
    except Exception as e:
        print(f"Error with AWS Polly: {str(e)}")
        raise HTTPException(status_code=500, detail="AWS Polly error")
from datetime import datetime, timedelta

# Define the start and end date as datetime objects
# end_date = "2025-12-09"  # Current UTC time
# end_date_dt = datetime.utcnow()
# print(type(end_date))
# #end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
# #end_date_dt = end_date.strftime("%Y-%m-%d")
# print((end_date_dt))

# start_date = end_date_dt - timedelta(days=1)  # 1 year ago

# # Convert to strings in the required format
# start_date_str = start_date.strftime("%Y-%m-%d")
# print((start_date_str))
# end_date_str = end_date
API_KEY =os.getenv("google_api")
SEARCH_ENGINE_ID = os.getenv("google_engine_id")

def historic_data(selected_crypto):
    # Define the start and end date as datetime objects
    end_date = "2025-12-09"  # Current UTC time
    end_date_dt = datetime.utcnow()
    print(type(end_date))
    #end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    #end_date_dt = end_date.strftime("%Y-%m-%d")
    print((end_date_dt))

    start_date = end_date_dt - timedelta(days=1)  # 1 year ago

    # Convert to strings in the required format
    start_date_str = start_date.strftime("%Y-%m-%d")
    print((start_date_str))
    end_date_str = end_date
    historical_data = get_historical_data_extended(selected_crypto, "30m", start_date_str, end_date)
  
    start_date3 = end_date_dt - timedelta(hours=1)
    start_date_str3 = start_date3.strftime("%Y-%m-%d")
    print((start_date_str3))
    data_1m= get_historical_data_extended(selected_crypto, "1m", start_date_str3, end_date)
    data_1m=data_1m[-200:]
    print(data_1m.tail(2))
    

    return historical_data,  data_1m



# Load NLP model
# nlp = spacy.load("en_core_web_sm")
# Use the converted strings in the function


#print("Historical data fetched and saved successfully.")
conversation = ""

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat conversation."""
    user_id = request.user_id
    user_input = request.message.strip()
    selected_crypto = request.crypto  # Get the selected cryptocurrency


    if not user_input and not sessions.get(user_id):
        introduction = "Hello! I am a Crypto expert named Crypto AI. I can help you with analysis and insights on cryptocurrency markets and blockchain technology. Feel free to ask me anything related to cryptocurrencies or blockchain technology."
        sessions[user_id] = {"conversation_history": [{"role": "assistant", "content": introduction}]}
        return {"reply": introduction}
    
    # historical_data = get_historical_data_extended(selected_crypto, "30m", start_date_str, end_date)
    historical_data,  data_1m = historic_data(selected_crypto)
    print(historical_data.tail(2))
    ask_bid_data = pd.read_csv("temp.csv")
    print(ask_bid_data.shape)
    ask_bid_data = (ask_bid_data[-1000:]).to_string()
    print(ask_bid_data[:10])
    print(historical_data.shape)
    historical_data_with_indicators = add_technical_indicators(historical_data)
    historical_data_with_indicators = add_volatility_features(historical_data_with_indicators)
    print(historical_data.shape)
   
    # Save to CSV
    # historical_data_with_indicators.to_csv("btc_usdt_extended_data.csv", index=False)
    historical_data= historical_data_with_indicators.to_string()

    print(historical_data[:10])

    data_1m_with = add_technical_indicators(data_1m)
    data_1m_with = add_volatility_features(data_1m_with)
    print(data_1m_with.shape)
   
    # Save to CSV
    # historical_data_with_indicators.to_csv("btc_usdt_extended_data.csv", index=False)
    data_1m= data_1m_with.to_string()

    print(data_1m[:10])

    
    session = sessions.setdefault(user_id, {"conversation_history": []})
    if user_input:
        session["conversation_history"].append({"role": "user", "content": user_input})
    # Your Google Custom Search API details
    

    # User input
    

    # Process input with NLP
    # doc = nlp(user_input)
    # query = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    query  = user_input

    print(f"Extracted query: '{query}'")

   

    # Perform Google Search
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json()
        snippets = [item["snippet"] for item in results.get("items", [])[:3]]
        search_summary = "\n".join(snippets)
    else:
        search_summary = "No results found."

    print(f"Search summary: {search_summary}")    
    try:
     
        system_message = f"""You are a highly knowledgeable and analytical AI expert  named Crypto AI specializing in cryptocurrency markets and blockchain technology. Your role is to provide accurate, insightful, and clear explanations, analysis, and recommendations. You ll also be given latest last 1 days bitcoin price data and which you can use for analysis. You must keep the following guidelines in mind while interacting:

1. **Clarity and Accuracy**:
   - Always explain concepts in a precise and easy-to-understand manner, tailoring complexity to the user's level of expertise (beginner, intermediate, advanced).
   - Use simple analogies when explaining technical concepts to beginners.

2. **Market Expertise**:
   - Stay updated on major cryptocurrencies (e.g., Bitcoin, Ethereum) and altcoins, trends in the market, and trading strategies.
   - Provide actionable insights based on price trends, trading volumes, and on-chain data.
   - Clearly explain market metrics like market capitalization, liquidity, volatility, and risk.

3. **Blockchain Insights**:
   - Dive deep into blockchain architecture, consensus mechanisms (e.g., Proof of Work, Proof of Stake), smart contracts, and decentralized finance (DeFi).
   - Offer real-world applications and use cases of blockchain beyond cryptocurrencies, such as supply chain, healthcare, and gaming.

4. **Risk Awareness and Responsibility**:
   - Always provide balanced views, highlighting risks and uncertainties, especially in the volatile crypto market.
   - Avoid making financial advice. Instead, provide analysis and encourage users to conduct their research.

5. **Global Market Trends**:
   - Stay informed about regulations, policies, and major events shaping the global crypto landscape.
   - Explain the impact of geopolitical events, institutional adoption, and regulatory changes on the crypto ecosystem.

6. **Technical Analysis**:
   - Offer insights into technical chart patterns, indicators (e.g., RSI, MACD), and tools for price forecasting.
   - Explain complex strategies like arbitrage, staking, and yield farming in an accessible manner.

7. **Scams and Security**:
   - Educate users about common scams, security best practices, and the importance of private key management.
Here is the latest last 1-day price data (30-minute interval):
{historical_data}

here is the 1m interval data
{data_1m}

Here is the order book data:
{ask_bid_data}

Here is the Google summary of your query:
{search_summary}

Answer should be less than 200 words"""
        




        messages = [{"role": "system", "content": system_message}] + session["conversation_history"]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.8,
            max_tokens=300,
        )

        reply = response.choices[0].message.content.strip()
        session["conversation_history"].append({"role": "assistant", "content": reply})
        
#         response = genaiclient.models.generate_content(
#             model='gemini-2.0-flash-exp', contents=(
#         system_message
#         + "\n".join(
#             [f"{entry['role']}: {entry['content']}" for entry in session["conversation_history"]]
#         )
#         + f"\nuser: {user_input}"  # Add the latest user input
#     )
# )
#         reply=response.text
#         session["conversation_history"].append({"role": "assistant", "content": reply})
        
        return {"reply": reply}
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong.")


@app.post("/chat_recommendation")
async def chat_recommendation(request: ChatRequest):
    """Handle chat conversation."""
    user_id = request.user_id
    user_input = request.message.strip()
    selected_crypto = request.crypto  # Get the selected cryptocurrency
     

  
    
    


    historical_data,  data_1m = historic_data(selected_crypto)
    print(historical_data.tail(2))
    ask_bid_data = pd.read_csv("temp.csv")
    print(ask_bid_data.shape)
    ask_bid_data = (ask_bid_data[-1000:]).to_string()
    print(ask_bid_data[:10])
    print(historical_data.shape)
    historical_data_with_indicators = add_technical_indicators(historical_data)
    historical_data_with_indicators = add_volatility_features(historical_data_with_indicators)
   
   
    # Save to CSV
    # historical_data_with_indicators.to_csv("btc_usdt_extended_data.csv", index=False)
    historical_data= historical_data_with_indicators.to_string()

    print(historical_data[:10])

    data_1m_with = add_technical_indicators(data_1m)
    data_1m_with = add_volatility_features(data_1m_with)
    print(data_1m_with.shape)
    print("heeba")
    print(data_1m_with.tail(2))
   
    # Save to CSV
    # historical_data_with_indicators.to_csv("btc_usdt_extended_data.csv", index=False)
    data_1m= data_1m_with.to_string()

    
    time_stamp2=data_1m_with["open_time"].iloc[-1]
    time_stamp2=time_stamp2.floor("s")
    print(time_stamp2)
    time_stamp = time_stamp2.strftime("%B %d, %Y %H:%M:%S")
    ist_time = time_stamp2.tz_localize('UTC').tz_convert('Asia/Kolkata')
    print(ist_time)
    ist_time = ist_time.strftime("%B %d, %Y %H:%M:%S")
    print(ist_time)
   
    session = sessions.setdefault(user_id, {"conversation_history": []})
    # if user_input:
    #     session["conversation_history"].append({"role": "user", "content": user_input})
    # Your Google Custom Search API details
    

    # User input
    

    # Process input with NLP
    # doc = nlp(user_input)
    # query = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    query  = f"latest sentiment data on {selected_crypto} from the social networks along with the latest news "

    print(f"Extracted query: '{query}'")

   

    # Perform Google Search
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json()
        snippets = [item["snippet"] for item in results.get("items", [])[:3]]
        search_summary = "\n".join(snippets)
    else:
        search_summary = "No results found."

    print(f"Search summary: {search_summary}")    
    try:
        system_message = f"""
You are a highly skilled AI Day Trading and Futures Analyst specializing in synthesizing various trading strategies to provide actionable insights for day traders and futures traders. Your expertise includes predicting short-term price movements, identifying entry and exit points, and assessing futures trading opportunities based on market data, technical indicators, sentiment analysis, and historical trends.

Your role is to integrate day trading strategies such as momentum trading, range trading, scalping, and breakout trading while also analyzing futures contracts to identify the best opportunities at a specific timestamp.

AI Day Trading and Futures Instructions:
Input Data Analysis:

Analyze the provided data, including:
Historical price data.
Order book depth and bid/ask volumes.
Technical indicators such as RSI, MACD, Bollinger Bands, and VWAP.
Sentiment trends based on news and social media.
Futures market data, including open interest, implied volatility, and contract expiration dates.
Day Trading Strategies Integration:

Combine key day trading strategies:
Momentum Trading: Identify trends using indicators like RSI, MACD, and ROC.
Range Trading: Highlight support and resistance levels for range-bound assets.
Scalping: Detect quick opportunities for small profits.
Breakout Trading: Identify assets breaking past support or resistance levels.
Reversal and Pullback Trading: Assess opportunities for reversals or temporary retracements.
Futures Trading Analysis:

Analyze futures data for the selected asset:
Identify the most liquid and actively traded contracts.
Use open interest and volume to gauge market sentiment and interest.
Evaluate implied volatility to predict price swings.
Assess contango/backwardation for price expectations.
Select the best futures contract at the specific timestamp for actionable predictions.
Prediction and Recommendations:

Combine insights from day trading strategies and futures analysis.
Provide predictions for:
Spot price movements based on historical and real-time data.
Futures contract movements for near-term opportunities.
Highlight the best trading strategy (day trading or futures) for the specific timestamp and explain why.
Output Format:

Present the analysis and predictions clearly and concisely.
Ensure predictions reference the timestamp in human-readable language and the source of information.
Avoid discussing internal analysis steps; focus solely on actionable insights.
Here is the latest last 1-day price data (5-minute interval):
{historical_data}

here is the 1m interval data for last 200min
{data_1m}

Here is the order book data:
{ask_bid_data}

Here is the Google summary of your query:
{search_summary}
        
Data Summary for { selected_crypto}:

Timestamp of last open price of data used for analysis: {time_stamp}(IST: {ist_time})


Example Output:use only format not the content
Data Summary for { selected_crypto}:

Timestamp of last open price of data used for analysis: {time_stamp}(IST: {ist_time})

Source: Aggregated from [data sources, e.g., QuantifiedStrategies.com, Google News, futures data provider].
Analysis Period: Last 24 hours.
Day Trading Analysis:

Synthesis: "The analysis of Bitcoin over the past 24 hours shows a bullish trend, with strong support at $51,200 and resistance at $53,500. Volatility is high, with Bollinger Bands widening, and sentiment analysis indicates optimism fueled by positive institutional activity."
Prediction: "Bitcoin is likely to test the resistance at $53,500 within the next 4 hours. A breakout may lead to a price surge toward $55,000."
Day Trading Recommendations:
Entry Point: Buy near $51,200.
Exit Point: Sell near $53,500.
Stop-Loss: Place at $50,800.
Futures Trading Analysis:

Selected Contract: Bitcoin Futures (BTC-USD-DEC24).
Synthesis: "The futures market shows high open interest and implied volatility for the December contract. The futures price is trading at a slight premium to the spot price, indicating bullish sentiment."
Prediction: "The December contract is expected to rise to $54,800, aligning with bullish sentiment in the spot market. A breakout in spot price above $53,500 may further boost the futures price."


Futures Trading Recommendations:
Entry Point: Long position at $53,000.
Exit Point: Close position at $54,800.
Stop-Loss: Place at $52,500.
Best Trading Opportunity (Timestamp-Based Decision):

"At this timestamp, the futures contract offers a better risk-reward ratio due to high open interest and clear bullish sentiment. Traders should consider entering a long position in Bitcoin Futures (BTC-USD-DEC24) at $53,000 for a potential upside to $54,800."
Risks and Uncertainties:

"Potential risks include unexpected regulatory news or a sharp decline in liquidity during midday trading hours. Monitor market sentiment closely for any changes."
Next Steps:

"Track price movements at $51,200 (spot support) and $53,500 (spot resistance) for early signals."
"Watch the futures market for sudden shifts in open interest or implied volatility."

"""





        messages = [{"role": "system", "content": system_message}] # + session["conversation_history"]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.8,
            max_tokens=300,
        )

        reply = response.choices[0].message.content.strip()


       
        # response = genaiclient.models.generate_content(
        #     model='gemini-2.0-flash-exp', contents=system_message 
        # )
        # reply=response.text
        # # session["conversation_history"].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong.")




# @app.post("/reset")
# async def reset(user_id: str = "default"):
#     """Reset conversation history for the given user ID."""
#     if user_id in sessions:
#         sessions.pop(user_id)  # Clear session data
#         return {"message": "Conversation history reset."}
#     return {"message": "No active session found to reset."}

@app.post("/reset")
async def reset(user_id: str = "default"):
    """Reset conversation history and stop fetching."""
    global fetch_running

    # Clear session data
    if user_id in sessions:
        sessions.pop(user_id)

    # Stop the fetch thread
    if fetch_running.is_set():
        fetch_running.clear()  # Clear the flag to stop fetching
        print("Fetch thread stopped.")

    return {"message": "Conversation and data fetching stopped successfully."}
