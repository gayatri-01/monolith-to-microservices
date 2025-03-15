import requests
import random
import time
from datetime import datetime
from multiprocessing import Pool

BASE_URL = "http://localhost:5000"
LOG_FILE = "application.log"

# Simulated user sessions
USER_SESSIONS = {user_id: random.randint(1000, 9999) for user_id in range(1, 1001)}

def log_event(user_id, event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_id = USER_SESSIONS[user_id]
    with open(LOG_FILE, "a") as log:
        log.write(f"{timestamp} | Session: {session_id} | User: {user_id} | {event}\n")

# Different user flow scenarios with varied API hits
def user_session(_):
    user_id = random.randint(1, 1000)
    account_id = random.randint(10000, 99999)
    session_type = random.choice(["basic", "loan", "investment", "bill_payment", "fraud_check", "support_request", "account_update", "shopping", "travel_booking"])
    
    # Step 1: Create an account
    requests.post(f"{BASE_URL}/user-registration/new-user", params={"name": f"User{user_id}"})
    log_event(user_id, f"Account created with ID: {account_id}")
    
    # Step 2: Register as a customer
    requests.post(f"{BASE_URL}/customer-profile/enroll", params={"name": f"User{user_id}"})
    log_event(user_id, f"Customer registered with ID: {account_id}")
    
    if session_type == "basic":
        for _ in range(random.randint(2, 5)):
            to_acc = random.randint(10000, 99999)
            amount = round(random.uniform(10, 500), 2)
            requests.post(f"{BASE_URL}/money-transfer/send-payment", params={"from": account_id, "to": to_acc, "amount": amount})
            log_event(user_id, f"Transaction recorded: Transfer from {account_id} to {to_acc} of amount ${amount}")
        
        requests.get(f"{BASE_URL}/account-summary/balance", params={"accountId": account_id})
        log_event(user_id, f"Checked balance for account {account_id}")
        
        message = f"Transaction summary sent to user {user_id}"
        requests.post(f"{BASE_URL}/alerts/notify", params={"message": message})
        log_event(user_id, f"Notification sent: {message}")
    
    elif session_type == "loan":
        loan_amount = round(random.uniform(1000, 50000), 2)
        requests.post(f"{BASE_URL}/credit-evaluation/apply-loan", params={"accountId": account_id, "amount": loan_amount})
        log_event(user_id, f"Loan of ${loan_amount} applied for account: {account_id}")
        
        requests.get(f"{BASE_URL}/credit-score/view", params={"accountId": account_id})
        log_event(user_id, f"Checked credit score for account {account_id}")
    
    elif session_type == "shopping":
        item_id = random.randint(1000, 5000)
        requests.post(f"{BASE_URL}/ecommerce/order-place", params={"userId": user_id, "item": item_id})
        log_event(user_id, f"User {user_id} placed an order for item {item_id}")
        
        requests.get(f"{BASE_URL}/ecommerce/order-status", params={"userId": user_id})
        log_event(user_id, f"User {user_id} checked order status")
    
    elif session_type == "travel_booking":
        destination = random.choice(["Paris", "New York", "Tokyo", "London", "Sydney"])
        requests.post(f"{BASE_URL}/travel/book-flight", params={"userId": user_id, "destination": destination})
        log_event(user_id, f"User {user_id} booked a flight to {destination}")
        
        requests.get(f"{BASE_URL}/travel/itinerary", params={"userId": user_id})
        log_event(user_id, f"User {user_id} checked their itinerary")
    
    elif session_type == "fraud_check":
        suspicious_activity = random.choice(["multiple_large_transactions", "login_from_unusual_location", "multiple_failed_logins", "account_takeover_attempt"])
        requests.post(f"{BASE_URL}/security-monitoring/flag-activity", params={"accountId": account_id, "activity": suspicious_activity})
        log_event(user_id, f"Fraud check triggered for {account_id} due to {suspicious_activity}")

def run_simulation():
    with Pool(10) as pool:  # 10 concurrent users
        pool.map(user_session, range(5000))  # Simulate 5000 user sessions

if __name__ == "__main__":
    start_time = time.time()
    run_simulation()
    print(f"Log generation completed in {time.time() - start_time:.2f} seconds")
