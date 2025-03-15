import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from datetime import datetime
from networkx.algorithms import community
import logging
from flask import Flask, request, jsonify

LOG_FILE = "application.log"  # Path to the log file

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)

@app.route('/user-registration/new-user', methods=['POST'])
def register_user():
    user_id = request.json.get('user_id')
    logging.info(f"Customer registered with ID: {user_id} - User: {user_id}")
    return jsonify({"message": f"User {user_id} registered"})

@app.route('/customer-profile/enroll', methods=['POST'])
def enroll_customer():
    user_id = request.json.get('user_id')
    logging.info(f"Customer {user_id} enrolled - User: {user_id}")
    return jsonify({"message": f"Customer {user_id} enrolled"})

@app.route('/money-transfer/send-payment', methods=['POST'])
def send_payment():
    from_acc = request.json.get('from')
    to_acc = request.json.get('to')
    amount = request.json.get('amount')
    user_id = request.json.get('user_id')
    logging.info(f"Transaction recorded: Transfer from {from_acc} to {to_acc} of amount ${amount} - User: {user_id}")
    return jsonify({"message": f"Transferred {amount} from {from_acc} to {to_acc}"})

@app.route('/account-summary/balance', methods=['GET'])
def account_balance():
    account_id = request.args.get('accountId')
    balance = round(random.uniform(100, 10000), 2)
    logging.info(f"Checked balance for account: {account_id}")
    return jsonify({"message": f"Balance for account {account_id} is ${balance}"})

@app.route('/alerts/notify', methods=['POST'])
def notify():
    message = request.json.get('message')
    user_id = request.json.get('user_id')
    logging.info(f"Notification sent: {message} - User: {user_id}")
    return jsonify({"message": f"Notification sent: {message}"})

@app.route('/credit-evaluation/apply-loan', methods=['POST'])
def apply_loan():
    account_id = request.json.get('accountId')
    amount = request.json.get('amount')
    user_id = request.json.get('user_id')
    logging.info(f"Loan of ${amount} applied for account {account_id} - User: {user_id}")
    return jsonify({"message": f"Loan of ${amount} applied for account {account_id}"})

@app.route('/credit-score/view', methods=['GET'])
def credit_score():
    account_id = request.args.get('accountId')
    score = random.randint(300, 850)
    logging.info(f"Credit score viewed for account: {account_id}")
    return jsonify({"message": f"Credit score for account {account_id} is {score}"})

@app.route('/ecommerce/order-place', methods=['POST'])
def place_order():
    user_id = request.json.get('userId')
    item = request.json.get('item')
    logging.info(f"User {user_id} placed an order for item {item}")
    return jsonify({"message": f"User {user_id} placed an order for item {item}"})

@app.route('/ecommerce/order-status', methods=['GET'])
def order_status():
    user_id = request.args.get('userId')
    logging.info(f"Order status checked for User {user_id}")
    return jsonify({"message": f"Order status for User {user_id}"})

@app.route('/travel/book-flight', methods=['POST'])
def book_flight():
    user_id = request.json.get('userId')
    destination = request.json.get('destination')
    logging.info(f"User {user_id} booked a flight to {destination}")
    return jsonify({"message": f"User {user_id} booked a flight to {destination}"})

@app.route('/travel/itinerary', methods=['GET'])
def travel_itinerary():
    user_id = request.args.get('userId')
    logging.info(f"User {user_id} itinerary checked")
    return jsonify({"message": f"User {user_id} itinerary checked"})

@app.route('/security-monitoring/flag-activity', methods=['POST'])
def flag_fraud_activity():
    account_id = request.json.get('accountId')
    activity = request.json.get('activity')
    logging.info(f"Fraud check triggered for {account_id} due to {activity}")
    return jsonify({"message": f"Fraud check triggered for {account_id} due to {activity}"})

if __name__ == "__main__":
    app.run(port=5000)
