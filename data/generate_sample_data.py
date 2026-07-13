"""
generate_sample_data.py
=========================
Generates a small SYNTHETIC demo dataset at data/raw/emails.csv so the
pipeline can be run end-to-end immediately after cloning the repo,
without requiring a manual dataset download first.

IMPORTANT: This is NOT the dataset to use for real portfolio results.
It exists purely so `python -m src.train` works out of the box for
testing/demo purposes. For a genuine portfolio project, replace
data/raw/emails.csv with a real public dataset, for example:

  - Kaggle: "Phishing Email Detection" datasets (search Kaggle directly)
  - Nazario Phishing Corpus (https://monkey.org/~jose/phishing/)
  - Enron Email Dataset (legitimate-email baseline)
  - SpamAssassin public corpus (ham/spam)

Any dataset works as long as the final CSV has a text column and a
label column — update config.yaml's `columns` section to match.
"""

import csv
import os
import random

random.seed(42)

PHISHING_TEMPLATES = [
    "Dear customer, your account has been suspended. Click here to verify your identity immediately: {url}",
    "URGENT: Unusual login attempt detected. Confirm your password now at {url} or your account will be locked.",
    "Congratulations! You have won a {prize}. Claim your prize now by clicking {url} before it expires.",
    "Your payment could not be processed. Update your billing information at {url} to avoid service interruption.",
    "We noticed suspicious activity on your account. Verify your identity now: {url}",
    "Final notice: your subscription will be cancelled unless you confirm your details at {url} today.",
    "Security alert: someone tried to access your account from a new device. Secure your account here: {url}",
    "Your package could not be delivered. Please confirm your address and pay a small fee at {url}",
    "IRS Notice: you have a pending tax refund. Submit your bank details at {url} to receive payment.",
    "Your mailbox is almost full. Click {url} to upgrade your storage immediately or lose access.",
    "Action required: your invoice is overdue. Pay now at {url} to avoid late fees and account suspension.",
    "We detected a virus on your device. Download our security tool now from {url} to protect your data.",
    "Your bank account has been temporarily locked due to suspicious activity. Verify at {url} now.",
    "You have a new voicemail. Login at {url} to listen before it is deleted in 24 hours.",
    "Your Apple ID has been disabled. Restore access immediately by verifying at {url}",
]

LEGITIMATE_TEMPLATES = [
    "Hi team, attached is the agenda for tomorrow's 10am meeting. Let me know if you have questions.",
    "Thanks for the update on the project timeline. I'll review the document and get back to you by Friday.",
    "Reminder: the quarterly report is due next Monday. Please send your section by end of day Thursday.",
    "Hey, are we still on for lunch on Wednesday? Let me know what time works for you.",
    "Please find attached the invoice for last month's services. Let me know if you have any questions.",
    "Great meeting you at the conference last week. Looking forward to collaborating on the project.",
    "The server maintenance is scheduled for this weekend between 2am and 4am. No downtime is expected.",
    "Here are the meeting notes from today's standup. Action items are listed at the bottom.",
    "Happy birthday! Hope you have a wonderful day. Let's grab coffee sometime this week to celebrate.",
    "Attached is the revised contract with the changes we discussed. Please review and sign at your convenience.",
    "Just a reminder that the office will be closed on Monday for the holiday. See everyone Tuesday.",
    "Thanks for sending over the design mockups. I have a few minor comments in the shared doc.",
    "Can you send me the latest version of the budget spreadsheet when you get a chance?",
    "The team lunch has been moved to 12:30pm in the main conference room. Hope to see you there.",
    "Your flight confirmation for next week's trip is attached. Let me know if the itinerary looks correct.",
]

URLS = [
    "http://secure-verify-account.com/login",
    "http://account-update-now.net/confirm",
    "http://bit.ly/3xKf9Lp",
    "http://mybank-secure-login.com",
    "http://prize-claim-center.com",
]

PRIZES = ["$1,000 Walmart gift card", "free iPhone 15", "$500 Amazon voucher", "all-expenses-paid vacation"]


def generate_dataset(n_per_class: int = 150):
    rows = []

    for _ in range(n_per_class):
        template = random.choice(PHISHING_TEMPLATES)
        text = template.format(url=random.choice(URLS), prize=random.choice(PRIZES))
        rows.append({"text": text, "label": "phishing"})

    for _ in range(n_per_class):
        text = random.choice(LEGITIMATE_TEMPLATES)
        rows.append({"text": text, "label": "legitimate"})

    random.shuffle(rows)
    return rows


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "emails.csv")

    rows = generate_dataset(n_per_class=150)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Synthetic demo dataset written to {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
