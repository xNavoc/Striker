import os
import sys
import argparse
import smtplib
from datetime import datetime
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Import individual model pipelines from the models/ package
try:
    from models import (
        model_hr,
        model_weibull,
        model_hits,
        model_total_bases,
        model_hr_rbi,
        model_pitcher_ks,
        model_master,
    )
except ImportError:
    # Fallback if executing directly within the models folder
    import model_hr
    import model_weibull
    import model_hits
    import model_total_bases
    import model_hr_rbi
    import model_pitcher_ks
    import model_master


def send_master_email(date_str: str, attachment_paths: list[str] = None):
    """Dispatches execution consensus cards to email recipients via SMTP."""
    sender_email = os.getenv("EMAIL_USER") or os.getenv("GMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS") or os.getenv("GMAIL_APP_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL") or sender_email

    if not sender_email or not sender_password:
        print("[!] Email dispatch skipped: EMAIL_USER or EMAIL_PASS environment variables not found.")
        return

    subject = f"⚾ Striker MLB Consensus Cards - {date_str}"
    body = (
        f"Striker predictive ensemble execution completed successfully for {date_str}.\n\n"
        f"Attached are today's synthesized consensus and clash cards."
    )

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if attachment_paths is None:
        attachment_paths = [
            f"exports/master/master_top50_card_{date_str}.png",
            f"exports/synergy/synergy_top50_card_{date_str}.png",
            f"exports/hr/hr_top50_card_{date_str}.png",
            f"exports/hits/hits_top50_card_{date_str}.png",
            f"exports/pitcher_ks/pitcher_ks_top50_card_{date_str}.png",
        ]

    attached_count = 0
    for filepath in attachment_paths:
        path = Path(filepath)
        if path.exists():
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{path.name}"'
            )
            msg.attach(part)
            attached_count += 1
            print(f"[✓] Attached {path.name}")
        else:
            print(f"[!] Warning: Attachment not found: {filepath}")

    if attached_count == 0:
        print("[!] No attachments found to send. Aborting email dispatch.")
        return

    try:
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print(f"[✓] Successfully delivered consensus cards to {recipient_email}")
    except Exception as e:
        print(f"[✗] Failed to deliver email: {e}")


def main():
    parser = argparse.ArgumentParser(description="Striker MLB Predictive Pipeline Runner")
    parser.add_argument("--target", type=str, default="all", help="Target model to run (all, hr, weibull, hits, etc.)")
    parser.add_argument("--mode", type=str, default="predict", choices=["train", "predict", "evaluate"], help="Pipeline execution mode")
    parser.add_argument("--date", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Execution date (YYYY-MM-DD)")
    args = parser.parse_args()

    date_str = args.date
    print(f"[{datetime.now()}] === STARTING MLB PREDICTIVE ENSEMBLE ({args.mode.upper()}) FOR {date_str} ===")

    # Ensure export directories exist
    for sub in ["hr", "weibull", "hits", "total_bases", "hr_rbi", "pitcher_ks", "synergy", "master"]:
        Path(f"exports/{sub}").mkdir(parents=True, exist_ok=True)

    # 1. Home Run Model
    if args.target in ["all", "hr"]:
        print(f"[{datetime.now()}] Running Clash-Refined HOME RUN Model for {date_str}...")
        if hasattr(model_hr, "run"):
            model_hr.run(date_str=date_str, mode=args.mode)

    # 2. Weibull PA Hazard Model
    if args.target in ["all", "weibull"]:
        print(f"[{datetime.now()}] Running Right-Censored WEIBULL Survival Hazard Model for {date_str}...")
        if hasattr(model_weibull, "run"):
            model_weibull.run(date_str=date_str, mode=args.mode)

    # 3. Hits & Contact Model
    if args.target in ["all", "hits"]:
        print(f"[{datetime.now()}] Running Contact & BABIP Refined HITS Model for {date_str}...")
        if hasattr(model_hits, "run"):
            model_hits.run(date_str=date_str, mode=args.mode)

    # 4. Total Bases Model
    if args.target in ["all", "total_bases"]:
        print(f"[{datetime.now()}] Running Slugging-Refined TOTAL BASES Model for {date_str}...")
        if hasattr(model_total_bases, "run"):
            model_total_bases.run(date_str=date_str, mode=args.mode)

    # 5. H+R+RBI Combo Model
    if args.target in ["all", "hr_rbi"]:
        print(f"[{datetime.now()}] Running Traffic-Refined H+R+RBI Combo Model for {date_str}...")
        if hasattr(model_hr_rbi, "run"):
            model_hr_rbi.run(date_str=date_str, mode=args.mode)

    # 6. Pitcher Strikeout Model
    if args.target in ["all", "pitcher_ks"]:
        print(f"[{datetime.now()}] Running Refined Pitcher Strikeout (K) Model for {date_str}...")
        if hasattr(model_pitcher_ks, "run"):
            model_pitcher_ks.run(date_str=date_str, mode=args.mode)

    # 7. Master Consensus Synthesis
    if args.target == "all" and args.mode == "predict":
        print(f"[{datetime.now()}] Synthesizing MASTER TOP 50 Daily Consensus for {date_str}...")
        if hasattr(model_master, "run"):
            model_master.run(date_str=date_str)

        # Dispatch output via email
        cards_to_deliver = [
            f"exports/master/master_top50_card_{date_str}.png",
            f"exports/synergy/synergy_top50_card_{date_str}.png",
            f"exports/hr/hr_top50_card_{date_str}.png",
            f"exports/hits/hits_top50_card_{date_str}.png",
            f"exports/total_bases/total_bases_top50_card_{date_str}.png",
            f"exports/hr_rbi/hr_rbi_top50_card_{date_str}.png",
            f"exports/pitcher_ks/pitcher_ks_top50_card_{date_str}.png",
        ]
        send_master_email(date_str=date_str, attachment_paths=cards_to_deliver)

    print(f"[{datetime.now()}] === ENSEMBLE EXECUTION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
