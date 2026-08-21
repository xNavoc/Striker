import sys
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

from core.data_loader import get_slate_date
from core.settlement_engine import get_yesterday_date
from models.model_hr import run_hr_model
from models.model_weibull import run_weibull_model
from models.model_hits import run_hits_model
from models.model_total_bases import run_tb_model
from models.model_hr_rbi import run_hr_rbi_model
from models.model_pitcher_ks import run_ks_model
from models.model_synergy import run_synergy_model
from models.model_master import run_master_leaderboard

def send_daily_email_digest(today_str: str):
    sender = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("ALERT_EMAIL_RECIPIENT", sender)

    if not sender or not password:
        print("[!] Email dispatch skipped: GMAIL_USER or GMAIL_APP_PASSWORD environment variables not found.")
        return

    print(f"[{datetime.now()}] Preparing and dispatching daily visual intelligence email to {recipient}...")

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = f"⚾ MLB Daily Intelligence Suite • Master Consensus & Prop Models • {today_str}"

    body_html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #050a18; color: #f8fafc; padding: 20px;">
        <h2 style="color: #38bdf8; border-bottom: 2px solid #1e293b; padding-bottom: 8px;">MLB Predictive Model Suite • {today_str}</h2>
        <p style="color: #cbd5e1; font-size: 14px;">All models executed successfully. Attached are today's visual board cards and data tables:</p>
        <ul style="color: #94a3b8; font-size: 13px; line-height: 1.6;">
          <li><b>Master Consensus:</b> Top 50 Unified Prop & Value Leaderboard</li>
          <li><b>Power Synergy Matrix:</b> Launch-Angle Matchup Physics (52%) + Weibull Hazards (48%)</li>
          <li><b>Home Run Model:</b> Clash-Refined Batted-Ball Trajectories & Split HR/9</li>
          <li><b>Weibull Survival:</b> Right-Censored PA Inter-HR Hazard & Drought Saturation</li>
          <li><b>Hits Model:</b> Poisson 1+ & Multi-Hit Probabilities & Contact BAA</li>
          <li><b>Total Bases Model:</b> Extra-Base Hit (XBH) Slugging & 1.5+ / 2.5+ TB Ladder</li>
          <li><b>H+R+RBI Combo:</b> Empirical Per-Game Baselines & Pitcher Traffic WHIP</li>
          <li><b>Pitcher Strikeouts:</b> Batters Faced Workload Engine & Matchup K% Ladder</li>
        </ul>
        <p style="color: #64748b; font-size: 12px; margin-top: 24px;">Automated Daily MLB Predictive Engine • Git Workflow Scheduled</p>
      </body>
    </html>
    """
    msg.attach(MIMEText(body_html, 'html'))

    attachment_paths = [
        f"exports/master/master_top50_card_{today_str}.png",
        f"exports/master/master_top50_{today_str}.csv",
        f"exports/synergy/synergy_top50_card_{today_str}.png",
        f"exports/synergy/synergy_top50_{today_str}.csv",
        f"exports/hr/hr_top50_card_{today_str}.png",
        f"exports/hr/hr_top50_{today_str}.csv",
        f"exports/weibull/weibull_top50_card_{today_str}.png",
        f"exports/weibull/weibull_top50_{today_str}.csv",
        f"exports/hits/hits_top50_card_{today_str}.png",
        f"exports/hits/hits_top50_{today_str}.csv",
        f"exports/total_bases/total_bases_top50_card_{today_str}.png",
        f"exports/total_bases/total_bases_top50_{today_str}.csv",
        f"exports/hr_rbi/hr_rbi_top50_card_{today_str}.png",
        f"exports/hr_rbi/hr_rbi_top50_{today_str}.csv",
        f"exports/pitcher_ks/pitcher_ks_top50_card_{today_str}.png",
        f"exports/pitcher_ks/pitcher_ks_top50_{today_str}.csv",
    ]

    attached_count = 0
    for file_path in attachment_paths:
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
                msg.attach(part)
                attached_count += 1
            except Exception as e:
                print(f"[!] Could not attach {file_path}: {e}")
        else:
            print(f"[!] Warning: Attachment not found: {file_path}")

    if attached_count == 0:
        print("[!] No attachments generated (likely due to empty slate or API block). Dispatching fallback structural alert.")
        
        # Reset the MIMEMultipart shell for a distinct alert payload
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = f"⚾ MLB Daily Intelligence Suite • Pipeline Alert • {today_str}"

        alert_html = f"""
        <html>
          <body style="font-family: Arial, sans-serif; background-color: #050a18; color: #f8fafc; padding: 20px;">
            <h2 style="color: #f87171; border-bottom: 2px solid #1e293b; padding-bottom: 8px;">Pipeline Alert: Empty Data Slate</h2>
            <p style="color: #cbd5e1; font-size: 14px;">The automated pipeline executed successfully, but <b>zero actionable targets</b> were generated.</p>
            <ul style="color: #94a3b8; font-size: 13px; line-height: 1.6;">
              <li>This is typically caused by a scheduled <b>MLB off-day</b> (no games played).</li>
              <li>Alternatively, the MLB Stats API experienced an outage or returned empty data structures for the date `{today_str}`.</li>
            </ul>
            <p style="color: #64748b; font-size: 12px; margin-top: 24px;">Automated Daily MLB Predictive Engine • {today_str}</p>
          </body>
        </html>
        """
        msg.attach(MIMEText(alert_html, 'html'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        if attached_count > 0:
            print(f"[✓] Daily email digest with {attached_count} attachments delivered to {recipient}.")
        else:
            print(f"[✓] Zero-data alert email delivered to {recipient}.")
    except Exception as e:
        print(f"[!] Failed to send email digest: {e}")

def parse_mode():
    args = sys.argv[1:]
    if "settle" in args:
        return "settle"
    for i, arg in enumerate(args):
        if arg in ["--mode", "-m"] and i + 1 < len(args):
            return args[i + 1].lower()
    return "predict"

def main():
    mode = parse_mode()
    target_date = get_yesterday_date() if mode == "settle" else get_slate_date()
    print(f"[{datetime.now()}] === STARTING MLB PREDICTIVE ENSEMBLE ({mode.upper()}) FOR {target_date} ===")

    for d in ['hr', 'weibull', 'synergy', 'hits', 'total_bases', 'hr_rbi', 'pitcher_ks', 'master', 'settlement']:
        os.makedirs(f"exports/{d}", exist_ok=True)

    # 1. Base Domain Models
    run_hr_model(mode)
    run_weibull_model(mode)
    run_hits_model(mode)
    run_tb_model(mode)
    run_hr_rbi_model(mode)
    run_ks_model(mode)

    # 2. Synthesis Models & Dispatch
    if mode == "predict":
        run_synergy_model(mode)
        run_master_leaderboard(mode)
        send_daily_email_digest(target_date)

    print(f"[{datetime.now()}] === ENSEMBLE EXECUTION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
