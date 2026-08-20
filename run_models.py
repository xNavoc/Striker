import argparse
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from models.model_hr import run_hr_model
from models.model_hits import run_hits_model
from models.model_total_bases import run_tb_model
from models.model_hr_rbi import run_hr_rbi_model
from models.model_pitcher_ks import run_ks_model
from models.model_weibull import run_weibull_model
from core.data_loader import get_slate_date

def ensure_export_directories():
    for sub in ['hr', 'hits', 'total_bases', 'hr_rbi', 'pitcher_ks', 'weibull', 'settlement']:
        os.makedirs(f"exports/{sub}", exist_ok=True)

def send_daily_suite_email(today_str: str):
    email_user = os.getenv("EMAIL_USER", "").strip()
    email_pass = os.getenv("EMAIL_PASS", "").strip()
    recipient = os.getenv("RECIPIENT_EMAIL", "").strip()

    if not email_user or not email_pass or not recipient:
        print("[!] Email credentials not configured. Skipping email dispatch.")
        return

    print(f"[i] Compiling MLB Target Suite email digest for {recipient}...")
    
    msg = MIMEMultipart('related')
    msg['Subject'] = f"MLB Daily Target Intelligence Suite • {today_str}"
    msg['From'] = email_user
    msg['To'] = recipient

    card_manifest = [
        ("Home Run Targets", f"exports/hr/hr_top50_card_{today_str}.png", "cid_hr"),
        ("1+ & 2+ Hit Targets", f"exports/hits/hits_top50_card_{today_str}.png", "cid_hits"),
        ("Total Bases Targets", f"exports/total_bases/total_bases_top50_card_{today_str}.png", "cid_tb"),
        ("H+R+RBI Combo Targets", f"exports/hr_rbi/hr_rbi_top50_card_{today_str}.png", "cid_combo"),
        ("Pitcher Strikeout Targets", f"exports/pitcher_ks/pitcher_ks_top50_card_{today_str}.png", "cid_ks"),
        ("Weibull Survival Targets", f"exports/weibull/weibull_top50_card_{today_str}.png", "cid_weibull"),
    ]

    cards_html = ""
    attachments = []

    for title, file_path, cid in card_manifest:
        if os.path.exists(file_path):
            cards_html += f"""
            <div style="margin-bottom: 25px;">
                <h3 style="color: #38bdf8; margin-bottom: 8px;">{title}</h3>
                <img src="cid:{cid}" style="max-width: 100%; border-radius: 8px; border: 1px solid #334155;" alt="{title}"/>
            </div>
            """
            attachments.append((file_path, cid))

    html_content = f"""
    <!DOCTYPE html>
    <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b1329; color: #f8fafc; padding: 20px;">
            <div style="max-width: 900px; margin: 0 auto;">
                <header style="border-bottom: 1px solid #1e293b; padding-bottom: 12px; margin-bottom: 20px;">
                    <h1 style="color: #f8fafc; font-size: 24px; margin: 0;">MLB Daily Target Intelligence Suite</h1>
                    <p style="color: #38bdf8; font-size: 14px; margin: 4px 0 0 0;">Unified Multi-Model Matchup Boards • {today_str}</p>
                </header>
                {cards_html}
            </div>
        </body>
    </html>
    """

    msg_alt = MIMEMultipart('alternative')
    msg.attach(msg_alt)
    msg_alt.attach(MIMEText(html_content, 'html'))

    for file_path, cid in attachments:
        try:
            with open(file_path, 'rb') as f:
                img_part = MIMEImage(f.read())
                img_part.add_header('Content-ID', f'<{cid}>')
                img_part.add_header('Content-Disposition', 'inline', filename=os.path.basename(file_path))
                msg.attach(img_part)
        except Exception as e:
            print(f"[!] Could not attach {file_path}: {e}")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_user, email_pass)
            server.sendmail(email_user, recipient, msg.as_string())
        print(f"[✓] Successfully delivered MLB Target Intelligence email to {recipient}!")
    except Exception as e:
        print(f"[!] SMTP Error delivering email: {e}")

if __name__ == "__main__":
    ensure_export_directories()

    parser = argparse.ArgumentParser(description="MLB Target Intelligence Engine")
    parser.add_argument(
        "--target",
        choices=["hr", "hits", "total_bases", "hr_rbi", "pitcher_ks", "weibull", "all"],
        default="all",
        help="Specify which dedicated target model to execute"
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "settle"],
        default="predict",
        help="Mode: predict today's slate or settle yesterday's board"
    )
    args = parser.parse_args()

    models_map = {
        "hr": run_hr_model,
        "hits": run_hits_model,
        "total_bases": run_tb_model,
        "hr_rbi": run_hr_rbi_model,
        "pitcher_ks": run_ks_model,
        "weibull": run_weibull_model
    }

    if args.target == "all":
        for target_name, runner in models_map.items():
            print(f"\n=======================================================")
            print(f"EXECUTING MODEL: [{target_name.upper()}] (MODE: {args.mode.upper()})")
            print(f"=======================================================")
            runner(mode=args.mode)
        
        if args.mode == "predict":
            today_str = get_slate_date()
            send_daily_suite_email(today_str)
    else:
        models_map[args.target](mode=args.mode)
