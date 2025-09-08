import sys, asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import sys
import asyncio
from pyrogram.enums import ChatMemberStatus
import sys, asyncio




import requests
from requests.exceptions import Timeout, RequestException
import logging
import random
from telegram import Update, ChatPermissions as PTBChatPermissions
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode
from telethon import TelegramClient
from telegram import ChatPermissions as PTBChatPermissions
from telegram.helpers import escape_markdown
from telethon.errors import ChatAdminRequiredError, UserAdminInvalidError
import re
from telethon.tl.types import ChatParticipantAdmin, ChatParticipantCreator
from datetime import datetime, timedelta, time as dt_time
import time  # standard module for time.time()
import time 
from pyrogram.enums import ParseMode
import io
import asyncio
from groq import Client
import json
import os
import requests 
from pyrogram import Client as PyroClient, filters as pyro_filters
from pyrogram.types import InputMediaPhoto
from pyrogram.errors import ChatAdminRequired
import tempfile
from PIL import Image, ImageDraw, ImageFont
from telegram import InputFile
from utils.render import create_quote_image
from typing import Dict, Set
from io import BytesIO
from telegram import InputSticker, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, Forbidden
import datetime as pydatetime
from telethon.tl.functions.channels import EditBannedRequest
from telethon.tl.types import ChatBannedRights
from telethon import events
import subprocess
import hashlib
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import urllib.parse
from groq import Client
import base64
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes
import cv2
import google.generativeai as genai
from urllib.parse import quote_plus
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
import re
import os
import tempfile
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
import re
from telegram.ext import MessageHandler, ApplicationBuilder
from telegram.ext import filters as ptb_filters
from duckduckgo_search import DDGS
import html
import time
import re
import html
from huggingface_hub import InferenceClient
import os
from pyrogram import filters
from pyrogram.types import Message
import asyncio
import io
from pyrogram.types import ChatPermissions as PyroChatPermissions
import re
from telegram.error import BadRequest
from pyrogram.errors import ChatNotModified
from telegram import InputFile
from io import BytesIO
from PIL import Image
from urllib.parse import quote_plus
from telegram import Update
from telegram.ext import ContextTypes
import cv2
import platform
import math
from telegram.ext import CallbackQueryHandler
import requests
from telegram import Update
from telegram.ext import ContextTypes
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
from datetime import datetime, timezone
from html import escape 
def boldify(text: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)








import os
from pyrogram import Client as PyroClient
import google.generativeai as genai
from groq import Client
API_ID = int(os.getenv("MNG_API_ID"))
API_HASH = os.getenv("MNG_API_HASH")
BOT_TOKEN = os.getenv("MNG_BOT_TOKEN")

pyro_client = PyroClient(
    "pic_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    in_memory=True
)


# AI keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Client(api_key=GROQ_API_KEY)

GEMINI_KEY = os.getenv("GEMINI_KEY")
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")




# === Initialize Logging ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# === Initialize Telethon client for MTProto lookups ===
tclient = TelegramClient("userbot_session", API_ID, API_HASH)
async def init_telethon():
    global tclient
    if not tclient.is_connected():
        await tclient.start()
        print("[mng2] Telethon client started")

temp_mutes = {}  
filters_dict = {}
ongoing_tagall = {}
afk_users = {}
waifu_data = {}  
WAIFU_EXPIRY_SECONDS = 86400
warnings = {}    
warn_reasons = {}   # {user_id: [reasons]}
blacklist = set()       
approved_users = set()  
filters_dict = {}              
welcome_message = None
nightmode = False
flood_settings = {}           
flood_modes = {}              
user_flood_counts = {}        
default_flood_mode = ("tmute", "30m")  
blacklist_modes = {}  # chat_id -> action (e.g., 'tmute', 'mute', 'ban', 'warn', 'kick', 'delete')
FILTERS_FILE = "filters.json"
sending_pics = False
pic_enabled = {}  # chat_id: True/False
# LOCK SYSTEM STORAGE AND LOGIC (from lock.py)
LOCKS_FILE = "locks.json"
BOT_USERNAME = "Yukino_Roxbot"  # (set to your bot, without @)
REGISTERED_USERS = set()
# Nightmode status per chat
nightmode_status = {}
welcome_settings = {}  # {chat_id: {"enabled": bool, "message": dict}}
goodbye_settings = {}  # {chat_id: {"enabled": bool, "message": dict}}
captcha_settings = {}   # {chat_id: True/False}
sticker_blacklist = {}   # chat_id -> set of sticker_file_ids
pack_blacklist = {}      # chat_id -> set of set_name (sticker pack names)
sticker_map = {}         # short_id -> real sticker.file_id
pack_map = {}            # short_id -> real pack_name



# Track approved users (chat_id: set of user_ids)
approved_users = {}

# Structure: { str(chat_id): {"locked": set_of_types_as_list} }
_state: Dict[str, Dict[str, Set[str]]] = {}

VALID_LOCKS = {
    "all", "audio", "bots", "buttons", "contact", "document", "egames", "forward",
    "game", "gif", "info", "invite", "inline", "location", "media", "messages",
    "text", "other", "photos", "pin", "poll", "previews", "rtl", "stickers",
    "url", "video", "voice",
}

ALIAS = {
    "text": "messages",
}

BIDI_FORBIDDEN = { "\u202A", "\u202B", "\u202D", "\u202E", "\u202C", "\u2066", "\u2067", "\u2068", "\u2069", "\u200F" }
URL_REGEX = re.compile(r"https?://|t\.me/|telegram\.me/|telegram\.dog/|joinchat/", re.IGNORECASE)
INVITE_REGEX = re.compile(r"t\.me/joinchat/|t\.me/\+|telegram\.me/\+", re.IGNORECASE)




def load_lock_state():
    global _state
    if os.path.exists(LOCKS_FILE):
        try:
            with open(LOCKS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
                _state = { k: {"locked": set(v.get("locked", []))} for k, v in raw.items() }
        except Exception:
            _state = {}
    else:
        _state = {}


def get_pack_name(user, is_video=False, is_animated=False, version=1):
    if is_animated:
        base = f"u{user.id}_animated"
    elif is_video:
        base = f"u{user.id}_video"
    else:
        base = f"u{user.id}"
    if version > 1:
        base += f"_v{version}"
    return f"{base}_by_{BOT_USERNAME}".lower()

def get_pack_title(user, is_video=False, is_animated=False, version=1):
    if is_animated:
        base = f"{user.first_name}'s Animated Pack"
    elif is_video:
        base = f"{user.first_name}'s Video Pack"
    else:
        base = f"{user.first_name}'s Pack"
    return base if version == 1 else f"{base} v{version}"



def save_lock_state():
    try:
        serializable = { k: {"locked": sorted(list(v.get("locked", set())))} for k, v in _state.items() }
        with open(LOCKS_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_locked(chat_id: int) -> Set[str]:
    cid = str(chat_id)
    if cid not in _state:
        _state[cid] = {"locked": set()}
    return _state[cid]["locked"]

def lock(chat_id: int, key: str) -> bool:
    key = key.lower()
    key = ALIAS.get(key, key)
    if key not in VALID_LOCKS:
        return False
    locked = get_locked(chat_id)
    if key == "all":
        locked.update(VALID_LOCKS - {"all"})
    else:
        locked.add(key)
    save_lock_state()
    return True

def unlock(chat_id: int, key: str) -> bool:
    key = key.lower()
    key = ALIAS.get(key, key)
    if key not in VALID_LOCKS:
        return False
    locked = get_locked(chat_id)
    if key == "all":
        locked.clear()
    else:
        locked.discard(key)
    save_lock_state()
    return True

def format_response(text: str) -> str:
    """
    Cleans up Gemini/Google responses for Telegram (HTML mode).
    Converts **bold** to <b>...</b> properly.
    Escapes other HTML special chars.
    """
    if not text:
        return ""

    # First escape HTML special chars
    safe_text = html.escape(text)

    # Convert Markdown-style bold (**...**) into <b>...</b>
    safe_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe_text)

    return safe_text

# --- Captcha state: (chat_id, user_id) -> {"answer": int, "tries": int, "question": str}
captcha_state: dict[tuple[int, int], dict] = {}

def _make_captcha():
    a, b = random.randint(1, 10), random.randint(1, 10)
    if random.choice([True, False]):
        return f"{a} + {b}", a + b
    else:
        a, b = max(a, b), min(a, b)  # avoid negatives
        return f"{a} - {b}", a - b

def _build_9_options_keyboard(chat_id: int, user_id: int, correct: int) -> InlineKeyboardMarkup:
    # possible answers for our ops are 0..20 (inclusive). Build 9 unique options incl. correct.
    options = {correct}
    while len(options) < 9:
        options.add(random.randint(0, 20))
    opts = list(options)
    random.shuffle(opts)
    rows = [opts[i:i+3] for i in range(0, 9, 3)]
    keyboard = [
        [InlineKeyboardButton(str(v), callback_data=f"capans:{chat_id}:{user_id}:{v}") for v in row]
        for row in rows
    ]
    return InlineKeyboardMarkup(keyboard)



def contains_bidi(text: str) -> bool:
    return any(ch in BIDI_FORBIDDEN for ch in text)

def has_url_entities(msg) -> bool:
    # msg is a telegram.Update.message here!
    if not (msg.text or msg.caption):
        return False
    if getattr(msg, "entities", None):
        for ent in msg.entities:
            if ent.type in ("url", "text_link", "mentionname", "web_app_data"):
                return True
    # Fallback: regex
    return bool(URL_REGEX.search(msg.text or msg.caption or ""))

def has_invite_link(msg) -> bool:
    text = (msg.text or msg.caption or "")
    return bool(INVITE_REGEX.search(text))



# Load filters at startup
if os.path.exists(FILTERS_FILE):
    with open(FILTERS_FILE, "r") as f:
        filters_db = json.load(f)  # dict[str(chat_id)] -> dict[keyword] -> filter data
else:
    filters_db = {}

# === Helper Functions ===

HARD_CODED_PROMPT = "You are a helpful intelligent and updated female assistant named 𝖣𝗂𝗄𝗌𝗁𝗂𝗄𝖺 ᥫ᭡. Please answer creatively and politely.Try to use shorter replies.: " #can use any prompt 

def save_filters():
    with open(FILTERS_FILE, "w") as f:
        json.dump(filters_db, f)


async def start_pyro():
    await pyro_client.start()


ANILIST_URL = "https://graphql.anilist.co"

ANIME_QUERY = """
query ($id: Int, $search: String) {
  Media(id: $id, search: $search, type: ANIME) {
    id
    idMal
    title {
      romaji
      english
      native
    }
    description(asHtml: false)
    source
    type
    averageScore
    duration
    status
    episodes
    nextAiringEpisode {
      airingAt
      episode
    }
    genres
    tags {
      name
    }
    trailer {
      site
      id
    }
    coverImage {
      large
      medium
    }
    siteUrl
    characters(perPage: 10) {
      edges {
        role
        node {
          id
          name {
            full
          }
          image {
            large
          }
          siteUrl
        }
      }
    }
  }
}
"""



CHARACTER_QUERY = """
query ($id: Int, $search: String) {
  Character(id: $id, search: $search) {
    id
    name {
      full
      native
    }
    image {
      large
    }
    description(asHtml: false)
    siteUrl
    media(perPage: 10) {
      edges {
        node {
          id
          title {
            romaji
          }
          type
          siteUrl
        }
        voiceActors(language: JAPANESE) {
          id
          name {
            full
          }
          siteUrl
        }
      }
    }
  }
}
"""







RELATIONS_QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    relations {
      edges {
        relationType
        node {
          title { romaji }
          type
          siteUrl
        }
      }
    }
  }
}
"""



async def prepare_static(bot, file_id):
    new_file = await bot.get_file(file_id)
    file_bytes = await new_file.download_as_bytearray()
    image = Image.open(BytesIO(file_bytes)).convert("RGBA")
    max_size = 512
    ratio = min(max_size / image.width, max_size / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    new_img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    new_img.paste(image, ((512 - new_size[0]) // 2, (512 - new_size[1]) // 2))
    bio = BytesIO()
    bio.name = "sticker.png"
    new_img.save(bio, "PNG", optimize=True)
    bio.seek(0)
    return bio

async def prepare_video(bot, file_id):
    new_file = await bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
        await new_file.download_to_drive(temp_in.name)
        temp_in_path = temp_in.name
    temp_out_path = tempfile.mktemp(suffix=".webm")
    cmd = [
        "ffmpeg",
        "-i", temp_in_path,
        "-t", "3",
        "-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2:color=0x00000000,fps=30,format=yuva420p",
        "-an",
        "-c:v", "libvpx-vp9",
        "-b:v", "256k",
        "-pix_fmt", "yuva420p",
        "-y", temp_out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bio = BytesIO(open(temp_out_path, "rb").read())
    bio.name = "sticker.webm"
    os.remove(temp_in_path)
    os.remove(temp_out_path)
    return bio

async def prepare_animated(bot, file_id):
    new_file = await bot.get_file(file_id)
    file_bytes = await new_file.download_as_bytearray()
    bio = BytesIO(file_bytes)
    bio.name = "sticker.tgs"
    return bio




# --- Store chat-wise settings ---
editdelete_enabled = {}  # chat_id -> bool
edit_message_jobs = {}  # {(chat_id, message_id): job} to track scheduled deletions


async def has_permission(update, perms):
    """
    Utility already used in your code.
    Checks if the user has the required admin permissions.
    """
    try:
        member = await update.effective_chat.get_member(update.effective_user.id)
        for perm in perms:
            if not getattr(member, perm, False):
                return False
        return True
    except Exception:
        return False




from telethon.tl.functions.channels import GetParticipantsRequest
from telethon.tl.types import ChannelParticipantsSearch

@pyro_client.on_message(pyro_filters.command("zombies"))
async def zombies(client, message):
    # Admin-only
    if not await is_admin_pyro(client, message.chat.id, message.from_user.id):
        return await message.reply_text("🚫 Only admins can use this command.")

    chat_id = message.chat.id
    deleted_ids = set()

    # Iterate only current members; skip LEFT/BANNED to avoid false positives
    async for member in client.get_chat_members(chat_id):
        u = member.user
        if not u:
            continue
        if member.status in (ChatMemberStatus.LEFT, ChatMemberStatus.BANNED):
            continue
        if getattr(u, "is_deleted", False):
            deleted_ids.add(u.id)

    if not deleted_ids:
        return await message.reply_text("No deleted accounts found in this chat.")

    ids_str = ", ".join(str(uid) for uid in sorted(deleted_ids))
    await message.reply_text(f"Deleted accounts: {ids_str}\nTotal: {len(deleted_ids)}")


@pyro_client.on_message(pyro_filters.command("rzombies"))
async def rzombies(client, message):
    # Admin-only
    if not await is_admin_pyro(client, message.chat.id, message.from_user.id):
        return await message.reply_text("🚫 Only admins can use this command.")

    chat_id = message.chat.id
    removed_count = 0
    skipped_admin = 0

    async for member in client.get_chat_members(chat_id):
        u = member.user
        if not u:
            continue
        if member.status in (ChatMemberStatus.LEFT, ChatMemberStatus.BANNED):
            continue
        if getattr(u, "is_deleted", False):
            if await is_admin_pyro(client, chat_id, u.id):
                skipped_admin += 1
                continue
            try:
                await client.ban_chat_member(chat_id, u.id)
                await client.unban_chat_member(chat_id, u.id)
                removed_count += 1
            except ChatAdminRequired:
                return await message.reply_text("🚫 I don't have permission to remove members.")
            except Exception:
                continue

    await message.reply_text(
        f"Removed {removed_count} deleted accounts. Skipped {skipped_admin} (admins)."
    )




async def edited_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle edited messages and schedule deletion if enabled (Consolidated)"""
    print(f"[DEBUG] Edited message handler called, update type: {type(update)}")
    print(f"[DEBUG] Has edited_message: {bool(update.edited_message)}")
    
    if not update.edited_message:
        print(f"[DEBUG] No edited_message found in update")
        return 

    msg = update.edited_message

    # 🚫 Skip whispers/system edits: if message has inline buttons, skip
    if msg.reply_markup and msg.reply_markup.inline_keyboard:
        print(f"[DEBUG] Skipping edit with inline buttons (likely whisper) for msg {msg.message_id}")
        return

    chat = msg.chat
    chat_id = chat.id
    message_id = msg.message_id
    user = msg.from_user
    user_id = user.id if user else None
    
    print(f"[DEBUG] Edited message - Chat: {chat_id}, Message: {message_id}, User: {user_id}")
    print(f"[DEBUG] Edit-delete enabled states: {editdelete_enabled}")
    
    # Check if edit-delete is enabled in this chat
    is_enabled = editdelete_enabled.get(chat_id, False)
    print(f"[DEBUG] Edit-delete enabled for chat {chat_id}: {is_enabled}")
    
    if not is_enabled:
        print(f"[DEBUG] Edit-delete not enabled for this chat, returning")
        return
    
    # Skip if no user_id (shouldn't happen but safety check)
    if not user_id:
        print(f"[DEBUG] No user_id found, returning")
        return

    # ✅ Ignore admins and owner
    try:
        member = await chat.get_member(user_id)
        if member.status in ("administrator", "creator"):
            print(f"[DEBUG] Skipping edit deletion for admin/owner {user_id} in chat {chat_id}")
            return
    except Exception as e:
        print(f"[DEBUG] Error checking member status for {user_id}: {e}")
        return
    
    print(f"[DEBUG] Processing edited message - scheduling deletion")
    
    # Cancel any existing deletion job for this message
    existing_job = edit_message_jobs.pop((chat_id, message_id), None)
    if existing_job:
        print(f"[DEBUG] Cancelled existing job for message {message_id}")
        existing_job.schedule_removal()
    
    # Schedule new deletion job after 30 seconds
    try:
        job = context.job_queue.run_once(
            delete_edited_message,
            when=30,
            data=(chat_id, message_id),
            name=f"delete_edited_{chat_id}_{message_id}"
        )
        
        edit_message_jobs[(chat_id, message_id)] = job
        print(f"[DEBUG] Scheduled deletion job for message {message_id} in 30 seconds")
        
        try:
            warn_msg = await context.bot.send_message(
                chat_id=chat_id,
                text=f"<a href='tg://user?id={user_id}'>{user.first_name}</a> edited their message. It will be deleted in 30 seconds!",
                reply_to_message_id=message_id,
                parse_mode="HTML"
            )

            # Schedule deletion of the warning message too
            context.job_queue.run_once(
                delete_edited_message,
                when=30,
                data=(chat_id, warn_msg.message_id),
                name=f"delete_warning_{chat_id}_{warn_msg.message_id}"
            )

        except Exception as e:
            print(f"[DEBUG] Failed to send confirmation message: {e}")
        
    except Exception as e:
        print(f"[DEBUG] Error scheduling deletion job: {e}")


async def editdelete_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /editdelete on/off command"""
    # Use the same permission check as other commands
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to use this command.")
    
    if not context.args or context.args[0].lower() not in ["on", "off"]:
        return await update.message.reply_text("Usage: /editdelete on|off")
    
    chat_id = update.effective_chat.id
    enabled = context.args[0].lower() == "on"
    editdelete_enabled[chat_id] = enabled
    
    status = "enabled" if enabled else "disabled"
    print(f"[DEBUG] Edit-delete {status} in chat {chat_id}")
    await update.message.reply_text(f"✅ Edit-delete feature {status} in this chat.")


async def delete_edited_message(context: ContextTypes.DEFAULT_TYPE):
    """Callback function to delete an edited message after 30 seconds"""
    job = context.job
    chat_id, message_id = job.data
    
    print(f"[DEBUG] delete_edited_message callback triggered for message {message_id} in chat {chat_id}")
    
    try:
        print(f"[DEBUG] Attempting to delete message {message_id} in chat {chat_id}")
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        print(f"[DEBUG] Successfully deleted message {message_id} in chat {chat_id}")
    except Exception as e:
        # Message might already be deleted or not accessible
        print(f"[DEBUG] Failed to delete edited message {message_id} in chat {chat_id}: {e}")
    
    # Remove the job from tracking
    edit_message_jobs.pop((chat_id, message_id), None)
    print(f"[DEBUG] Removed job tracking for message {message_id} in chat {chat_id}")


import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import CommandHandler, MessageHandler, filters, ContextTypes






async def character_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /character <name>")
        return

    search_name = " ".join(context.args)
    variables = {"search": search_name}

    response = requests.post(
        ANILIST_URL,
        json={"query": CHARACTER_QUERY, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json"}
    ).json()

    print("=== DEBUG Character Response ===")
    print(json.dumps(response, indent=2))

    if "errors" in response or not response.get("data", {}).get("Character"):
        await update.message.reply_text("❌ Character not found.")
        return

    char = response["data"]["Character"]

    # Basic fields
    name = char["name"]["full"]
    native = char["name"].get("native", "")
    cid = char["id"]
    url = char["siteUrl"]
    image = char["image"]["large"]

    # Find first JP voice actor (from media edges)
    vactor = None
    for edge in char.get("media", {}).get("edges", []):
        if edge.get("voiceActors"):
            vactor = edge["voiceActors"][0]
            break

    va_text = f"<a href='{vactor['siteUrl']}'>{vactor['name']['full']}</a>" if vactor else "N/A"

    # Caption
    caption = (
        f"{native}\n"
        f"({name})\n"
        f"<b>ID:</b> <code>{cid}</code>\n\n"
        f"<b>Voice Actor:</b> {va_text}\n\n"
        f"🔗 <a href='{url}'>Visit Website</a>"
    )

    # Inline buttons
    keyboard = [
        [InlineKeyboardButton("📖 Description", callback_data=f"char_desc:{cid}")],
        [InlineKeyboardButton("📺 List Series", callback_data=f"char_series:{cid}")]
    ]

    await update.message.reply_photo(
        photo=image,
        caption=caption,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )



async def character_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    char_id = int(data.split(":")[1])
    variables = {"id": char_id}
    response = requests.post(
        ANILIST_URL,
        json={"query": CHARACTER_QUERY, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json"}
    ).json()
    char = response["data"]["Character"]

    # --- Description ---
    if data.startswith("char_desc:"):
        desc = char.get("description", "No description available")
        desc = re.sub(r"<.*?>", "", desc)  # strip HTML tags
        if len(desc) > 900:
            desc = desc[:900] + "..."

        text = f"<b>Description:</b>\n\n{desc}"
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f"char_home:{char_id}")]]
        await query.edit_message_caption(
            caption=text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # --- Series (Anime, italic clickable) ---
    elif data.startswith("char_series:") or data.startswith("char_anime:"):
        anime_list = [
            f"• <i><a href='{edge['node']['siteUrl']}'>{edge['node']['title']['romaji']}</a></i>"
            for edge in char["media"]["edges"] if edge["node"]["type"] == "ANIME"
        ]
        text = "📺 <b>Anime:</b>\n" + ("\n".join(anime_list) if anime_list else "No anime found.")

        keyboard = [
            [InlineKeyboardButton("📚 Manga", callback_data=f"char_manga:{char_id}")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"char_home:{char_id}")]
        ]
        await query.edit_message_caption(
            caption=text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # --- Series (Manga, italic clickable) ---
    elif data.startswith("char_manga:"):
        manga_list = [
            f"• <i><a href='{edge['node']['siteUrl']}'>{edge['node']['title']['romaji']}</a></i>"
            for edge in char["media"]["edges"] if edge["node"]["type"] == "MANGA"
        ]
        text = "📚 <b>Manga:</b>\n" + ("\n".join(manga_list) if manga_list else "No manga found.")

        keyboard = [
            [InlineKeyboardButton("📺 Anime", callback_data=f"char_anime:{char_id}")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"char_home:{char_id}")]
        ]
        await query.edit_message_caption(
            caption=text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # --- Back to Character Home ---
    elif data.startswith("char_home:"):
        caption = (
            f"{char['name']['native']}\n({char['name']['full']})\n"
            f"<b>ID:</b> <code>{char['id']}</code>\n\n"
            f"🔗 <a href='{char['siteUrl']}'>Visit Website</a>"
        )

        keyboard = [
            [InlineKeyboardButton("📖 Description", callback_data=f"char_desc:{char_id}")],
            [InlineKeyboardButton("📺 List Series", callback_data=f"char_series:{char_id}")]
        ]
        await query.edit_message_caption(
            caption=caption,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )



async def kang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message = update.message

    # Make sure user started bot privately first
    if user.id not in REGISTERED_USERS:
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("⚡️ Start bot", url=f"https://t.me/{BOT_USERNAME}?start=kang")]
        ])
        await message.reply_text(
            "ꜱᴛᴀʀᴛ ᴍᴇ ɪɴ ᴘᴍ ᴛᴏ ᴋᴀɴɢ ꜱᴛɪᴄᴋᴇʀꜱ",
            reply_markup=keyboard
        )
        return

    if not message.reply_to_message:
        await message.reply_text("Reply to a sticker, photo, or gif to kang.")
        return

    # Send waiting message immediately
    wait_msg = await message.reply_text("⏳ Please wait while adding your sticker...")

    args = message.text.strip().split()
    version = 1
    if len(args) > 1 and args[1].isdigit():
        version = int(args[1])

    reply = message.reply_to_message
    is_video = False
    is_animated = False
    sticker_file = None
    emoji = "🤔"

    if reply.sticker:
        sticker = reply.sticker
        emoji = sticker.emoji or "🤔"
        if sticker.is_animated:
            is_animated = True
            sticker_file = await prepare_animated(context.bot, sticker.file_id)
        elif sticker.is_video:
            is_video = True
            sticker_file = await prepare_video(context.bot, sticker.file_id)
        else:
            sticker_file = await prepare_static(context.bot, sticker.file_id)
    elif reply.photo:
        sticker_file = await prepare_static(context.bot, reply.photo[-1].file_id)
    elif reply.animation:  # gif/mp4
        is_video = True
        sticker_file = await prepare_video(context.bot, reply.animation.file_id)
    else:
        await wait_msg.edit_text("❌ Unsupported file type. Reply to a sticker, photo, or gif.")
        return

    pack_name = get_pack_name(user, is_video, is_animated, version)
    pack_title = get_pack_title(user, is_video, is_animated, version)

    print("==== DEBUG ====")
    print("PACK NAME :", pack_name)
    print("PACK TITLE:", pack_title)
    print("IS VIDEO :", is_video)
    print("IS ANIM :", is_animated)
    print("=============")

    if is_animated:
        sticker_format = "animated"
    elif is_video:
        sticker_format = "video"
    else:
        sticker_format = "static"

    # reset file buffer before creating InputSticker
    sticker_file.seek(0)
    input_sticker = InputSticker(sticker=sticker_file, emoji_list=[emoji])



    try:
        sticker_file.seek(0)
        await context.bot.create_new_sticker_set(
            user_id=user.id,
            name=pack_name,
            title=pack_title,
            stickers=[input_sticker],
            sticker_format=sticker_format,
        )

    except Exception as ce:
        print("CREATE ERROR:", ce)
        if "already occupied" in str(ce):
            sticker_file.seek(0)
            await context.bot.add_sticker_to_set(
                user_id=user.id,
                name=pack_name,
                sticker=input_sticker,
            )


        else:
            await wait_msg.edit_text(f"❌ Failed to create sticker pack:\n{ce}")
            return

    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("👉 View Sticker Pack", url=f"https://t.me/addstickers/{pack_name}")]]
    )

    sticker_file.seek(0)
    # Update wait message before sending final sticker
    await wait_msg.edit_text("✅ Sticker added successfully!")
    await message.reply_sticker(sticker_file, reply_markup=keyboard)



async def welcome_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("🚫 Only admins can toggle welcome messages.")

    if not context.args or context.args[0].lower() not in ["on", "off"]:
        return await update.message.reply_text("Usage: /welcome on|off")

    chat_id = update.effective_chat.id
    enabled = context.args[0].lower() == "on"
    welcome_settings.setdefault(chat_id, {})["enabled"] = enabled
    await update.message.reply_text(f"✅ Welcome messages {'enabled' if enabled else 'disabled'} in this chat.")


async def set_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("🚫 Only admins can set welcome message.")

    chat_id = update.effective_chat.id
    welcome_settings.setdefault(chat_id, {"enabled": True})

    # If user replied to a message → save that message as template
    if update.message.reply_to_message:
        msg = update.message.reply_to_message
        welcome_settings[chat_id]["message"] = msg.to_dict()
        return await update.message.reply_text("✅ Welcome message set from the replied message.")

    # Else, take args as plain text
    text = " ".join(context.args)
    if not text:
        return await update.message.reply_text("Usage:\nReply with /setwelcome\nor /setwelcome <text>")
    welcome_settings[chat_id]["message"] = {"text": text}
    await update.message.reply_text("✅ Welcome message set.")


async def greet_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    chat_id = chat.id

    # If captcha is OFF, just run the normal welcome and return
    if not captcha_settings.get(chat_id, False):
        # your existing welcome logic
        if chat_id in welcome_settings and welcome_settings[chat_id].get("enabled", True):
            msg_data = welcome_settings[chat_id].get("message")
            if msg_data:
                for member in update.message.new_chat_members:
                    mention = f"<a href='tg://user?id={member.id}'>{member.first_name}</a>"
                    chatname = chat.title
                    def replace_vars(text): return text.replace("{mention}", mention).replace("{chat}", chatname)
                    if "text" in msg_data:
                        await update.message.reply_text(replace_vars(msg_data["text"]), parse_mode=ParseMode.HTML)
                    else:
                        try:
                            await context.bot.copy_message(
                                chat_id=chat_id,
                                from_chat_id=chat_id,
                                message_id=msg_data["message_id"],
                                caption=replace_vars(msg_data.get("caption", "")) if msg_data.get("caption") else None,
                                parse_mode=ParseMode.HTML
                            )
                        except Exception as e:
                            await update.message.reply_text(f"⚠️ Failed to send welcome message: {e}")
        return

    # --- Captcha is ON: mute + welcome with deep link button
    for member in update.message.new_chat_members:
        # 1) Mute immediately (bot must have "Restrict members" admin right)
        try:
            await chat.restrict_member(member.id,
                permissions=PTBChatPermissions(can_send_messages=False))

        except Exception as e:
            # don’t block the flow if we fail to mute (e.g., missing rights or user is admin)
            print(f"Captcha mute failed for {member.id} in {chat_id}: {e}")

        # 2) Build deep link that opens DM with /start payload
        start_payload = f"cap_{chat_id}_{member.id}"
        url = f"https://t.me/{BOT_USERNAME}?start={start_payload}"

        # 3) Send welcome text/media + captcha button (fallback to “Hi” if no welcome set)
        btn = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Solve Captcha", url=url)]])
        mention = f"<a href='tg://user?id={member.id}'>{member.first_name}</a>"
        chatname = chat.title

        msg_data = (welcome_settings.get(chat_id) or {}).get("message")
        if msg_data:  # custom welcome
            def replace_vars(text): return text.replace("{mention}", mention).replace("{chat}", chatname)
            if "text" in msg_data:
                await update.message.reply_text(replace_vars(msg_data["text"]), parse_mode=ParseMode.HTML, reply_markup=btn)
            else:
                try:
                    await context.bot.copy_message(
                        chat_id=chat_id,
                        from_chat_id=chat_id,
                        message_id=msg_data["message_id"],
                        caption=replace_vars(msg_data.get("caption", "")) if msg_data.get("caption") else None,
                        parse_mode=ParseMode.HTML,
                        reply_markup=btn
                    )
                except Exception as e:
                    await update.message.reply_text(f"⚠️ Failed to send welcome message: {e}", reply_markup=btn)
        else:  # fallback welcome
            await update.message.reply_text(
                f"Hi {mention}! Tap the button below to verify and unmute.",
                parse_mode=ParseMode.HTML,
                reply_markup=btn
            )



async def lock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info", "can_delete_messages"]):
        return await update.message.reply_text("🚫 You need both 'Can Change Info' and 'Can Delete Messages' permissions to manage locks.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("🚫 Only admins can use this command.")
        return
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /lock <type>\nUse /locks to see all types."
        )
        return
    key = context.args[0].lower()
    ok = lock(update.effective_chat.id, key)
    if not ok:
        await update.message.reply_text("❌ Unknown lock type. Use /locks to see valid types.")
        return
    await update.message.reply_text(f"✅ {key} locked.")

async def unlock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info", "can_delete_messages"]):
        return await update.message.reply_text("🚫 You need both 'Can Change Info' and 'Can Delete Messages' permissions to manage locks.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("🚫 Only admins can use this command.")
        return
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /unlock <type>\nUse /locks to see all types."
        )
        return
    key = context.args[0].lower()

    ok = unlock(update.effective_chat.id, key)
    if not ok:
        await update.message.reply_text("❌ Unknown lock type. Use /locks to see valid types.")
        return
    await update.message.reply_text(f"✅ {key} unlocked.")

async def locks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    locked = get_locked(update.effective_chat.id)
    pretty = "\n".join(
        [f"• {k} — {'ON' if k in locked else 'OFF'}" for k in sorted(VALID_LOCKS - {"all"})]
    )
    await update.message.reply_text(
        "**Current locks:**\n" + pretty + "\n\nUse /lock <type> or /unlock <type>."
    )


async def is_admin_pyro(client, chat_id, user_id):
    try:
        member = await client.get_chat_member(chat_id, user_id)
        return member.privileges is not None or member.status in ("administrator", "creator", "owner")
    except:
        return False
    

@pyro_client.on_message(pyro_filters.command("enablepic"))
async def enable_pic(client, message):
    if not await is_admin_pyro(client, message.chat.id, message.from_user.id):
        return await message.reply_text("🚫 Only admins can use this command.")
    pic_enabled[message.chat.id] = True
    await message.reply_text("✅ Profile picture sending is now *enabled* in this chat.")

@pyro_client.on_message(pyro_filters.command("disablepic"))
async def disable_pic(client, message):
    if not await is_admin_pyro(client, message.chat.id, message.from_user.id):
        return await message.reply_text("🚫 Only admins can use this command.")
    pic_enabled[message.chat.id] = False
    await message.reply_text("❌ Profile picture sending is now *disabled* in this chat.")

@pyro_client.on_message(pyro_filters.command("pic") & pyro_filters.reply)
async def get_profile_pics(client, message):
    global sending_pics
    if not pic_enabled.get(message.chat.id, True):
        return
    if sending_pics:
        await message.reply_text("♦️ You can't use this command now cause I am already sending profile pictures of a user!")
        return
    sending_pics = True
    try:
        user = message.reply_to_message.from_user
        photos = []
        async for photo in client.get_chat_photos(user.id):
            photos.append(photo.file_id)
        if not photos:
            await message.reply_text("🚫 No profile photos found for this user.")
            sending_pics = False
            return
        for i in range(0, len(photos), 10):
            media_group = [InputMediaPhoto(file_id) for file_id in photos[i:i+10]]
            await client.send_media_group(chat_id=message.chat.id, media=media_group)
    except Exception as e:
        await message.reply_text(f"⚠️ Error: {str(e)}")
    finally:
        sending_pics = False



def fetch_anime(search: str):
    variables = {"search": search}
    response = requests.post(ANILIST_URL, json={"query": ANIME_QUERY, "variables": variables})
    data = response.json()
    if "errors" in data:
        raise ValueError("Anime not found")
    media = data["data"]["Media"]

    title = media["title"]["english"] or media["title"]["romaji"] or "Unknown"
    description = media["description"] or "No description available."
    if description:
        # Clean HTML tags simple way
        description = re.sub(r'<.*?>', '', description)
    cover = media["coverImage"]["large"]
    episodes = media.get("episodes", "N/A")
    status = media.get("status", "N/A")
    score = media.get("averageScore", "N/A")

    details = (
        f"🎬 *{title}*\n\n"
        f"📖 {description[:800]}...\n\n"
        f"🌀 Status: {status}\n"
        f"🎞 Episodes: {episodes}\n"
        f"⭐ Score: {score}"
    )
    return cover, details

def fetch_character(search: str):
    variables = {"search": search}
    response = requests.post(ANILIST_URL, json={"query": CHARACTER_QUERY, "variables": variables})
    data = response.json()
    if "errors" in data:
        raise ValueError("Character not found")
    char = data["data"]["Character"]

    name = char["name"]["full"]
    native = char["name"]["native"] or ""
    description = char["description"] or "No description available."
    if description:
        description = re.sub(r'<.*?>', '', description)
    image = char["image"]["large"]

    details = f"👤 *{name}* ({native})\n\n📖 {description[:800]}..."
    return image, details





def clean_text(text: str) -> str:
    """Remove any HTML tags from Google snippets/titles."""
    text = re.sub(r"<.*?>", "", text)   # remove tags like <b>, <em>, etc.
    return html.escape(text)            # escape any special chars



def clean_text(text: str) -> str:
    """Remove any HTML tags from Google results safely."""
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)   # remove <b>, <em>, etc.
    return html.escape(text)            # escape &, <, >, quotes

def google_search(query: str, current_time: str, current_date: str) -> str:
    try:
        GOOGLE_API_KEY = "AIzaSyD9vtvK6rVkTvtY4HSsD-T5brO2ltusSAI"
        GOOGLE_CX = "55847e577a2004512"

        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
        )
        res = requests.get(url, timeout=15)
        data = res.json()

        if "items" not in data:
            raise Exception("Google quota exceeded or no results")

        items = data["items"][:2]

        formatted = (
            f"You are Dikshika ᥫ᭡, a polite and helpful assistant.\n"
            f"You know the current time ({current_time}) and date ({current_date}).\n"
            f"Use structured formatting with ✘ headings and • bullet points in your response.\n"
            f"Keep answers clear, concise, and helpful.\n\n"
            f"Query: {query}\n\n"
        )

        for item in items:
            title = clean_text(item.get("title", ""))
            snippet = clean_text(item.get("snippet", ""))
            formatted += f"✘ <b>{title}</b>\n• {snippet}\n\n"

        return formatted.strip()

    except Exception as e:
        raise RuntimeError(f"Google error: {str(e)}")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args and not update.message.reply_to_message:
        await update.message.reply_text("Please provide a question or reply to a message and use /ask.")
        return

    user_input = " ".join(context.args) if context.args else (
        update.message.reply_to_message.text if update.message.reply_to_message and update.message.reply_to_message.text else ""
    )
    if not user_input:
        await update.message.reply_text("Replied message has no text.")
        return



    # Build structured prompt
    query_prompt = (
        f"You are Dikshika ᥫ᭡, a polite and structured assistant.\n"
        f"Provide a concise, accurate response with:\n"
        f"✘ Bold section headings\n"
        f"• Bullet points\n"
        f"• Clear spacing\n"
        f"• Highlight **keywords** in bold\n\n"
        f"Query: {user_input}"
    )

    try:
        # First try Google Search
        result = google_search(user_input)
    except Exception:
        # Fallback to Gemini
        try:
            response = gemini_model.generate_content(query_prompt)
            result = format_response(response.text)
        except Exception as e:
            result = f"⚠️ Gemini API Error: {e}"

    # Send response (handle Telegram message length limits)
    max_len = 4000
    reply_to_id = update.message.message_id if not update.message.reply_to_message else update.message.reply_to_message.message_id

    if len(result) > max_len:
        for i in range(0, len(result), max_len):
            await update.message.reply_text(
                result[i:i+max_len],
                parse_mode="HTML",
                disable_web_page_preview=True,
                reply_to_message_id=reply_to_id)
    else:
        await update.message.reply_text(
            result,
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_to_message_id=reply_to_id)





async def anime_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /anime <anime name>")
        return

    query = " ".join(context.args)
    variables = {"search": query}
    response = requests.post(ANILIST_URL, json={"query": ANIME_QUERY, "variables": variables}).json()

    if "errors" in response:
        await update.message.reply_text("❌ Anime not found.")
        return

    anime = response["data"]["Media"]

    # Format fields
    title = anime["title"]["romaji"] or anime["title"]["english"] or "Unknown"
    english = anime["title"]["english"] or title
    mal_id = anime.get("idMal", "N/A")
    source = anime.get("source", "N/A")
    atype = anime.get("type", "N/A")
    score = f"{anime['averageScore']}% 🌟" if anime.get("averageScore") else "N/A"
    duration = f"{anime['duration']} min/ep" if anime.get("duration") else "N/A"
    status = anime.get("status", "N/A")
    episodes = anime.get("episodes", "N/A")
    if status == "FINISHED" and episodes != "N/A":
        status = f"{status} | {episodes} eps"


    # Next airing info
    next_airing = "N/A"
    if anime.get("nextAiringEpisode"):
        airing_time = anime["nextAiringEpisode"]["airingAt"] - int(time.time())
        days, rem = divmod(airing_time, 86400)
        hrs, rem = divmod(rem, 3600)
        mins, secs = divmod(rem, 60)
        next_airing = f"{days} Days, {hrs} Hours, {mins} Minutes, {secs} Seconds | {anime['nextAiringEpisode']['episode']} eps"

            # Limit genres to top 5
    genres = ", ".join(anime.get("genres", [])[:5]) or "N/A"

    # Limit tags to top 8
    tags = ", ".join(tag["name"] for tag in anime.get("tags", [])[:8]) or "N/A"

    # Trailer
    trailer = "N/A"
    if anime.get("trailer"):
        if anime["trailer"]["site"] == "youtube":
            trailer = f"https://youtu.be/{anime['trailer']['id']}"
        else:
            trailer = f"{anime['trailer']['site']}/{anime['trailer']['id']}"

    # Caption text
    caption = (
        f"[🇯🇵]<b>{title}</b> | {english}\n"
        f"<b>ID</b> | <b>MAL ID</b>: <code>{anime['id']}</code> | <code>{mal_id}</code>\n"
        f"➤ <b>SOURCE:</b> <code>{source}</code>\n"
        f"➤ <b>TYPE:</b> <code>{atype}</code>\n"
        f"➤ <b>SCORE:</b> <code>{score}</code>\n"
        f"➤ <b>DURATION:</b> <code>{duration}</code>\n"
        f"➤ <b>STATUS:</b> <code>{status}</code>\n"
        f"➤ <b>NEXT AIRING:</b> <code>{next_airing}</code>\n"
        f"➤ <b>GENRES:</b> <code>{genres}</code>\n"
        f"➤ <b>TAGS:</b> <code>{tags}</code>\n"
        f'🎬 <a href="{trailer}">Trailer</a>\n'
        f"📖 <a href='{anime['siteUrl']}'>Official Site</a>"
    )


    # Ensure caption under 1024 chars
    if len(caption) > 1024:
        caption = caption[:1000] + "..."


    # Inline buttons
    keyboard = [
        [InlineKeyboardButton("🎭 Characters", callback_data=f"anime_chars:{anime['id']}")],
        [InlineKeyboardButton("📖 Description", callback_data=f"anime_desc:{anime['id']}")],
        [InlineKeyboardButton("📺 List Series", callback_data=f"anime_series:{anime['id']}")]
    ]

    await update.message.reply_photo(
        photo=anime["coverImage"]["large"],
        caption=caption,
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )




async def add_filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to use filter commands.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("Only admins can add filters.")
        return

    if len(context.args) < 1:
        await update.message.reply_text("Usage: Reply to a message with /filter keyword")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("You must reply to a message to add a filter.")
        return

    keyword = context.args[0].lower()
    chat_id = str(update.effective_chat.id)

    if chat_id not in filters_db:
        filters_db[chat_id] = {}

    # Save by message_id the replied-to message
    filters_db[chat_id][keyword] = update.message.reply_to_message.message_id

    save_filters()
    await update.message.reply_text(f"Filter added for '{keyword}'.")




async def unfilter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to use filter commands.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be admin to remove filters.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /unfilter keyword")
        return

    keyword = context.args[0].lower()
    chat_id = str(update.effective_chat.id)

    if chat_id in filters_db and keyword in filters_db[chat_id]:
        del filters_db[chat_id][keyword]
        save_filters()
        await update.message.reply_text(f"Removed filter '{keyword}'.")
    else:
        await update.message.reply_text(f"No filter found for '{keyword}'.")





async def list_filters_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if chat_id not in filters_db or not filters_db[chat_id]:
        await update.message.reply_text("No filters added.")
        return

    filter_list = "\n".join(filters_db[chat_id].keys())
    await update.message.reply_text("Filters:\n" + filter_list)













def boldify(text: str) -> str:
    """Convert Markdown-style **bold** to HTML <b>bold</b>"""
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)




async def set_blacklist_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if not await is_admin(update, user_id):
        return await update.message.reply_text("You must be an admin to set blacklist mode.")

    if not context.args:
        return await update.message.reply_text("Usage: /setblacklistmode <mode> [duration if tmute]")

    mode = context.args[0].lower()

    valid_modes = ["warn", "delete", "kick", "ban", "mute", "tmute"]
    if mode not in valid_modes:
        return await update.message.reply_text(
            f"Invalid mode. Choose one of: {', '.join(valid_modes)}"
        )

    if mode == "tmute":
        # default duration if not provided
        duration_str = context.args[1] if len(context.args) > 1 else "30m"
        seconds = parse_time_to_seconds(duration_str)
        if seconds <= 0:
            return await update.message.reply_text("Invalid duration format. Example: 5s, 10m, 2h, 1d")

        blacklist_modes[chat_id] = (mode, duration_str)
        return await update.message.reply_text(
            f"✅ Blacklist mode set to *Temporary Mute* for {duration_str}.",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        blacklist_modes[chat_id] = mode
        return await update.message.reply_text(f"✅ Blacklist mode set to *{mode.title()}*.", parse_mode=ParseMode.MARKDOWN)







async def is_admin(update: Update, user_id: int) -> bool:
    try:
        member = await update.effective_chat.get_member(user_id)
        return member.status in ("administrator", "creator")
    except Exception:
        return False
    

async def is_member_admin(chat, user_id):
    try:
        member = await chat.get_member(user_id)
        return member.status in ("administrator", "creator", "owner")
    except Exception:
        return False

    
async def setflood(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to configure flood settings.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("Only admins can set flood limit.")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /setflood <number>")
        return
    flood_limit = int(context.args[0])

    flood_settings[update.effective_chat.id] = flood_limit
    await update.message.reply_text(f"Flood limit set to {flood_limit} messages.")



async def setfloodmode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to configure flood settings.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("Only admins can set flood mode.")
        return
    if not context.args:
        await update.message.reply_text(
            "Usage: /setfloodmode <tmute|mute|ban|kick> [duration]\nExample: /setfloodmode tmute 7m"
        )
        return
    action = context.args[0].lower()
    duration = context.args[1] if len(context.args) > 1 else "30m"
    # Only 'tmute' needs a duration
    if action not in ("tmute", "mute", "ban", "kick"):
        await update.message.reply_text("Action must be one of: tmute, mute, ban, kick")
        return
    flood_modes[update.effective_chat.id] = (action, duration)
    await update.message.reply_text(
        f"Flood mode set: {action}" + (f" {duration}" if action == "tmute" else "")
    )


async def take_flood_action(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id, action, duration="30m"):
    chat_id = update.effective_chat.id
    user = update.effective_user if update.effective_user.id == user_id else None
    name = format_name(user) if user else str(user_id)
    now = int(time.time())

    # --- Prevent duplicate actions ---
    if action == "tmute" and user_id in temp_mutes:
        mute_chat, unmute_time = temp_mutes[user_id]
        if mute_chat == chat_id and now < unmute_time:
            return

    if action == "kick":
        await update.effective_chat.ban_member(user_id)
        await update.effective_chat.unban_member(user_id)
        await context.bot.send_message(chat_id, f"👢 Kicked {name} for flooding.")
        return

    if action == "ban":
        await update.effective_chat.ban_member(user_id)
        await context.bot.send_message(chat_id, f"⛔ Banned {name} for flooding.")
        return

    if action == "mute":
        await update.effective_chat.restrict_member(user_id, permissions=PTBChatPermissions(can_send_messages=False))
        await context.bot.send_message(chat_id, f"🔇 Muted {name} for flooding.")
        return

    if action == "tmute":
        seconds = parse_time_to_seconds(duration)
        until = now + seconds
        await update.effective_chat.restrict_member(
            user_id,
            permissions=PTBChatPermissions(can_send_messages=False),
            until_date=until
        )
        temp_mutes[user_id] = (chat_id, until)
        await context.bot.send_message(chat_id, f"⏳ Temporarily muted {name} for flooding ({duration}).")
        return


async def get_all_members(chat_id):
    entity = await tclient.get_entity(chat_id)
    participants = await tclient.get_participants(entity)
    return participants




async def resolve_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Resolve user from reply, user_id or global @username lookup using Bot API and Telethon
    if update.message.reply_to_message:
        return update.message.reply_to_message.from_user

    if not context.args:
        await update.message.reply_text("Please reply to a user or provide @username or user ID.")
        return None

    username_or_id = context.args[0].lstrip("@")
    chat = update.effective_chat

    # Try Bot API first
    try:
        if username_or_id.isdigit():
            return await context.bot.get_chat(int(username_or_id))
        else:
            return await context.bot.get_chat(username_or_id)
    except Exception:
        pass  # fallback to Telethon lookup

    # Telethon global user lookup
    try:
        user = await tclient.get_entity(username_or_id)
        return user
    except Exception as e:
        await update.message.reply_text(f"Could not fetch user @{username_or_id}: {e}")
        return None

def format_name(user):
    name = f"{getattr(user, 'first_name', '') or ''} {getattr(user, 'last_name', '') or ''}".strip()
    return name or "Unknown"

# === Command Handlers ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    REGISTERED_USERS.add(user.id)  # mark user as registered
    args = context.args

    # === Captcha deep-link ===
    if args and args[0].startswith("cap_"):
        try:
            _, cid, uid = args[0].split("_", 2)
            chat_id = int(cid)
            user_id = int(uid)
        except Exception:
            return await update.message.reply_text("⚠️ Invalid captcha link.")

        if user.id != user_id:
            return await update.message.reply_text("⚠️ This captcha isn’t for you.")

        # Generate captcha question
        q, ans = _make_captcha()
        captcha_state[(chat_id, user_id)] = {"answer": ans, "tries": 0, "question": q}

        kb = _build_9_options_keyboard(chat_id, user_id, ans)
        return await update.message.reply_text(
            f"Solve to unmute:\n<b>{q} = ?</b>\n\nYou have 3 tries.",
            parse_mode=ParseMode.HTML,
            reply_markup=kb
        )

    # === Kang registration (your old logic) ===
    if args and args[0] == "kang":
        await update.message.reply_text(
            "✅ You’re registered! Now go back and use /kang again."
        )
    else:
        await update.message.reply_text(
            "👋 Hi! I’m your group management bot.\n\n"
            "Add me in your group as admin and enjoy the wholesome sets of useful codes."
        )



async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    commands = [
        "/info - Show user info ",
        "/warn - Warn user, ban after 3 warns ",
        "/warns- shows total number of warns a user have"
        "/del - Delete message (reply)",
        "/ban - Ban user ",
        "/unban - Unban user by ID",
        "/admins - List admins",
        "/promote - Promote user to admin ",
        "/demote - Demote admin ",
        "/addblacklist - Add blacklist word",
        "/unblacklist - Remove blacklist word",
        "/blacklist - Show blacklist",
        "/unblacklistall- deletes all blacklist words"
        "/approve - Approve user ",
        "/unapprove - Unapprove user ",
        "/approved- shows list of all approved users",
        "/unapproveall- unapproves all approved users",
        "/purge - Delete last N messages",
        "/filter - Add filter",
        "/filters - List filters",
        "/unfilter - Remove filter",
        "/afk - Set AFK status",
        "/mute - Mute user ",
        "/unmute - Unmute user ",
        "/id - Show user or chat ID",
        "/kick - Kick user ",
        "/tmute - Temporarily mute user ",
        "/kickme - Kick yourself",
        "/waifu - Get you partner, approved by astro ",
        "/tagall - Tag all members",
        "/pin - Pin message (reply)",
        "/unpin - Unpin all messages",
        "/setflood - can restrict user from spamming",
        "/setfloodmode - action taken if user floods the chat",
        "/q - can quote texts",
        "/blacklistmode - tmute/mute/ban/warn/kick/delete ",
        "/lock - lock any type",
        "/unlock - unlock any type",
        "/locks - list of all locks present",
        "/kang- save any sticker/gif in your pack",
        "/zombies- Show the number of deleted accounts present",
        "/rzombies- Remove all the deleted accpunts present",
        "/tr- translate text to your desired language",
        "/getsticker- gives sticker in png form with sticker id",
        "/pp- search for the photo and gives description about it",
        "/calc- calculates anything",
        "/report- report to admins",
        "/nightmode- deletes non-text messages of unapproved and non admin users",
        "/ud- tells meaning of the wprd",
        "/rmwarn- remove one warning of user",
        "/resetwarns- removes all warning of user",
        "/welcome on/of- turn welcome messages on/off",
        "/setwelcome- set a welcome message for new users",
        "/goodbye- turn goodbye message on/off",
        "/setgoodbye- set a goodbye message who lefts the group",
        "/when- shows when the message was sent",
        "/captcha- force user to solve captcha before interacting in group when he/she joins",
        "/pic - shows all pfp of a user",
        "/free- free a user and allow to send stickers",
        "/unfree- restrict a user from sending stickers",
        "/freelist- shows all freed user",
        "/character- get a overview of any anime character",
        "/unfilterall- removes all filters",
        "/zombies- show total deleted accounts in group",
        "/rzombies- remove all deleted accounts",
        "/editdelete - deletes all edited message after 30s"

    ]
    
    await update.message.reply_text("Available commands:\n" + "\n".join(commands))


def escape_md(text: str) -> str:
    """
    Escapes all Telegram MarkdownV2 special characters in the given text.
    """
    escape_chars = r'[_*[\]()~`>#+\-=|{}.!]'
    return re.sub(escape_chars, r'\\\g<0>', text)


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    chat = update.effective_chat

    # Resolve user to show info for
    if not context.args and not update.message.reply_to_message:
        user = update.effective_user
    else:
        user = await resolve_user(update, context)

    if user is None:
        await update.message.reply_text("Could not find the user.")
        return

    user_id = user.id

    # Get user's presence status in the group
    presence = "Member"
    if chat.type in ["group", "supergroup"]:
        try:
            member = await chat.get_member(user_id)
            if member.status in ("creator", "owner"):
                presence = "Owner"
            elif member.status == "administrator":
                presence = "Admin"
            else:
                presence = "Member"
        except Exception:
            presence = "Unknown"

    # Get user bio if any
    bio = "No bio available."
    try:
        full_user = await bot.get_chat(user_id)
        if full_user.bio:
            bio = full_user.bio
    except Exception:
        pass

    # Get total profile photos count
    user = await bot.get_chat(user_id)
    profile_photos = await bot.get_user_profile_photos(user.id)

    total_photos = profile_photos.total_count if profile_photos else 0

    # Approved status
    approved = "Yes" if user_id in approved_users else "No"

    # AFK status
    afk_status = "User is currently afk!" if user_id in afk_users else "User is not afk!"

    # Banned status
    banned = "No"
    try:
        member_status = await chat.get_member(user_id)
        if member_status.status == "kicked":
            banned = "Yes"
    except Exception:
        banned = "Yes"  # Possibly banned or not present

    # Muted status
    muted = "No"
    try:
        member_status = await chat.get_member(user_id)
        if member_status.can_send_messages is False:
            muted = "Yes"
    except Exception:
        muted = "No"

    # Escape all user-generated texts for MarkdownV2
    first_name_escaped = escape_md(user.first_name or "")
    last_name_escaped = escape_md(user.last_name or "")
    username_escaped = escape_md(user.username) if user.username else "N/A"
    presence_escaped = escape_md(presence)
    bio_escaped = escape_md(bio)
    approved_escaped = escape_md(approved)
    afk_escaped = escape_md(afk_status)
    banned_escaped = escape_md(banned)
    muted_escaped = escape_md(muted)

    # Escape mention name for clickable mention
    mention = f"[{escape_md(user.first_name or 'User')}](tg://user?id={user_id})"

    # Build message text in requested format
    message_text = (
        "⌬ ᴜsᴇʀ ɪɴғᴏʀᴍᴀᴛɪᴏɴ ⌬\n"
        "•────────[×]────────•\n\n"
        f"✧ ᴜꜱᴇʀɪᴅ : {user_id}\n"
        f"✧ ɴᴀᴍᴇ : {first_name_escaped} {last_name_escaped}\n"
        f"✧ ᴜꜱᴇʀɴᴀᴍᴇ : @{username_escaped}\n"
        f"✧ ᴘʀᴇꜱᴇɴᴄᴇ : {presence_escaped}\n"
        f"✧ ᴍᴇɴᴛɪᴏɴ : {mention}\n"
        f"✧ ʙɪᴏ : {bio_escaped}\n"
        f"✧ ᴘʀᴏꜰɪʟᴇ ᴘʜᴏᴛᴏꜱ : {total_photos}\n\n"
        "•────────[×]────────•\n\n"
        f"✧ ᴀᴘᴘʀᴏᴠᴇᴅ : {approved_escaped}\n"
        f"✧ ᴀꜰᴋ ꜱᴛᴀᴛᴜꜱ : {afk_escaped}\n"
        f"✧ ʙᴀɴɴᴇᴅ ʜᴇʀᴇ : {banned_escaped}\n"
        f"✧ ᴍᴜᴛᴇᴅ : {muted_escaped}\n\n"
        "•────────[×]────────•"
    )

    # Prepare to send profile photo if available
    photo_file = None
    if total_photos > 0:
        try:
            photo = profile_photos.photos[0][-1]  # highest res photo
            file = await bot.get_file(photo.file_id)
            photo_bytearray = await file.download_as_bytearray()
            photo_file = BytesIO(photo_bytearray)
            photo_file.name = "profile.jpg"
        except Exception:
            photo_file = None

    # Send photo with caption if photo available, else send text message only
    if photo_file:
        await bot.send_photo(
            chat_id=chat.id,
            photo=photo_file,
            caption=message_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_notification=True,
        )
    else:
        await update.message.reply_text(
            text=message_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
            disable_notification=True,
        )




async def warn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("You must be an admin to warn users.")

    user = await resolve_user(update, context)
    if not user:
        return

    # don't allow warning admins
    if await is_member_admin(update.effective_chat, user.id):
        return await update.message.reply_text("I cannot warn an admin!")

    # extract reason
    reason = " ".join(context.args[1:]) if context.args else ""
    if update.message.text and update.message.text.strip() != "/warn":
        parts = update.message.text.split(maxsplit=2)
        if len(parts) >= 3:
            reason = parts[2]
    if not reason:
        reason = "No reason given"

    # update warnings
    count = warnings.get(user.id, 0) + 1
    warnings[user.id] = count
    warn_reasons.setdefault(user.id, []).append(reason)

    if count < 3:
        await update.message.reply_text(
            f"<a href='tg://user?id={user.id}'>{format_name(user)}</a> warned: {count}/3 warnings.\nReason: {reason}",
            parse_mode="HTML"
        )
    else:
        try:
            await update.effective_chat.ban_member(user.id)
            await update.message.reply_text(
                f"<a href='tg://user?id={user.id}'>{format_name(user)}</a> has been banned after 3 warnings.",
                parse_mode="HTML"
            )
            warnings[user.id] = 0
            warn_reasons[user.id] = []
        except BadRequest as e:
            await update.message.reply_text(f"Couldn't ban user: {e}")


async def resetwarns(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("Only admins can reset warnings.")
        return

    user = None
    # Get user from reply or mention
    if update.message.reply_to_message:
        user = update.message.reply_to_message.from_user
    elif context.args:
        user = await resolve_user(update, context)
    if not user:
        await update.message.reply_text("Please reply or mention a user to reset warns.")
        return

    if await is_member_admin(update.effective_chat, user.id):
        await update.message.reply_text("Cannot reset warns for admins.")
        return

    warnings[user.id] = 0
    await update.message.reply_text(f"All warnings for {format_name(user)} have been reset.")


async def rmwarn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("Only admins can remove warnings.")
        return

    user = None
    # Get user from reply or mention
    if update.message.reply_to_message:
        user = update.message.reply_to_message.from_user
    elif context.args:
        user = await resolve_user(update, context)
    if not user:
        await update.message.reply_text("Please reply or mention a user to remove a warn.")
        return

    if await is_member_admin(update.effective_chat, user.id):
        await update.message.reply_text("Cannot remove warns for admins.")
        return

    count = warnings.get(user.id, 0)
    if count > 0:
        warnings[user.id] = count - 1
        await update.message.reply_text(f"One warning removed from {format_name(user)}. \nTotal warns: {warnings[user.id]}/3.")
    else:
        await update.message.reply_text(f"{format_name(user)} has no warnings.")



async def delmsg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_delete_messages"]):
        return await update.message.reply_text("🚫 You need 'Can Delete Messages' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to delete messages.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to the message to delete it.")
        return
    try:
        await update.message.reply_to_message.delete()
        await update.message.reply_text("Message deleted.")
    except BadRequest as e:
        await update.message.reply_text(f"Error: {e}")

async def ban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to ban users.")
        return
    user = await resolve_user(update, context)
    if not user:
        return
    if await is_member_admin(update.effective_chat, user.id):
        await update.message.reply_text("I cannot ban an admin!")
        return

    try:
        await update.effective_chat.ban_member(user.id)
        await update.message.reply_text(f"{format_name(user)} has been banned.")
    except BadRequest as e:
        await update.message.reply_text(f"Error: {e}")

async def unban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be admin to unban users.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /unban <user_id or @username>")
        return

    # Attempt to resolve user from username or user_id
    username_or_id = context.args[0]
    user = None

    # Remove @ if present
    username_or_id = username_or_id.lstrip('@')

    # Try to resolve user using bot API or Telethon
    try:
        user = await resolve_user(update, context)
    except Exception as e:
        await update.message.reply_text(f"Failed to resolve user: {e}")
        return

    if not user:
        await update.message.reply_text("Could not find the user. Please try unban with user ID or reply to user's message.")
        return

    try:
        await update.effective_chat.unban_member(user.id)
        await update.message.reply_text(f"Unbanned user {getattr(user, 'first_name', user.id)}.")
    except Exception as e:
        await update.message.reply_text(f"Failed to unban user: {e}")


async def admins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admins = await update.effective_chat.get_administrators()
    text = "Group admins:\n"
    for admin in admins:
        user = admin.user
        text += f"- {format_name(user)} (@{user.username})\n"
    await update.message.reply_text(text)
async def promote(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    bot = context.bot

    # Command issuer
    admin = await bot.get_chat_member(chat.id, update.effective_user.id)

    # Target user
    target_user = await resolve_user(update, context)
    if not target_user:
        await update.message.reply_text("⚠️ Please reply to a user or mention them to promote/demote.")
        return
    user_id = target_user.id

    if not user_id:
        await update.message.reply_text("⚠️ Please reply to a user or mention them to promote.")
        return

    # Self-promote prevention
    if user_id == update.effective_user.id:
        await update.message.reply_text("You can't promote yourself!")
        return

    # Permission check
    can_promote = False
    if admin.status == "creator":
        can_promote = True
    elif hasattr(admin, "can_promote_members"):
        can_promote = admin.can_promote_members
    elif hasattr(admin, "privileges"):
        can_promote = getattr(admin.privileges, "can_promote_members", False)

    if not can_promote:
        await update.message.reply_text("You haven't enough rights to perform this action.")
        return

    # Bot itself
    bot_member = await bot.get_chat_member(chat.id, bot.id)
    bot_can_promote = False
    if bot_member.status == "creator":
        bot_can_promote = True
    elif hasattr(bot_member, "can_promote_members"):
        bot_can_promote = bot_member.can_promote_members
    elif hasattr(bot_member, "privileges"):
        bot_can_promote = getattr(bot_member.privileges, "can_promote_members", False)

    if not bot_can_promote:
        await update.message.reply_text("⚠️ I don’t have permission to promote members.")
        return

    # Target user info
    target = await bot.get_chat_member(chat.id, user_id)

    # Already admin?
    if target.status in ["administrator", "creator"]:
        await update.message.reply_text(
            f"<a href='tg://user?id={user_id}'>{target.user.first_name}</a> is already an admin.",
            parse_mode="HTML"
        )
        return

    # Custom admin title (fix: exclude first arg if it's mention/ID)
    title = "Admin"
    if context.args:
        args = context.args
        if args[0].startswith("@") or args[0].isdigit():
            args = args[1:]
        if args:
            title = " ".join(args)

    try:
        await bot.promote_chat_member(
            chat.id,
            user_id,
            can_change_info=True,
            can_delete_messages=True,
            can_invite_users=True,
            can_restrict_members=True,
            can_pin_messages=True,
            can_promote_members=False,
        )
        # Set custom title if supported
        if title:
            try:
                await bot.set_chat_administrator_custom_title(chat.id, user_id, title)
            except Exception:
                pass

        await update.message.reply_text(
            f"✅ Promoted <a href='tg://user?id={user_id}'>{target.user.first_name}</a> as admin.",
            parse_mode="HTML"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to promote: {e}")



async def demote(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    bot = context.bot

    # Command issuer
    admin = await bot.get_chat_member(chat.id, update.effective_user.id)

    # Target user
    target_user = await resolve_user(update, context)
    if not target_user:
        await update.message.reply_text("⚠️ Please reply to a user or mention them to promote/demote.")
        return
    user_id = target_user.id

    if not user_id:
        await update.message.reply_text("⚠️ Please reply to a user or mention them to demote.")
        return

    # Self-demote prevention
    if user_id == update.effective_user.id:
        await update.message.reply_text("You can't demote yourself!")
        return

    # Permission check
    can_promote = False
    if admin.status == "creator":
        can_promote = True
    elif hasattr(admin, "can_promote_members"):
        can_promote = admin.can_promote_members
    elif hasattr(admin, "privileges"):
        can_promote = getattr(admin.privileges, "can_promote_members", False)

    if not can_promote:
        await update.message.reply_text("You haven't enough rights to perform this action.")
        return

    # Bot itself
    bot_member = await bot.get_chat_member(chat.id, bot.id)
    bot_can_promote = False
    if bot_member.status == "creator":
        bot_can_promote = True
    elif hasattr(bot_member, "can_promote_members"):
        bot_can_promote = bot_member.can_promote_members
    elif hasattr(bot_member, "privileges"):
        bot_can_promote = getattr(bot_member.privileges, "can_promote_members", False)

    if not bot_can_promote:
        await update.message.reply_text("⚠️ I don’t have permission to demote members.")
        return

    # Target is a bot?
    target = await bot.get_chat_member(chat.id, user_id)
    if target.user.is_bot:
        await update.message.reply_text(
            "Due to telegram limitations I can't demote bots. Demote them manually!"
        )
        return

    # Try to demote
    try:
        await bot.promote_chat_member(
            chat.id,
            user_id,
            can_change_info=False,
            can_delete_messages=False,
            can_invite_users=False,
            can_restrict_members=False,
            can_pin_messages=False,
            can_promote_members=False
        )
        await update.message.reply_text("✅ User demoted successfully.")
    except Exception:
        await update.message.reply_text(
            "Error while demote: maybe they aren't promoted by me."
        )



import uuid

async def addblacklist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info", "can_delete_messages"]):
        return await update.message.reply_text("🚫 You need both 'Can Change Info' and 'Can Delete Messages' permissions.")

    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("You must be an admin to add blacklist entries.")

    msg = update.message

    # --- Case 1: Admin replied to a sticker ---
    if msg.reply_to_message and msg.reply_to_message.sticker:
        sticker = msg.reply_to_message.sticker

        # generate short IDs
        sid = str(uuid.uuid4())[:8]
        pid = str(uuid.uuid4())[:8]

        sticker_map[sid] = sticker.file_id
        pack_map[pid] = sticker.set_name or "single"

        buttons = [
            [
                InlineKeyboardButton("➕ Add this sticker", callback_data=f"bl_add_sticker:{sid}"),
                InlineKeyboardButton("➕ Add whole pack", callback_data=f"bl_add_pack:{pid}"),
            ]
        ]
        return await msg.reply_text("Choose how to blacklist:", reply_markup=InlineKeyboardMarkup(buttons))

    # --- Case 2: Admin provided a word (existing logic) ---
    if not context.args:
        return await msg.reply_text("Usage: /addblacklist <word> or reply with a sticker")

    word = context.args[0].lower()
    blacklist.add(word)
    await msg.reply_text(f"✅ Added word '{word}' to blacklist.")



async def unblacklist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info", "can_delete_messages"]):
        return await update.message.reply_text("🚫 You need both 'Can Change Info' and 'Can Delete Messages' permissions.")

    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("You must be an admin to remove blacklist entries.")

    msg = update.message

    # --- Case 1: Admin replied to a sticker ---
    if msg.reply_to_message and msg.reply_to_message.sticker:
        sticker = msg.reply_to_message.sticker

        # generate short IDs
        sid = str(uuid.uuid4())[:8]
        pid = str(uuid.uuid4())[:8]

        sticker_map[sid] = sticker.file_id
        pack_map[pid] = sticker.set_name or "single"

        buttons = [
            [
                InlineKeyboardButton("❌ Remove this sticker", callback_data=f"bl_remove_sticker:{sid}"),
                InlineKeyboardButton("❌ Remove whole pack", callback_data=f"bl_remove_pack:{pid}"),
            ]
        ]
        return await msg.reply_text("Choose how to unblacklist:", reply_markup=InlineKeyboardMarkup(buttons))

    # --- Case 2: Admin provided a word (existing logic) ---
    if not context.args:
        return await msg.reply_text("Usage: /unblacklist <word> or reply with a sticker")

    word = context.args[0].lower()
    blacklist.discard(word)
    await msg.reply_text(f"✅ Removed word '{word}' from blacklist.")




async def blacklist_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id

    action, sid = query.data.split(":", 1)

    if action == "bl_add_sticker":
        real_id = sticker_map.get(sid)
        if real_id:
            sticker_blacklist.setdefault(chat_id, set()).add(real_id)
        await query.edit_message_text("✅ Sticker blacklisted.")

    elif action == "bl_add_pack":
        real_pack = pack_map.get(sid)
        if real_pack:
            pack_blacklist.setdefault(chat_id, set()).add(real_pack)
        await query.edit_message_text("✅ Whole pack blacklisted.")

    elif action == "bl_remove_sticker":
        real_id = sticker_map.get(sid)
        if real_id:
            sticker_blacklist.get(chat_id, set()).discard(real_id)
        await query.edit_message_text("❌ Sticker removed from blacklist.")

    elif action == "bl_remove_pack":
        real_pack = pack_map.get(sid)
        if real_pack:
            pack_blacklist.get(chat_id, set()).discard(real_pack)
        await query.edit_message_text("❌ Pack removed from blacklist.")


async def showblacklist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info", "can_delete_messages"]):
        return await update.message.reply_text("🚫 You need both 'Can Change Info' and 'Can Delete Messages' permissions.")

    if blacklist:
        await update.message.reply_text("Blacklist:\n" + ", ".join(blacklist))
    else:
        await update.message.reply_text("Blacklist is empty.")

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to approve users.")
        return

    user = await resolve_user(update, context)
    if not user:
        return

    if await is_member_admin(update.effective_chat, user.id):
        mention = f"[{escape_md(user.first_name or 'User')}](tg://user?id={user.id})"
        text = " is already admin, they can't be approved."
        await update.message.reply_text(
            f"{mention}{escape_md(text)}", 
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    chat_id = update.effective_chat.id
    approved_users.setdefault(chat_id, set()).add(user.id)   # ✅ FIXED

    text = f"{format_name(user)} approved."
    await update.message.reply_text(
        escape_md(text),
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def unapprove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to unapprove users.")
        return

    user = await resolve_user(update, context)
    if not user:
        return

    if await is_member_admin(update.effective_chat, user.id):
        mention = f"[{escape_md(user.first_name or 'User')}](tg://user?id={user.id})"
        text = " is an admin, they can't be unapproved."
        await update.message.reply_text(
            f"{mention}{escape_md(text)}", 
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    chat_id = update.effective_chat.id
    approved_users.setdefault(chat_id, set()).discard(user.id)   # ✅ FIXED

    text = f"{format_name(user)} unapproved."
    await update.message.reply_text(
        escape_md(text),
        parse_mode=ParseMode.MARKDOWN_V2
    )










async def filter_trigger_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if message is None or message.text is None:
        return

    chat_id = str(message.chat.id)
    text = message.text.lower()

    if chat_id in filters_db:
        if text in filters_db[chat_id]:
            saved_msg_id = filters_db[chat_id][text]
            try:
                # Try replying to the user's message
                await context.bot.copy_message(
                    chat_id=message.chat.id,
                    from_chat_id=message.chat.id,
                    message_id=saved_msg_id,
                    reply_to_message_id=message.message_id
                )
            except Exception:
                # Fallback: just send normally if reply fails
                await context.bot.copy_message(
                    chat_id=message.chat.id,
                    from_chat_id=message.chat.id,
                    message_id=saved_msg_id,
                )

async def unfilterall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    message = update.effective_message

    # Check if the user is chat creator
    member = await chat.get_member(user.id)
    if member.status != "creator":
        return await message.reply_text("🚫 Only the group owner can use this command.")

    chat_id = str(chat.id)

    if chat_id not in filters_db or not filters_db[chat_id]:
        return await message.reply_text("ℹ️ No filters are set in this chat.")

    # Clear all filters for this chat
    filters_db[chat_id].clear()

    return await message.reply_text("✅ All filters have been removed.")



async def has_permission(update: Update, permissions: list[str]) -> bool:
    """
    Check if the user who issued the command has all required permissions.
    """
    try:
        chat = update.effective_chat
        user = update.effective_user
        member = await chat.get_member(user.id)

        # Group creator always has all permissions
        if member.status == "creator":
            return True

        if member.status != "administrator":
            return False

        # Now check specific admin rights
        admin_perms = member.can_manage_chat, member.can_delete_messages, member.can_restrict_members, member.can_promote_members, member.can_change_info, member.can_invite_users, member.can_pin_messages

        # Map Telegram permissions to attributes
        perm_map = {
            "can_manage_chat": member.can_manage_chat,
            "can_delete_messages": member.can_delete_messages,
            "can_restrict_members": member.can_restrict_members,
            "can_promote_members": member.can_promote_members,
            "can_change_info": member.can_change_info,
            "can_invite_users": member.can_invite_users,
            "can_pin_messages": member.can_pin_messages,
        }

        return all(perm_map.get(p, False) for p in permissions)

    except Exception as e:
        print(f"[has_permission error] {e}")
        return False


async def edited_message_handler_v2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle edited messages and schedule deletion if enabled - V2"""
    print(f"[DEBUG] Edited message handler V2 called, update type: {type(update)}")
    print(f"[DEBUG] Has edited_message: {bool(update.edited_message)}")
    
    if not update.edited_message:
        print(f"[DEBUG] No edited_message found in update")
        return

    msg = update.edited_message

    # 🚫 Skip whispers/system edits: if message has inline buttons, skip
    if msg.reply_markup and msg.reply_markup.inline_keyboard:
        print(f"[DEBUG] Skipping edit with inline buttons (likely whisper) for msg {msg.message_id}")
        return

    chat = msg.chat
    chat_id = chat.id
    message_id = msg.message_id
    user = msg.from_user
    user_id = user.id if user else None
    
    print(f"[DEBUG] Edited message - Chat: {chat_id}, Message: {message_id}, User: {user_id}")
    print(f"[DEBUG] Edit-delete enabled states: {editdelete_enabled}")
    
    # Check if edit-delete is enabled in this chat
    is_enabled = editdelete_enabled.get(chat_id, False)
    print(f"[DEBUG] Edit-delete enabled for chat {chat_id}: {is_enabled}")
    
    if not is_enabled:
        print(f"[DEBUG] Edit-delete not enabled for this chat, returning")
        return
    
    # Skip if no user_id (shouldn't happen but safety check)
    if not user_id:
        print(f"[DEBUG] No user_id found, returning")
        return

    # ✅ Ignore admins and owner
    try:
        member = await chat.get_member(user_id)
        if member.status in ("administrator", "creator"):
            print(f"[DEBUG] Skipping edit deletion for admin/owner {user_id} in chat {chat_id}")
            return
    except Exception as e:
        print(f"[DEBUG] Error checking member status for {user_id}: {e}")
        return
    
    print(f"[DEBUG] Processing edited message - scheduling deletion")
    
    # Cancel any existing deletion job for this message
    existing_job = edit_message_jobs.pop((chat_id, message_id), None)
    if existing_job:
        print(f"[DEBUG] Cancelled existing job for message {message_id}")
        existing_job.schedule_removal()
    
    # Schedule new deletion job after 30 seconds
    try:
        job = context.job_queue.run_once(
            delete_edited_message,
            when=30,
            data=(chat_id, message_id),
            name=f"delete_edited_{chat_id}_{message_id}"
        )
        
        edit_message_jobs[(chat_id, message_id)] = job
        print(f"[DEBUG] Scheduled deletion job for message {message_id} in 30 seconds")
        
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⏱️ Edit detected! Message will be deleted in 30 seconds.",
                reply_to_message_id=message_id
            )
        except Exception as e:
            print(f"[DEBUG] Failed to send confirmation message: {e}")
        
    except Exception as e:
        print(f"[DEBUG] Error scheduling deletion job: {e}")


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat.id
    sender_id = message.from_user.id

    # ---- LOCK ENFORCEMENT (admins bypass most locks) ----
    locked = get_locked(chat_id)
    is_admin_user = await is_admin(update, sender_id)

    # Block /info command
    if "info" in locked and message.text and message.text.startswith("/info"):
        try:
            await message.delete()
        except Exception:
            pass
        return

    # Block pin service message (if applicable)
    if "pin" in locked and getattr(message, "pinned_message", None) is not None:
        try:
            await message.delete()
        except Exception:
            pass
        return

    # ENFORCE LOCKED TYPES FOR NON-ADMINS
    if not is_admin_user and sender_id not in approved_users.get(chat_id, set()):
        m = message
        # media locks etc...
        if "media" in locked and (getattr(m, "media", None) or getattr(m, "sticker", None) or getattr(m, "animation", None)
                                  or getattr(m, "document", None) or getattr(m, "photo", None) or getattr(m, "video", None)
                                  or getattr(m, "audio", None) or getattr(m, "voice", None)):
            try:
                await m.delete()
            except Exception:
                pass
            return

        if "messages" in locked and (m.text and not getattr(m, "media", None) and not getattr(m, "via_bot", None)):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "audio" in locked and getattr(m, "audio", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "voice" in locked and getattr(m, "voice", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "document" in locked and getattr(m, "document", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "photos" in locked and getattr(m, "photo", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "video" in locked and getattr(m, "video", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "stickers" in locked and getattr(m, "sticker", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "gif" in locked and getattr(m, "animation", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "contact" in locked and getattr(m, "contact", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "location" in locked and (getattr(m, "location", None) or getattr(m, "venue", None)):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "poll" in locked and getattr(m, "poll", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "inline" in locked and getattr(m, "via_bot", None) is not None:
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "buttons" in locked and getattr(m, "reply_markup", None) is not None:
            try:
                has_buttons = bool(getattr(m.reply_markup, "inline_keyboard", None)) or bool(getattr(m.reply_markup, "keyboard", None))
            except Exception:
                has_buttons = True
            if has_buttons:
                try:
                    await m.delete()
                except Exception:
                    pass
                return
        if "forward" in locked and (getattr(m, "forward_from", None) or getattr(m, "forward_from_chat", None)
                                    or getattr(m, "forward_sender_name", None) or getattr(m, "forward_date", None)):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "url" in locked and has_url_entities(m):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "previews" in locked and has_url_entities(m):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "rtl" in locked and (m.text or getattr(m, "caption", None)):
            text_ = m.text or getattr(m, "caption", "")
            if contains_bidi(text_):
                try:
                    await m.delete()
                except Exception:
                    pass
                return
        if "other" in locked and (getattr(m, "dice", None) or getattr(m, "game", None) or getattr(m, "invoice", None)):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "game" in locked and getattr(m, "game", None):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "egames" in locked and (getattr(m, "game", None) or getattr(m, "dice", None)):
            try:
                await m.delete()
            except Exception:
                pass
            return
        if "invite" in locked and has_invite_link(m):
            try:
                await m.delete()
            except Exception:
                pass
            return

    # ---- END LOCK ENFORCEMENT ----

    if sender_id in approved_users.get(chat_id, set()):
        return

    # Flood Protection
    flood_limit = flood_settings.get(chat_id)
    flood_mode = flood_modes.get(chat_id, default_flood_mode)

    if flood_limit and not is_admin_user and sender_id not in approved_users.get(chat_id, set()):
        key = (chat_id, sender_id)
        last_count, last_time = user_flood_counts.get(key, (0, 0))  # store timestamp, not msg_id
        now = time.time()

        # If messages are within 5 seconds → count as spam
        if now - last_time <= 5:
            curr_count = last_count + 1
        else:
            curr_count = 1

        user_flood_counts[key] = (curr_count, now)  # save timestamp instead of message_id

        if curr_count >= flood_limit:
            action, duration = flood_mode
            user_flood_counts[key] = (0, 0)  # reset first to avoid double trigger
            await take_flood_action(update, context, sender_id, action, duration)



    # ---- Blacklist Check (words + stickers + packs) ----
    if not is_admin_user and sender_id not in approved_users.get(chat_id, set()):
        triggered_type = None
        triggered_value = None

        # text
        text = message.text or message.caption or ""
        if text:
            lower_text = text.lower()
            for word in blacklist:
                if word in lower_text:
                    triggered_type = "word"
                    triggered_value = word
                    break

        # sticker
        if not triggered_type and message.sticker:
            if chat_id in sticker_blacklist and message.sticker.file_id in sticker_blacklist[chat_id]:
                triggered_type = "sticker"
                triggered_value = "sticker"
            elif message.sticker.set_name and chat_id in pack_blacklist and message.sticker.set_name in pack_blacklist[chat_id]:
                triggered_type = "pack"
                triggered_value = f"sticker pack {message.sticker.set_name}"

        if triggered_type:
            action = blacklist_modes.get(chat_id, "warn")

            try:
                await message.delete()
            except Exception:
                pass

            reason_text = f"{message.from_user.mention_html()} "

            if action == "warn":
                count = warnings.get(sender_id, 0) + 1
                warnings[sender_id] = count
                if count < 3:
                    await update.effective_chat.send_message(
                        f"{reason_text}warned ({count}/3).\nReason: used blacklisted {triggered_type} \"{triggered_value}\"",
                        parse_mode=ParseMode.HTML,
                    )
                else:
                    try:
                        await message.chat.ban_member(sender_id)
                        await update.effective_chat.send_message(
                            f"{reason_text}banned after 3 warnings.\nReason: used blacklisted {triggered_type} \"{triggered_value}\"",
                            parse_mode=ParseMode.HTML,
                        )
                        warnings[sender_id] = 0
                    except Exception as e:
                        await update.effective_chat.send_message(f"Failed to ban user: {e}")

            elif (isinstance(action, str) and action in ("tmute", "mute")) or (
                isinstance(action, tuple) and action[0] == "tmute"
            ):
                try:
                    if isinstance(action, tuple) and action[0] == "tmute":
                        _, duration_str = action
                        seconds = parse_time_to_seconds(duration_str)
                        if seconds <= 0:
                            seconds = 1800
                        until_date = datetime.utcnow() + timedelta(seconds=seconds)
                        await message.chat.restrict_member(
                            sender_id,
                            permissions=PTBChatPermissions(can_send_messages=False),
                            until_date=until_date,
                        )
                        temp_mutes[sender_id] = (chat_id, int(time.time()) + seconds)
                        mute_type = f"temporarily muted for {duration_str}"
                    elif action == "tmute":
                        seconds = 1800
                        until_date = datetime.utcnow() + timedelta(seconds=seconds)
                        await message.chat.restrict_member(
                            sender_id,
                            permissions=PTBChatPermissions(can_send_messages=False),
                            until_date=until_date,
                        )
                        temp_mutes[sender_id] = (chat_id, int(time.time()) + seconds)
                        mute_type = "temporarily muted for 30m"
                    else:
                        await message.chat.restrict_member(
                            sender_id,
                            permissions=PTBChatPermissions(can_send_messages=False),
                        )
                        mute_type = "muted"

                    await update.effective_chat.send_message(
                        f"{reason_text}has been {mute_type}.\nReason: used blacklisted {triggered_type} \"{triggered_value}\"",
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as e:
                    await update.effective_chat.send_message(f"Failed to mute user: {e}")

            elif action == "ban":
                try:
                    await message.chat.ban_member(sender_id)
                    await update.effective_chat.send_message(
                        f"{reason_text}banned.\nReason: used blacklisted {triggered_type} \"{triggered_value}\"",
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as e:
                    await update.effective_chat.send_message(f"Failed to ban user: {e}")

            elif action == "kick":
                try:
                    await message.chat.ban_member(sender_id)
                    await message.chat.unban_member(sender_id)
                    await update.effective_chat.send_message(
                        f"{reason_text}kicked.\nReason: used blacklisted {triggered_type} \"{triggered_value}\"",
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as e:
                    await update.effective_chat.send_message(f"Failed to kick user: {e}")

            elif action == "delete":
                await update.effective_chat.send_message(
                    f"Deleted message from {reason_text}\nReason: used blacklisted {triggered_type} \"{triggered_value}\"",
                    parse_mode=ParseMode.HTML,
                )

    # === AFK Logic ===
    if sender_id in afk_users:
        afk_data = afk_users.pop(sender_id)
        duration = datetime.utcnow() - afk_data["time"]
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_parts = []
        if hours:
            duration_parts.append(f"{hours}h")
        if minutes:
            duration_parts.append(f"{minutes}m")
        if seconds:
            duration_parts.append(f"{seconds}s")
        duration_str = " ".join(duration_parts) or "moments"
        user_chat = await context.bot.get_chat(sender_id)
        display_name = get_full_name(user_chat)
        text = f"[{display_name}](tg://user?id={sender_id}) Is now back online and they were afk for {duration_str}."

        if afk_data.get("reason") and afk_data["reason"] != "None":
            text += f"\nReason: {afk_data['reason']}"
        try:
            if afk_data.get("media"):
                if afk_data.get("media_type") == "photo":
                    await message.reply_photo(afk_data["media"], caption=text, parse_mode="HTML")
                elif afk_data.get("media_type") == "sticker":
                    file = await context.bot.get_file(afk_data["media"])
                    sticker_bytes = await file.download_as_bytearray()
                    image_obj = Image.open(BytesIO(sticker_bytes)).convert("RGBA")
                    bg = Image.new("RGBA", image_obj.size, (0, 0, 0, 255))
                    bg.paste(image_obj, (0, 0), image_obj)
                    bio = BytesIO()
                    bio.name = "sticker.png"
                    bg.save(bio, "PNG")
                    bio.seek(0)
                    await message.reply_photo(photo=InputFile(bio), caption=text, parse_mode="Markdown")
                else:
                    await message.reply_text(text, parse_mode="Markdown")
            else:
                await message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await message.reply_text(text, parse_mode="Markdown")

    mentioned_ids = set()
    if message.entities:
        for ent in message.entities:
            if ent.type == "text_mention" and ent.user:
                mentioned_ids.add(ent.user.id)
            elif ent.type == "mention":
                username = message.text[ent.offset: ent.offset + ent.length]
                for afk_id in afk_users.keys():
                    user = await context.bot.get_chat(afk_id)
                    if user.username and username.lower() == f"@{user.username.lower()}":
                        mentioned_ids.add(afk_id)

    if message.reply_to_message:
        replied_user = message.reply_to_message.from_user
        if replied_user and replied_user.id in afk_users:
            mentioned_ids.add(replied_user.id)

    for uid in mentioned_ids:
        if uid in afk_users:
            afk_data = afk_users[uid]
            duration = datetime.utcnow() - afk_data["time"]
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_parts = []
            if hours:
                duration_parts.append(f"{hours}h")
            if minutes:
                duration_parts.append(f"{minutes}m")
            if seconds:
                duration_parts.append(f"{seconds}s")
            duration_str = " ".join(duration_parts) or "moments"
            user_chat = await context.bot.get_chat(uid)
            display_name = get_full_name(user_chat)
            text = f"[{display_name}](tg://user?id={uid}) is AFK since {duration_str}."

            if afk_data.get("reason") and afk_data["reason"] != "None":
                text += f"\nReason: {afk_data['reason']}"
            try:
                if afk_data.get("media"):
                    if afk_data.get("media_type") == "photo":
                        await message.reply_photo(afk_data["media"], caption=text, parse_mode="HTML")
                    elif afk_data.get("media_type") == "sticker":
                        file = await context.bot.get_file(afk_data["media"])
                        sticker_bytes = await file.download_as_bytearray()
                        image_obj = Image.open(BytesIO(sticker_bytes)).convert("RGBA")
                        bg = Image.new("RGBA", image_obj.size, (0, 0, 0, 255))
                        bg.paste(image_obj, (0, 0), image_obj)
                        bio = BytesIO()
                        bio.name = "sticker.png"
                        bg.save(bio, "PNG")
                        bio.seek(0)
                        await message.reply_photo(photo=InputFile(bio), caption=text, parse_mode="Markdown")
                    else:
                        await message.reply_text(text, parse_mode="Markdown")
                else:
                    await message.reply_text(text, parse_mode="Markdown")
            except Exception:
                await message.reply_text(text, parse_mode="Markdown")

    # ---- NIGHTMODE ENFORCEMENT ----
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    if nightmode_status.get(chat_id, False):
        night_start = dtime(23, 0)
        night_end = dtime(7, 0)
        in_night = (now >= night_start or now <= night_end)
        if in_night and not is_admin_user and sender_id not in approved_users.get(chat_id, set()):
            if not message.text:
                try:
                    await message.delete()
                except Exception:
                    pass
                return

async def has_permission_pyro(client, message: Message, perms: list[str]) -> bool:
    member = await client.get_chat_member(message.chat.id, message.from_user.id)

    if member.status == "creator":
        return True

    if member.status == "administrator":
        if hasattr(member, "privileges") and member.privileges:
            return all(getattr(member.privileges, p, False) for p in perms)
        return all(getattr(member, p, False) for p in perms)

    return False





async def get_partner(user_id: int, chat_members):
    now = int(time.time())
    expired = [k for k, v in waifu_data.items() if now - v[1] > WAIFU_EXPIRY_SECONDS]
    for k in expired:
        del waifu_data[k]
    if user_id in waifu_data:
        partner_id, ts = waifu_data[user_id]
        if now - ts <= WAIFU_EXPIRY_SECONDS:
            return partner_id
    possible = [m for m in chat_members if m.id != user_id and not m.bot]
    if not possible:
        return None
    partner = random.choice(possible)
    waifu_data[user_id] = (partner.id, now)
    waifu_data[partner.id] = (user_id, now)
    return partner.id


async def send_waifu_photo(update, context, user_id: int, partner_id: int):
    bot = context.bot
    chat_id = update.effective_chat.id

    try:
        user = await tclient.get_entity(user_id)
    except Exception:
        user = None

    try:
        partner = await tclient.get_entity(partner_id)
    except Exception:
        partner = None

    user_name = getattr(user, 'first_name', 'User') or 'User'
    partner_name = getattr(partner, 'first_name', 'User') or 'User'

    caption = (
        f"<a href='tg://user?id={user_id}'>{user_name}</a> today's Waifu is : "
        f"<a href='tg://user?id={partner_id}'>{partner_name}</a> ({partner_id})\n\n"
        "🌌 Hope you have a nice day couples!\n\n"
        "♦️ Next waifu ~ 12:00 AM ( IST )"
    )

    if partner:
        photos = await tclient.get_profile_photos(partner)
        if photos.total > 0:
            photo = photos[0]
            photo_bytes = await tclient.download_media(photo, file=bytes)
            photo_io = io.BytesIO(photo_bytes)
            photo_io.name = "profile_photo.jpg"

            try:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_io,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                )
                return
            except Exception as e:
                await update.message.reply_text(
                    f"Failed to send photo, sending text only.\n{e}"
                )

    await bot.send_message(chat_id=chat_id, text=caption, parse_mode=ParseMode.HTML)


async def waifu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    message = update.message
    user_to_check = None

    # Prioritize reply user if any
    if message.reply_to_message:
        user_to_check = message.reply_to_message.from_user

    elif context.args:
        username_or_id = context.args[0].lstrip("@")

        # Try Bot API first
        try:
            if username_or_id.isdigit():
                user_to_check = await bot.get_chat(int(username_or_id))
            else:
                user_to_check = await bot.get_chat(username_or_id)
        except Exception:
            # Fallback to Telethon global user lookup
            try:
                user_to_check = await tclient.get_entity(username_or_id)
            except Exception as e:
                await update.message.reply_text(f"Could not find the user @{username_or_id}: {e}")
                return
    else:
        user_to_check = message.from_user

    chat = update.effective_chat
    participants = await tclient.get_participants(await tclient.get_entity(chat.id))

    # ✅ New check: user must be in group
    if not any(p.id == user_to_check.id for p in participants):
        await update.message.reply_text(f"❌ {user_to_check.first_name} is not in this group.")
        return

    partner_id = await get_partner(user_to_check.id, participants)
    if partner_id is None:
        await update.message.reply_text("Sorry, no users to pair with.")
        return

    await send_waifu_photo(update, context, user_to_check.id, partner_id)


# Register the command with
# 







# afk_users dictionary stores info: {user_id: {"start_time": datetime, "reason_text": str, "reason_message": Message or None}}





def escape_md(text: str) -> str:
    """
    Escapes all Telegram MarkdownV2 special characters in the given text.
    """
    escape_chars = r'[_*[\]()~`>#+\-=|{}.!]'
    return re.sub(escape_chars, r'\\\g<0>', text)













async def resolve_username_to_user(username: str, context):
    try:
        user = await context.bot.get_chat(username)
        return user
    except Exception:
        return None






async def mute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use mute commands.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to mute users.")
        return
    user = await resolve_user(update, context)
    if not user:
        return
    if await is_member_admin(update.effective_chat, user.id):
        await update.message.reply_text("I cannot mute an admin!")
        return

    try:
        await update.effective_chat.restrict_member(user.id,
            permissions=PTBChatPermissions(can_send_messages=False))

        await update.message.reply_text(f"Muted {format_name(user)}.")
    except BadRequest as e:
        await update.message.reply_text(f"Error: {e}")


def get_full_name(user):
    first = getattr(user, 'first_name', '') or ''
    last = getattr(user, 'last_name', '') or ''
    full_name = (first + ' ' + last).strip()
    return full_name if full_name else "User"




async def unmute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use mute commands.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be admin to unmute users.")
        return

    user = await resolve_user(update, context)
    if not user:
        return

    try:
        await update.effective_chat.restrict_member(
            user.id,
            permissions=PTBChatPermissions(
                can_send_messages=True,
                #can_send_audios=True,
                can_send_documents=True,
                can_send_photos=True,
                can_send_videos=True,
                can_send_video_notes=True,
                can_send_voice_notes=True,
                can_send_polls=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True,
                can_change_info=True,
                can_invite_users=True,
                can_pin_messages=True,
            ),
        )
        full_name = get_full_name(user)
        await update.message.reply_text(f"Unmuted {full_name}.")
    except BadRequest as e:
        await update.message.reply_text(f"Error: {e}")




async def user_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If there is a reply, show replied user ID
    if update.message.reply_to_message:
        user = update.message.reply_to_message.from_user
        await update.message.reply_text(f"User ID: `{user.id}`", parse_mode='MarkdownV2')
        return

    # If username or user ID argument provided
    if context.args:
        user = await resolve_user(update, context)
        if not user:
            return
        await update.message.reply_text(f"User ID: `{user.id}`", parse_mode='MarkdownV2')
        return

    # No args and no reply: show chat ID
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Chat ID: `{chat_id}`", parse_mode='MarkdownV2')


async def kick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to kick users.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to kick users.")
        return
    user = await resolve_user(update, context)
    if not user:
        return
    if await is_member_admin(update.effective_chat, user.id):
        await update.message.reply_text("I cannot kick an admin!")
        return

    try:
        await update.effective_chat.ban_member(user.id)
        await update.effective_chat.unban_member(user.id)
        await update.message.reply_text(f"Kicked {format_name(user)}.")
    except BadRequest as e:
        await update.message.reply_text(f"Error: {e}")




time_pattern = re.compile(r'(\d+)([smhd])')



time_pattern = re.compile(r'(\d+)\s*([smhd])')

def parse_time_to_seconds(time_str: str) -> int:
    """
    Parses time strings such as '1 s', '5s', '1m', '1h 30m', '2d 4h' into total seconds.
    """
    total_seconds = 0
    parts = time_str.lower().split()
    for part in parts:
        for value, unit in time_pattern.findall(part):
            value = int(value)
            if unit == 's':
                total_seconds += value
            elif unit == 'm':
                total_seconds += value * 60
            elif unit == 'h':
                total_seconds += value * 3600
            elif unit == 'd':
                total_seconds += value * 86400
    return total_seconds



async def tmute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_restrict_members"]):
        return await update.message.reply_text("🚫 You need 'Can Restrict Members' permission to use mute commands.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to temp mute users.")
        return

    user = await resolve_user(update, context)
    if not user:
        return
    if await is_member_admin(update.effective_chat, user.id):
        await update.message.reply_text("I cannot temp mute an admin!")
        return


    if not context.args:
        await update.message.reply_text("Usage: /tmute <time> (e.g., 5s, 1m 30s, 2h)")
        return

    time_input = " ".join(context.args)
    seconds = parse_time_to_seconds(time_input)

    if seconds == 0:
        await update.message.reply_text("Invalid time format. Examples: 5s, 1m, 2h 30m")
        return

    try:
        # Mute user indefinitely (no until_date here)
        await update.effective_chat.restrict_member(user.id,
            permissions=PTBChatPermissions(can_send_messages=False))

        # Store unmute time manually
        unmute_time = int(time.time()) + seconds
        temp_mutes[user.id] = (update.effective_chat.id, unmute_time)
        await update.message.reply_text(f"Temporarily muted {format_name(user)} for {time_input}.")
    except BadRequest as e:
        await update.message.reply_text(f"Failed to temp mute user: {e}")





async def unmute_expired_task(application):
    while True:
        now_ts = int(time.time())
        expired_users = [uid for uid, (_, ts) in temp_mutes.items() if ts <= now_ts]
        for uid in expired_users:
            chat_id, _ = temp_mutes[uid]
            try:
                await application.bot.restrict_chat_member(
                    chat_id=chat_id,
                    user_id=uid,
                    permissions=PTBChatPermissions(
                        can_send_messages=True,
                        can_send_audios=True,
                        can_send_documents=True,
                        can_send_photos=True,
                        can_send_videos=True,
                        can_send_video_notes=True,
                        can_send_voice_notes=True,
                        can_send_polls=True,
                        can_send_other_messages=True,
                        can_add_web_page_previews=True,
                        can_change_info=True,
                        can_invite_users=True,
                        can_pin_messages=True,
                    ),
                )
                del temp_mutes[uid]
                # Optionally notify group:
                # await application.bot.send_message(chat_id, f"User {uid} has been unmuted.")
            except Exception as e:
                print(f"Failed to unmute {uid}: {e}")
        await asyncio.sleep(5)  # Check every 5 seconds





async def kickme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    try:
        await update.effective_chat.ban_member(user.id)
        await update.effective_chat.unban_member(user.id)
        await update.message.reply_text("You have been kicked.")
    except BadRequest as e:
        await update.message.reply_text(f"Error: {e}")



async def nightmode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global nightmode
    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to toggle nightmode.")
        return
    if not context.args or context.args[0].lower() not in ("on", "off"):
        await update.message.reply_text("Usage: /nightmode <on|off>")
        return
    nightmode = context.args[0].lower() == "on"
    await update.message.reply_text(f"Night mode set to: {nightmode}")


async def safe_sleep(seconds, chat_id):
    for _ in range(seconds * 10):  # checks every 0.1 sec
        if not ongoing_tagall.get(chat_id, False):
            return
        await asyncio.sleep(0.1)




async def tagall(update, context):
    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("Only admins can use /tagall.")
        return

    chat = update.effective_chat
    chat_id = chat.id

    if ongoing_tagall.get(chat_id, False):
        await update.message.reply_text("A tagall is already running. Use /stop to cancel.")
        return

    ongoing_tagall[chat_id] = True
    message_text = " ".join(context.args) if context.args else ""
    message_text = escape_markdown(message_text)

    try:
        participants = await get_all_members(chat_id)
        user_mentions = []

        for user in participants:
            if user.bot:
                continue
            name = escape_markdown(user.first_name or "User")
            user_mentions.append(f"[{name}](tg://user?id={user.id})")

        batch_size = 5
        for i in range(0, len(user_mentions), batch_size):
            # Check here if /stop was called, break if so
            if not ongoing_tagall.get(chat_id, False):
                break

            batch_mentions = user_mentions[i : i + batch_size]
            tag_message = message_text + "\n\n" + " ".join(batch_mentions)

            await context.bot.send_message(
                chat_id=chat_id,
                text=tag_message,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True,
            )

            # Check again before sleeping, to exit early if stopped
            if not ongoing_tagall.get(chat_id, False):
                break

            await safe_sleep(1, chat_id)

    except Exception as e:
        await update.message.reply_text(f"Failed to fetch members or send tags: {e}")

    ongoing_tagall[chat_id] = False





async def stop_tagall(update, context):
    chat = update.effective_chat
    chat_id = chat.id
    if ongoing_tagall.get(chat_id, False):
        ongoing_tagall[chat_id] = False
        await update.message.reply_text("Tagall stopped.")
    else:
        await update.message.reply_text("No ongoing tagall to stop.")



async def pin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_pin_messages"]):
        return await update.message.reply_text("🚫 You need 'Can Pin Messages' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("You must be an admin to pin messages.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a message to pin it.")
        return
    try:
        await update.message.reply_to_message.pin()
        await update.message.reply_text("Message pinned.")
    except BadRequest as e:
        await update.message.reply_text(f"Error: {e}")

async def unpin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_pin_messages"]):
        return await update.message.reply_text("🚫 You need 'Can Pin Messages' permission to use this command.")

    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("Only admins can unpin messages.")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("Please reply to the pinned message to unpin it.")
        return

    try:
        await update.message.reply_to_message.unpin()
        await update.message.reply_text("Unpinned the selected message.")
    except Exception as e:
        await update.message.reply_text(f"Failed to unpin message: {e}")










async def purge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_delete_messages"]):
        return await update.message.reply_text("🚫 You need 'Can Delete Messages' permission to use this command.")

    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a message to start purging.")
        return

    chat = update.effective_chat
    start_msg = update.message.reply_to_message.message_id
    end_msg = update.message.message_id

    start_time = datetime.now()
    deleted = 0

    for msg_id in range(start_msg, end_msg + 1):
        try:
            await context.bot.delete_message(chat.id, msg_id)
            deleted += 1
        except Exception as e:
            # Debug log (optional, remove in production)
            print(f"Failed to delete {msg_id}: {e}")
            continue

    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    result_text = (
        f"✅ Purge successful\n"
        f"{deleted} messages deleted\n"
        f"Took {hours} hr {minutes} min {seconds} sec"
    )

    await context.bot.send_message(
        chat_id=chat.id,
        text=result_text
    )









async def quote_command(update, context):
    message = update.message
    if not message or not message.reply_to_message:
        await message.reply_text("Please reply to a message and use /q")
        return

    replied = message.reply_to_message
    text = replied.text or replied.caption
    user = replied.from_user

    if not text:
        await message.reply_text("Cannot quote empty message")
        return

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    profile_image_path = None
    try:
        photos = await context.bot.get_user_profile_photos(user.id)
        if photos.total_count > 0:
            file = await context.bot.get_file(photos.photos[0][-1].file_id)
            profile_image_path = os.path.join(temp_dir, f"{user.id}_profile.jpg")
            await file.download_to_drive(profile_image_path)
        else:
            # User has no profile photo - generate placeholder
            initial = (user.first_name[0] if user.first_name else "U").upper()
            profile_image_path = os.path.join(temp_dir, f"{user.id}_placeholder.png")
            create_placeholder_avatar(profile_image_path, initial, user.id)
    except Exception as e:
        print(f"Error fetching profile photo, generating placeholder: {e}")
        initial = (user.first_name[0] if user.first_name else "U").upper()
        profile_image_path = os.path.join(temp_dir, f"{user.id}_placeholder.png")
        create_placeholder_avatar(profile_image_path, initial, user.id)

    temp_png = tempfile.mktemp(suffix=".png")
    temp_webp = tempfile.mktemp(suffix=".webp")

    try:
        wait_msg = await message.reply_text("Please wait while quoting your text...")

        await create_quote_image(
            name=user.first_name or "User",
            message=text,
            profile_image=profile_image_path,
            output_path=temp_png,
        )
        img = Image.open(temp_png)
        img.save(temp_webp, "WEBP", lossless=True)

        await wait_msg.delete()  # remove waiting message
        with open(temp_webp, "rb") as sticker_file:
            await message.reply_sticker(sticker=InputFile(sticker_file))

    except Exception as e:
        await message.reply_text("Failed to create sticker.")
        print(f"Error: {e}")

    finally:
        for p in [profile_image_path, temp_png, temp_webp]:
            if p and os.path.exists(p):
                os.remove(p)



def create_initial_avatar(path, initial):
    # Create 60x60 px image with solid background and centered initial
    img = Image.new('RGBA', (60, 60), color=(100, 100, 200, 255))  # Your preferred bg color
    draw = ImageDraw.Draw(img)
    # Use a truetype font available on your system or bundled
    font_path = "fonts/arial.ttf"  # Update your font path accordingly
    font = ImageFont.truetype(font_path, 40)
    bbox = draw.textbbox((0, 0), initial, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    draw.text(((60 - w) / 2, (60 - h) / 2 - 5), initial, font=font, fill=(255, 255, 255, 255))
    img.save(path)



def get_color_for_user(user_id: int):
    colors = [
        (66, 133, 244),    # Blue
        (219, 68, 55),     # Red
        (244, 180, 0),     # Yellow
        (15, 157, 88),     # Green
        (171, 71, 188),    # Purple
        (0, 172, 193),     # Teal
    ]
    # Hash user_id to pick a color index deterministically
    idx = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % len(colors)
    return colors[idx]

def create_placeholder_avatar(path: str, initial: str, user_id: int):
    bg_color = get_color_for_user(user_id)
    img = Image.new('RGBA', (60, 60), color=bg_color + (255,))
    draw = ImageDraw.Draw(img)
    font_path = "fonts/arial.ttf"  # Update path to your font
    font = ImageFont.truetype(font_path, 40)
    bbox = draw.textbbox((0, 0), initial, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text(((60 - w) / 2, (60 - h) / 2 - 5), initial, font=font, fill=(255, 255, 255, 255))
    img.save(path)




# Helper to resolve username to user
async def resolve_username_to_user(username: str, context):
    try:
        user = await context.bot.get_chat(username)
        return user
    except Exception:
        return None
    


async def afk_command(update, context):
    """
    Sets user as AFK.
    Saves reason and optional media (photo or sticker) if replied to while setting AFK.
    """
    global afk_users

    if not update.message:
        return

    user = update.effective_user
    reason = "None"
    if context.args:
        reason = " ".join(context.args).strip() or "None"

    afk_media = None
    afk_media_type = None

    if update.message.reply_to_message:
        reply = update.message.reply_to_message
        if reply.photo:
            afk_media = reply.photo[-1].file_id
            afk_media_type = "photo"
        elif reply.sticker:
            afk_media = reply.sticker.file_id
            afk_media_type = "sticker"

    afk_users[user.id] = {
        "time": datetime.utcnow(),
        "reason": reason,
        "media": afk_media,
        "media_type": afk_media_type,
    }

    text = f"[{user.first_name}](tg://user?id={user.id}) Is now away from keyboard! Sayonara!"
    if reason and reason != "None":
        text += f"\nReason: {reason}"

    await update.message.reply_text(text, parse_mode="Markdown")







async def getsticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message or not update.message.reply_to_message.sticker:
        await update.message.reply_text("Please reply to a sticker with /getsticker.")
        return

    sticker = update.message.reply_to_message.sticker
    file_id = sticker.file_id
    emoji = sticker.emoji if sticker.emoji else "No emoji"

    # Download sticker file
    file = await context.bot.get_file(file_id)
    file_bytes = await file.download_as_bytearray()

    # Convert to PNG if it's a static sticker (webp)
    image_io = BytesIO()
    if not sticker.is_animated and not sticker.is_video:
        img = Image.open(BytesIO(file_bytes)).convert("RGBA")
        img.save(image_io, format="PNG")
        image_io.name = "sticker.png"
        image_io.seek(0)

        await update.message.reply_photo(
            photo=InputFile(image_io),
            caption=f"Sticker ID:\n<code>{file_id}</code>\nEmoji: {emoji}",
            parse_mode="HTML"
        )
    else:
        # For animated/video stickers just send back the file
        await update.message.reply_document(
            document=InputFile(BytesIO(file_bytes), filename="sticker.tgs" if sticker.is_animated else "sticker.webm"),
            caption=f"Sticker ID:\n<code>{file_id}</code>\nEmoji: {emoji}",
            parse_mode="HTML"
        )



async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message or not update.message.reply_to_message.text:
        await update.message.reply_text("Reply to a text message with /tr <lang_code>")
        return

    if not context.args:
        await update.message.reply_text("Usage: /tr <lang_code>")
        return

    target_lang = context.args[0].lower()
    original_text = update.message.reply_to_message.text

    try:
        # Ask Groq to translate
        prompt = f"Translate this text to {target_lang}:\n\n{original_text}"
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        translated = response.choices[0].message.content.strip()

        await update.message.reply_text(
            f"**Translated ({target_lang}):**\n{translated}",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Error translating: {e}")



async def translate_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    codes = """
🌐 Language Codes:
- English: en
- Hindi: hi
- Spanish: es
- French: fr
- German: de
- Russian: ru
- Chinese (Simplified): zh-cn
- Japanese: ja
- Arabic: ar
- Bengali: bn
- Tamil: ta
- Telugu: te
- Malayalam: ml
(and many more supported by Google Translate)
"""
    await update.message.reply_text(codes)





async def describe_image(image_path):
    """Send image to Gemini API and return a concise formatted description"""
    try:
        with open(image_path, "rb") as img:
            response = gemini_model.generate_content([
                "Describe this image in no more than 5-6 lines. "
                "Use bold for important keywords or names. "
                "Be concise and avoid excessive details.",
                {"mime_type": "image/jpeg", "data": img.read()}
            ])
        return response.text.strip()
    except Exception as e:
        return f"❌ Error analyzing image: {e}"

# Markdown escape helper
def escape_md(text: str) -> str:
    """Escape special characters for Markdown formatting in Telegram"""
    escape_chars = r'[_*\[\]()~`>#+\-=|{}.!]'
    return re.sub(escape_chars, r'\\\g<0>', text)

async def pp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    if not message.reply_to_message or not message.reply_to_message.photo:
        await message.reply_text("⚠️ Please reply to an image with /pp.")
        return

    # Download the replied photo
    photo = message.reply_to_message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = os.path.join(tempfile.gettempdir(), f"{photo.file_id}.jpg")
    await file.download_to_drive(file_path)

    await message.reply_text("🔍 Analyzing image...")

    # Get the Gemini description
    description = await describe_image(file_path)

    # Clean and escape description for safe Markdown rendering
    clean_description = " ".join(description.split())
    safe_description = boldify(clean_description)




    # Prepare Google search link
    search_query = quote_plus(clean_description)

    buttons = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔎 Google Search", url=f"https://www.google.com/search?q={search_query}")]]
    )

    # Send formatted message with Markdown parsing
    await message.reply_text(
        f"Looks like:\n{safe_description}",
        reply_markup=buttons,
        parse_mode=ParseMode.HTML

    )

    # Clean up downloaded file
    os.remove(file_path)



async def calc_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    expression = " ".join(context.args)

    if not expression:
        await message.reply_text("⚠️ Usage: `/calc 5+2`", parse_mode=ParseMode.MARKDOWN)
        return

    try:
        # Only allow safe characters: digits, operators, parentheses, decimal points, and spaces
        if not re.match(r'^[0-9+\-*/().\s]+$', expression):
            await message.reply_text("❌ Invalid characters in expression!")
            return

        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": None}, {"math": math})
        await message.reply_text(f"🧮 Result: `{result}`", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")



async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    if not message.reply_to_message:
        await message.reply_text("⚠️ Please reply to a message to report it.")
        return

    try:
        chat_id = update.effective_chat.id
        bot = context.bot

        # Get list of admins
        admins = await bot.get_chat_administrators(chat_id)

        # Add invisible mentions (zero-width character)
        invisible_mentions = "".join(
            [f"[‎](tg://user?id={admin.user.id})" for admin in admins if not admin.user.is_bot]
        )

        # Send message with invisible mentions
        await message.reply_text(
            f"🚨 Reported to admins{invisible_mentions}",
            parse_mode="Markdown"
        )

    except Exception as e:
        await message.reply_text(f"❌ Error reporting: {e}")




# --- CALLBACK HANDLERS ---
async def anime_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    # --- Characters ---
    if data.startswith("anime_chars:"):
        parts = data.split(":")
        anime_id = int(parts[1])
        page = int(parts[2]) if len(parts) > 2 else 1

        variables = {"id": anime_id, "page": page}
        response = requests.post(
            ANILIST_URL,
            json={"query": ANIME_QUERY, "variables": variables},
            headers={"Content-Type": "application/json"}
        ).json()

        media = response.get("data", {}).get("Media", {})
        chars_info = media.get("characters", {})
        edges = chars_info.get("edges", [])
        page_info = chars_info.get("pageInfo", {})

        if not edges:
            await query.edit_message_caption(
                caption="❌ No character data found for this anime.",
                parse_mode="HTML"
            )
            return

        characters = [
            f"• <a href='{edge['node']['siteUrl']}'>{edge['node']['name']['full']}</a> ({edge['role']})"
            for edge in edges
        ]
        total = page_info.get("total", len(edges))
        text = "🎭 <b>Characters List:</b>\n" + "\n".join(characters) + f"\n\nTotal Characters: {total}"

        # Pagination buttons
        buttons = []
        if page > 1:
            buttons.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"anime_chars:{anime_id}:{page-1}"))
        if page_info.get("hasNextPage"):
            buttons.append(InlineKeyboardButton("➡️ Next", callback_data=f"anime_chars:{anime_id}:{page+1}"))

        # Back button
        buttons.append(InlineKeyboardButton("🔙 Back", callback_data=f"anime_home:{anime_id}"))
        keyboard = [buttons]

        await query.edit_message_caption(
            caption=text[:1000] + "..." if len(text) > 1024 else text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # --- Description ---
    elif data.startswith("anime_desc:"):
        anime_id = int(data.split(":")[1])
        variables = {"id": anime_id}
        response = requests.post(
            ANILIST_URL,
            json={"query": ANIME_QUERY, "variables": variables}
        ).json()
        anime = response["data"]["Media"]

        desc = anime.get("description", "No description available")
        desc = re.sub(r"<.*?>", "", desc)  # strip AniList HTML
        if len(desc) > 900:
            desc = desc[:900] + "..."

        text = f"📖 <b>Description:</b>\n\n{desc}"

        keyboard = [
            [InlineKeyboardButton("🔗 More Info", url=anime["siteUrl"])],
            [InlineKeyboardButton("🔙 Back", callback_data=f"anime_home:{anime_id}")]
        ]

        await query.edit_message_caption(caption=text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))

    # --- Related Series ---
    elif data.startswith("anime_series:"):
        anime_id = int(data.split(":")[1])
        variables = {"id": anime_id}
        response = requests.post(
            ANILIST_URL,
            json={"query": RELATIONS_QUERY, "variables": variables}
        ).json()
        relations = response["data"]["Media"]["relations"]["edges"]

        series_list = [
            f"• <a href='{rel['node']['siteUrl']}'>{rel['node']['title']['romaji']}</a> ({rel['relationType']})"
            for rel in relations[:15]
        ]
        text = "📺 <b>Related Series:</b>\n" + ("\n".join(series_list) if series_list else "No related series found.")

        keyboard = [
            [InlineKeyboardButton("🔗 More Info", url=f"https://anilist.co/anime/{anime_id}")],
            [InlineKeyboardButton("🔙 Back", callback_data=f"anime_home:{anime_id}")]
        ]

        await query.edit_message_caption(
            caption=text[:1000] + "..." if len(text) > 1024 else text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # --- Back to Home ---
    elif data.startswith("anime_home:"):
        anime_id = int(data.split(":")[1])
        variables = {"id": anime_id}
        response = requests.post(
            ANILIST_URL,
            json={"query": ANIME_QUERY, "variables": variables}
        ).json()
        anime = response["data"]["Media"]

        title = anime["title"]["romaji"] or anime["title"]["english"] or "Unknown"
        english = anime["title"]["english"] or title
        mal_id = anime.get("idMal", "N/A")
        source = anime.get("source", "N/A")
        atype = anime.get("type", "N/A")
        score = f"{anime['averageScore']}% 🌟" if anime.get("averageScore") else "N/A"
        duration = f"{anime['duration']} min/ep" if anime.get("duration") else "N/A"
        status = anime.get("status", "N/A")
        episodes = anime.get("episodes", "N/A")
        if status == "FINISHED" and episodes != "N/A":
            status = f"{status} | {episodes} eps"

        genres = ", ".join(anime.get("genres", [])[:5]) or "N/A"
        tags = ", ".join(tag["name"] for tag in anime.get("tags", [])[:8]) or "N/A"
        trailer = "N/A"
        if anime.get("trailer"):
            if anime["trailer"]["site"] == "youtube":
                trailer = f"https://youtu.be/{anime['trailer']['id']}"
            else:
                trailer = f"{anime['trailer']['site']}/{anime['trailer']['id']}"

        caption = (
            f"[🇯🇵]<b>{title}</b> | {english}\n"
            f"<b>ID</b> | <b>MAL ID</b>: <code>{anime['id']}</code> | <code>{mal_id}</code>\n"
            f"➤ <b>SOURCE:</b> <code>{source}</code>\n"
            f"➤ <b>TYPE:</b> <code>{atype}</code>\n"
            f"➤ <b>SCORE:</b> <code>{score}</code>\n"
            f"➤ <b>DURATION:</b> <code>{duration}</code>\n"
            f"➤ <b>STATUS:</b> <code>{status}</code>\n"
            f"➤ <b>GENRES:</b> <code>{genres}</code>\n"
            f"➤ <b>TAGS:</b> <code>{tags}</code>\n"
            f'🎬 <a href="{trailer}">Trailer</a>\n'
            f"📖 <a href='{anime['siteUrl']}'>Official Site</a>"
        )

        keyboard = [
            [InlineKeyboardButton("🎭 Characters", callback_data=f"anime_chars:{anime['id']}:1")],
            [InlineKeyboardButton("📖 Description", callback_data=f"anime_desc:{anime['id']}")],
            [InlineKeyboardButton("📺 List Series", callback_data=f"anime_series:{anime['id']}")]
        ]

        await query.edit_message_caption(
            caption=caption[:1000] + "..." if len(caption) > 1024 else caption,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )



async def ud_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ud <word>\nExample: /ud hello")
        return

    word = " ".join(context.args)
    url = f"https://api.urbandictionary.com/v0/define?term={word}"
    try:
        response = requests.get(url)
        data = response.json()

        if not data.get("list"):
            await update.message.reply_text(f"No Urban Dictionary entries found for '{word}'.")
            return

        # Pick the top definition
        top_def = data["list"][0]
        definition = top_def.get("definition", "").replace('[', '').replace(']', '')
        example = top_def.get("example", "").replace('[', '').replace(']', '')

        reply_text = (
            f"<b>{word.capitalize()}</b>:\n"
            f"{definition}\n\n"
            f"<i>Example:</i> {example}"
        )
        await update.message.reply_text(reply_text, parse_mode="HTML")

    except Exception as e:
        await update.message.reply_text(f"Error fetching definition: {e}")





from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import pytz
from datetime import datetime, time as dtime

async def nightmode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.message.from_user.id):
        await update.message.reply_text("🚫 Only admins can manage Nightmode.")
        return

    chat_id = update.effective_chat.id
    status = nightmode_status.get(chat_id, False)

    buttons = []
    if status:
        buttons.append(InlineKeyboardButton("Deactivate", callback_data=f"nightmode_off:{chat_id}"))
    else:
        buttons.append(InlineKeyboardButton("Activate", callback_data=f"nightmode_on:{chat_id}"))

    keyboard = InlineKeyboardMarkup([buttons])
    await update.message.reply_text(
        f"🌙 Nightmode is currently {'ON' if status else 'OFF'}",
        reply_markup=keyboard,
    )


async def nightmode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id

    if query.data.startswith("nightmode_on"):
        nightmode_status[chat_id] = True
        schedule_nightmode_jobs(context.application, chat_id)
        await query.edit_message_text("🌙 Nightmode activated.")
    elif query.data.startswith("nightmode_off"):
        nightmode_status[chat_id] = False
        await query.edit_message_text("☀️ Nightmode deactivated.")


def schedule_nightmode_jobs(application, chat_id):
    # remove old jobs for this chat
    for job in application.job_queue.get_jobs_by_name(str(chat_id)):
        job.schedule_removal()

    # 11:00 PM
    application.job_queue.run_daily(
        nightmode_start,
        time=dtime(23, 0, tzinfo=pytz.timezone("Asia/Kolkata")),
        name=str(chat_id),
        chat_id=chat_id,
    )
    # 7:00 AM
    application.job_queue.run_daily(
        nightmode_end,
        time=dtime(7, 0, tzinfo=pytz.timezone("Asia/Kolkata")),
        name=str(chat_id),
        chat_id=chat_id,
    )

async def nightmode_start(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    if nightmode_status.get(chat_id, False):
        await context.bot.send_message(chat_id, "🌙 Nightmode has started in this group. Only text allowed for non-approved/non-admins.")

async def nightmode_end(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    if nightmode_status.get(chat_id, False):
        await context.bot.send_message(chat_id, "☀️ Nightmode has ended. Everyone can send anything now.")




# --- NEW COMMANDS ---

async def unapproveall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user

    # Fetch member info
    member = await context.bot.get_chat_member(chat.id, user.id)

    # Only group owner can use
    if member.status != "creator":
        return await update.message.reply_text("🚫 Only the group owner can use this command.")

    chat_id = chat.id
    if chat_id in approved_users:
        approved_users[chat_id].clear()
        await update.message.reply_text("✅ All approved users have been unapproved.")
    else:
        await update.message.reply_text("ℹ️ No approved users found in this chat.")


async def approved(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await has_permission(update, ["can_change_info"]):
        return await update.message.reply_text("🚫 You need 'Can Change Info' permission to use this command.")

    chat_id = update.effective_chat.id
    if chat_id not in approved_users or not approved_users[chat_id]:
        return await update.message.reply_text("ℹ️ No users are approved in this chat.")
    text = "✅ Approved users:\n"
    for uid in approved_users[chat_id]:
        try:
            user = await context.bot.get_chat(uid)
            text += f"• {user.first_name} (@{user.username})\n"
        except:
            text += f"• {uid}\n"
    await update.message.reply_text(text)

from telegram.constants import ChatMemberStatus

async def unblacklistall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat = update.effective_chat

    # Check if user is group owner
    member = await chat.get_member(user_id)
    if member.status != ChatMemberStatus.OWNER:
        return await update.message.reply_text("🚫 Only the group owner can use this command.")

    # Clear blacklist
    blacklist.clear()
    await update.message.reply_text("✅ All blacklist words have been removed by the group owner.")


async def warns(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = None

    # 1. Try to resolve target user if mentioned/replied
    if update.message.reply_to_message or context.args:
        user = await resolve_user(update, context)

    # 2. If no user mentioned → fallback to self
    if not user:
        user = update.effective_user
        if await is_member_admin(update.effective_chat, user.id):
            return await update.message.reply_text(
                "As per You are admin.. You don't have any warnings!"
            )

    # 3. If mentioned user is admin
    if await is_member_admin(update.effective_chat, user.id):
        return await update.message.reply_text(
            f"<a href='tg://user?id={user.id}'>{format_name(user)}</a> is admin so they haven't any warnings.",
            parse_mode="HTML"
        )

    # 4. Normal user: show warns
    count = warnings.get(user.id, 0)
    reasons = warn_reasons.get(user.id, [])

    if count == 0:
        return await update.message.reply_text(
            f"<a href='tg://user?id={user.id}'>{format_name(user)}</a> has no warnings.",
            parse_mode="HTML"
        )

    reason_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(reasons)]) if reasons else "No reasons recorded."

    await update.message.reply_text(
        f"<a href='tg://user?id={user.id}'>{format_name(user)}</a> has {count}/3 warnings! Be careful!\n\n"
        f"- Reasons:\n{reason_text}",
        parse_mode="HTML"
    )




async def goodbye_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("🚫 Only admins can toggle goodbye messages.")

    if not context.args or context.args[0].lower() not in ["on", "off"]:
        return await update.message.reply_text("Usage: /goodbye on|off")

    chat_id = update.effective_chat.id
    enabled = context.args[0].lower() == "on"
    goodbye_settings.setdefault(chat_id, {})["enabled"] = enabled
    await update.message.reply_text(f"✅ Goodbye messages {'enabled' if enabled else 'disabled'} in this chat.")


async def set_goodbye(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("🚫 Only admins can set goodbye message.")

    chat_id = update.effective_chat.id
    goodbye_settings.setdefault(chat_id, {"enabled": True})

    # If replied to a message → save that message (media supported)
    if update.message.reply_to_message:
        msg = update.message.reply_to_message
        goodbye_settings[chat_id]["message"] = msg.to_dict()
        return await update.message.reply_text("✅ Goodbye message set from the replied message.")

    # Else, take args as plain text
    text = " ".join(context.args)
    if not text:
        return await update.message.reply_text("Usage:\nReply with /setgoodbye\nor /setgoodbye <text>")
    goodbye_settings[chat_id]["message"] = {"text": text}
    await update.message.reply_text("✅ Goodbye message set.")


async def set_goodbye(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("🚫 Only admins can set goodbye message.")

    chat_id = update.effective_chat.id
    goodbye_settings.setdefault(chat_id, {"enabled": True})

    # If replied to a message → save that message (media supported)
    if update.message.reply_to_message:
        msg = update.message.reply_to_message
        goodbye_settings[chat_id]["message"] = msg.to_dict()
        return await update.message.reply_text("✅ Goodbye message set from the replied message.")

    # Else, take args as plain text
    text = " ".join(context.args)
    if not text:
        return await update.message.reply_text("Usage:\nReply with /setgoodbye\nor /setgoodbye <text>")
    goodbye_settings[chat_id]["message"] = {"text": text}
    await update.message.reply_text("✅ Goodbye message set.")


async def send_goodbye(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in goodbye_settings or not goodbye_settings[chat_id].get("enabled", True):
        return

    msg_data = goodbye_settings[chat_id].get("message")
    if not msg_data:
        return  # no goodbye message set

    member = update.message.left_chat_member
    mention = f"<a href='tg://user?id={member.id}'>{member.first_name}</a>"
    chatname = update.effective_chat.title

    def replace_vars(text):
        return text.replace("{mention}", mention).replace("{chat}", chatname)

    if "text" in msg_data:
        await update.message.reply_text(
            replace_vars(msg_data["text"]),
            parse_mode=ParseMode.HTML
        )
    else:
        try:
            await context.bot.copy_message(
                chat_id=chat_id,
                from_chat_id=chat_id,
                message_id=msg_data["message_id"],
                caption=replace_vars(msg_data.get("caption", "")) if msg_data.get("caption") else None,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ Failed to send goodbye message: {e}")






from datetime import datetime, timezone, timedelta
import pytz  # already used earlier

async def when(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ Reply to a message and use /when to see when it was sent.")
        return

    replied_msg = update.message.reply_to_message
    msg_time_utc = replied_msg.date  # datetime object in UTC

    # Convert UTC → IST
    ist = pytz.timezone("Asia/Kolkata")
    msg_time_ist = msg_time_utc.astimezone(ist)

    # Format IST time
    formatted_time = msg_time_ist.strftime("%A, %B %d, %Y at %I:%M:%S %p (IST)")

    # Calculate how long ago
    now = datetime.now(timezone.utc)
    diff = now - msg_time_utc

    days, seconds = diff.days, diff.seconds
    if days > 0:
        ago = f"{days} day{'s' if days > 1 else ''} ago"
    elif seconds >= 3600:
        hrs = seconds // 3600
        ago = f"{hrs} hour{'s' if hrs > 1 else ''} ago"
    elif seconds >= 60:
        mins = seconds // 60
        ago = f"{mins} minute{'s' if mins > 1 else ''} ago"
    else:
        ago = f"{seconds} second{'s' if seconds != 1 else ''} ago"

    text = (
        f"📅 Originally posted on {formatted_time}\n\n"
        f"⏳ That's {ago}."
    )

    await update.message.reply_text(text)



async def captcha_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, update.message.from_user.id):
        return await update.message.reply_text("🚫 Only admins can toggle captcha.")

    if not context.args or context.args[0].lower() not in ["on", "off"]:
        return await update.message.reply_text("Usage: /captcha on|off")

    chat_id = update.effective_chat.id
    enabled = context.args[0].lower() == "on"
    captcha_settings[chat_id] = enabled
    await update.message.reply_text(f"✅ Captcha {'enabled' if enabled else 'disabled'} in this chat.")




async def captcha_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        _, cid, uid, picked = query.data.split(":")
        chat_id = int(cid)
        user_id = int(uid)
        picked = int(picked)
    except Exception:
        return

    if query.from_user.id != user_id:
        return await query.answer("⚠️ Not your captcha.", show_alert=True)

    state = captcha_state.get((chat_id, user_id))
    if not state:
        return await query.edit_message_text("Captcha expired. Go back to the group and press the button again.")

    correct = state["answer"]
    if picked == correct:
        # unmute
        try:
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions=PTBChatPermissions(
                    can_send_messages=True,
                    can_send_audios=True,
                    can_send_documents=True,
                    can_send_photos=True,
                    can_send_videos=True,
                    can_send_video_notes=True,
                    can_send_voice_notes=True,
                    can_send_polls=True,
                    can_send_other_messages=True,
                    can_add_web_page_previews=True,
                    can_change_info=True,
                    can_invite_users=True,
                    can_pin_messages=True,
                ),
            )
        except Exception as e:
            print(f"Unmute failed: {e}")

        captcha_state.pop((chat_id, user_id), None)
        return await query.edit_message_text("✅ Correct! You’ve been unmuted.")
    else:
        state["tries"] += 1
        remaining = 3 - state["tries"]
        if remaining <= 0:
            captcha_state.pop((chat_id, user_id), None)
            try:
                await context.bot.ban_chat_member(chat_id, user_id)
                await context.bot.unban_chat_member(chat_id, user_id)  # kick
            except Exception as e:
                print(f"Kick failed: {e}")
            return await query.edit_message_text("❌ Wrong again. You used all tries and were removed from the group.")
        else:
            # keep same question, refresh choices
            kb = _build_9_options_keyboard(chat_id, user_id, correct)
            await query.edit_message_text(
                f"❌ Wrong. {remaining} attempt(s) left.\n\n<b>{state['question']} = ?</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=kb
            )


from telethon import events

MOD_IDS = {7038303029, 8056147438, 7560366347, 7556899383}  # replace with your actual mod IDs

async def send_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id not in MOD_IDS:   # same check as before
        return

    # === Case 1: In group and replying to someone ===
    if update.message.chat.type in ("group", "supergroup") and update.message.reply_to_message:
        if not context.args:
            await update.message.reply_text("Please provide a message to send.")
            return
        msg = " ".join(context.args)
        try:
            await update.message.delete()
        except:
            pass
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=msg,
            reply_to_message_id=update.message.reply_to_message.message_id
        )
        return

    # === Case 2: In private chat ===
    if update.message.chat.type == "private":
        if update.message.reply_to_message:
            args = context.args
            if not args:
                await update.message.reply_text("Usage: /send <chat_id> (as reply to media/text)")
                return
            try:
                target_chat_id = int(args[0])
                reply_msg = update.message.reply_to_message
                if reply_msg.photo:
                    file_id = reply_msg.photo[-1].file_id
                    await context.bot.send_photo(target_chat_id, file_id, caption=reply_msg.caption or "")
                elif reply_msg.document:
                    await context.bot.send_document(target_chat_id, reply_msg.document.file_id, caption=reply_msg.caption or "")
                elif reply_msg.text:
                    await context.bot.send_message(target_chat_id, reply_msg.text)
                await update.message.reply_text("Anonymous message sent.")
            except Exception as e:
                await update.message.reply_text(f"Failed to send message:\n{e}")
            return

        if len(context.args) < 2:
            await update.message.reply_text("Usage:\n1. Reply to media/text: /send <chat_id>\n2. Or: /send <chat_id> <message>")
            return
        try:
            target_chat_id = int(context.args[0])
            message = " ".join(context.args[1:])
            await context.bot.send_message(target_chat_id, message)
            await update.message.reply_text("Message sent.")
        except Exception as e:
            await update.message.reply_text(f"Failed to send message:\n{e}")
        return

    await update.message.reply_text(
        "Usage:\n"
        "- In group (reply): /send <message>\n"
        "- In PM:\n"
        "  • /send <chat_id> <message>\n"
        "  • reply to media/text with /send <chat_id>"
    )

# === FREE / UNFREE SYSTEM ===
free_users = {}  # {chat_id: set(user_ids)}

async def get_target_user(client, message, args):
    """Get target user from reply or /command argument"""
    if message.reply_to_message:
        return message.reply_to_message.from_user

    if args:
        username_or_id = args[0].lstrip("@")
        try:
            if username_or_id.isdigit():
                return await client.get_users(int(username_or_id))
            else:
                return await client.get_users(username_or_id)
        except Exception:
            return None
    return None


@pyro_client.on_message(pyro_filters.command("free"))
async def free_user(client, message):
    if not await is_admin_pyro(client, message.chat.id, message.from_user.id):
        return await message.reply_text("🚫 Only admins can use this command.")

    args = message.command[1:] if hasattr(message, "command") else []
    target = await get_target_user(client, message, args)

    if not target:
        return await message.reply_text("Reply to a user or provide a username/ID.\nUsage: /free @username")

    free_users.setdefault(message.chat.id, set()).add(target.id)
    await message.reply_text(f"✅ {target.mention} can now send stickers & GIFs.")


@pyro_client.on_message(pyro_filters.command("unfree"))
async def unfree_user(client, message):
    if not await is_admin_pyro(client, message.chat.id, message.from_user.id):
        return await message.reply_text("🚫 Only admins can use this command.")

    args = message.command[1:] if hasattr(message, "command") else []
    target = await get_target_user(client, message, args)

    if not target:
        return await message.reply_text("Reply to a user or provide a username/ID.\nUsage: /unfree @username")

    free_users.setdefault(message.chat.id, set()).discard(target.id)
    await message.reply_text(f"❌ {target.mention} can no longer send stickers & GIFs.")


@pyro_client.on_message(pyro_filters.command("freelist"))
async def free_list(client, message):
    if not await is_admin_pyro(client, message.chat.id, message.from_user.id):
        return await message.reply_text("🚫 Only admins can use this command.")

    users = free_users.get(message.chat.id, set())
    if not users:
        return await message.reply_text("❌ No free users in this chat.")

    text = "✅ Free users:\n"
    for uid in users:
        try:
            user = await client.get_users(uid)
            text += f"- <a href='tg://user?id={uid}'>{user.first_name}</a>\n"
        except Exception:
            text += f"- <a href='tg://user?id={uid}'>{uid}</a>\n"

    await message.reply_text(text, parse_mode=__import__("pyrogram.enums").enums.ParseMode.HTML)






# === ENFORCE STICKER & GIF RESTRICTION ===
@pyro_client.on_message(pyro_filters.sticker | pyro_filters.animation)
async def block_unfree_media(client, message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    # ✅ Allow admins and bots always
    if message.from_user.is_bot or await is_admin_pyro(client, chat_id, user_id):
        return

    # 🚫 Default = unfree, only free_users allowed
    if user_id not in free_users.get(chat_id, set()):
        try:
            await message.delete()
        except Exception:
            pass



load_lock_state()


def main():
    

    tclient.start()
    logger.info("Telethon userbot started")



    # Build the python-telegram-bot application
    application = ApplicationBuilder().token(BOT_TOKEN).concurrent_updates(True).build()

    # Register all handlers here (already present in your code)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CommandHandler("warn", warn))
    application.add_handler(CommandHandler("del", delmsg))
    application.add_handler(CommandHandler("ban", ban))
    application.add_handler(CommandHandler("unban", unban))
    application.add_handler(CommandHandler("admins", admins))
    application.add_handler(CommandHandler("promote", promote))
    application.add_handler(CommandHandler("demote", demote))
    application.add_handler(CommandHandler("addblacklist", addblacklist))
    application.add_handler(CommandHandler("unblacklist", unblacklist))
    application.add_handler(CommandHandler("blacklist", showblacklist))
    application.add_handler(CommandHandler("approve", approve))
    application.add_handler(CommandHandler("unapprove", unapprove))
    application.add_handler(CommandHandler("purge", purge))
    application.add_handler(CommandHandler("mute", mute))
    application.add_handler(CommandHandler("unmute", unmute))
    application.add_handler(CommandHandler("id", user_id))
    application.add_handler(CommandHandler("kick", kick))
    application.add_handler(CommandHandler("tmute", tmute))
    application.add_handler(CommandHandler("kickme", kickme))
    application.add_handler(CommandHandler("nightmode", nightmode_command))
    application.add_handler(CommandHandler("tagall", tagall))
    application.add_handler(CommandHandler("stop", stop_tagall))
    application.add_handler(CommandHandler("pin", pin))
    application.add_handler(CommandHandler("unpin", unpin))
    application.add_handler(CommandHandler("waifu", waifu_command))
    application.add_handler(CommandHandler("setflood", setflood))
    application.add_handler(CommandHandler("setfloodmode", setfloodmode))
    application.add_handler(CommandHandler("blacklistmode", set_blacklist_mode))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("unfilter", unfilter))
    application.add_handler(CommandHandler("filter", add_filter_command))
    application.add_handler(CommandHandler("filters", list_filters_command))
    application.add_handler(MessageHandler(ptb_filters.TEXT & ~ptb_filters.COMMAND, filter_trigger_handler))
    application.add_handler(CommandHandler("anime", anime_command))
    application.add_handler(CommandHandler("q", quote_command))
    application.add_handler(CommandHandler("lock", lock_command))
    application.add_handler(CommandHandler("unlock", unlock_command))
    application.add_handler(CommandHandler("locks", locks_command))
    application.add_handler(CommandHandler("kang", kang))
    application.add_handler(MessageHandler(ptb_filters.ALL & ~ptb_filters.COMMAND, message_handler), group=1)
    application.add_handler(CommandHandler("afk", afk_command))
    application.add_handler(CommandHandler("getsticker", getsticker))
    application.add_handler(CommandHandler("tr", translate_command))
    application.add_handler(CommandHandler("translate", translate_list))
    application.add_handler(CommandHandler("pp", pp_command))
    application.add_handler(CommandHandler("calc", calc_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("anime", anime_command))
    application.add_handler(CallbackQueryHandler(anime_callback, pattern="^anime_"))
    application.add_handler(CommandHandler("ud", ud_command))
    application.add_handler(CommandHandler("nightmode", nightmode_command))
    application.add_handler(CallbackQueryHandler(nightmode_callback, pattern=r"^nightmode_(on|off):-?\d+$"))
    application.add_handler(CommandHandler("resetwarns", resetwarns))
    application.add_handler(CommandHandler("rmwarn", rmwarn))
    application.add_handler(CommandHandler("welcome", welcome_toggle))
    application.add_handler(CommandHandler("setwelcome", set_welcome))
    application.add_handler(MessageHandler(ptb_filters.StatusUpdate.NEW_CHAT_MEMBERS, greet_new_member))
    application.add_handler(CommandHandler("unapproveall", unapproveall))
    application.add_handler(CommandHandler("approved", approved))
    application.add_handler(CommandHandler("unblacklistall", unblacklistall))
    application.add_handler(CommandHandler("warns", warns))
    application.add_handler(CommandHandler("goodbye", goodbye_toggle))
    application.add_handler(CommandHandler("setgoodbye", set_goodbye))
    application.add_handler(MessageHandler(ptb_filters.StatusUpdate.LEFT_CHAT_MEMBER, send_goodbye))
    application.add_handler(CommandHandler("when", when))
    application.add_handler(CommandHandler("captcha", captcha_toggle))
    application.add_handler(CallbackQueryHandler(captcha_pick, pattern=r"^capans:"))
    application.add_handler(CommandHandler("send", send_command))
    application.add_handler(CommandHandler("character", character_command))
    application.add_handler(CallbackQueryHandler(character_callback, pattern="^char_"))
    application.add_handler(CommandHandler("unfilterall", unfilterall))
    application.add_handler(CallbackQueryHandler(blacklist_callback, pattern=r"^bl_"))
    application.add_handler(CommandHandler("editdelete", editdelete_command))
    # Register handler specifically for edited messages using update types
    application.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, edited_message_handler, block=False), group=2)
    # Run periodic unmute task
    application.job_queue.run_repeating(lambda ctx: asyncio.create_task(unmute_expired_task(application)), interval=5, first=5)

    # Start Pyrogram client as background async task
    loop = asyncio.get_event_loop()
    loop.create_task(start_pyro())

    logger.info("Bot started polling")
    application.run_polling()


if __name__ == "__main__":
    main()
