import os
import json
import random
import re
import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Load Menu and Restaurant Data
MENU_FILE = os.path.join(os.path.dirname(__file__), "menu_data.json")
with open(MENU_FILE, "r", encoding="utf-8") as f:
    DATA = json.load(f)

RESTAURANT = DATA.get("restaurant", {})
MENU_ITEMS = DATA.get("menu", [])
CATEGORIES = DATA.get("categories", [])
PROMOTIONS = DATA.get("promotions", [])

# In-memory stores
CARTS: Dict[str, Dict[str, Any]] = {}
RESERVATIONS: Dict[str, Dict[str, Any]] = {}
ORDERS: Dict[str, Dict[str, Any]] = {}

# --- Helper Functions & Tools ---

def get_restaurant_info() -> Dict[str, Any]:
    """Get general restaurant details like hours, address, phone, and delivery policy."""
    return {
        "name": RESTAURANT.get("name"),
        "tagline": RESTAURANT.get("tagline"),
        "address": RESTAURANT.get("address"),
        "phone": RESTAURANT.get("phone"),
        "hours": RESTAURANT.get("opening_hours"),
        "delivery_time": RESTAURANT.get("delivery_time"),
        "currency": RESTAURANT.get("currency", "£"),
        "delivery_fee": RESTAURANT.get("delivery_fee", 2.50)
    }

def get_promotions() -> List[Dict[str, Any]]:
    """Retrieve active discount coupons and promotional offers."""
    return PROMOTIONS

def search_menu(
    query: str = "",
    category: str = "",
    max_price: Optional[float] = None,
    veg_only: bool = False,
    vegan_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Search the restaurant menu by keyword, category, dietary restrictions, or budget.
    """
    results = []
    q = query.lower().strip()
    cat = category.lower().strip()

    for item in MENU_ITEMS:
        # Category filter
        if cat and cat not in item.get("category", "").lower():
            continue
        # Veg / Vegan filter
        if veg_only and not item.get("veg", False):
            continue
        if vegan_only and not item.get("vegan", False):
            continue
        # Price filter
        if max_price is not None and item.get("price", 0.0) > float(max_price):
            continue
        # Keyword search across name, description, tags
        if q:
            match_name = q in item.get("name", "").lower()
            match_desc = q in item.get("description", "").lower()
            match_tags = any(q in tag.lower() for tag in item.get("tags", []))
            match_cat = q in item.get("category", "").lower()
            if not (match_name or match_desc or match_tags or match_cat):
                continue
        results.append(item)

    return results

def get_item_by_id(item_id: str) -> Optional[Dict[str, Any]]:
    """Find a menu item by its unique ID or name match."""
    for item in MENU_ITEMS:
        if item["id"] == item_id or item["name"].lower() == item_id.lower():
            return item
    return None

def manage_cart_add(session_id: str, item_id: str, quantity: int = 1) -> Dict[str, Any]:
    """Add a food item to the customer's cart."""
    item = get_item_by_id(item_id)
    if not item:
        # Try fuzzy name match
        matched = search_menu(query=item_id)
        if matched:
            item = matched[0]
        else:
            return {"error": f"Item '{item_id}' not found on the menu."}

    if session_id not in CARTS:
        CARTS[session_id] = {"items": {}, "coupon": None}

    cart = CARTS[session_id]
    curr_qty = cart["items"].get(item["id"], {}).get("quantity", 0)
    new_qty = curr_qty + quantity
    if new_qty <= 0:
        cart["items"].pop(item["id"], None)
    else:
        cart["items"][item["id"]] = {
            "id": item["id"],
            "name": item["name"],
            "price": item["price"],
            "emoji": item["emoji"],
            "quantity": new_qty,
            "veg": item["veg"]
        }

    return get_cart_summary(session_id)

def manage_cart_remove(session_id: str, item_id: str) -> Dict[str, Any]:
    """Remove an item completely from the cart."""
    if session_id in CARTS and item_id in CARTS[session_id]["items"]:
        del CARTS[session_id]["items"][item_id]
    return get_cart_summary(session_id)

def get_cart_summary(session_id: str, coupon_code: str = "") -> Dict[str, Any]:
    """Calculate subtotal, discount, delivery fee, and grand total of the cart."""
    cart = CARTS.get(session_id, {"items": {}, "coupon": None})
    if coupon_code:
        cart["coupon"] = coupon_code.upper().strip()

    items_list = list(cart["items"].values())
    subtotal = sum(item["price"] * item["quantity"] for item in items_list)
    discount = 0.0
    applied_coupon = None

    if cart.get("coupon"):
        c_code = cart["coupon"]
        for p in PROMOTIONS:
            if p["code"] == c_code:
                if subtotal >= p["min_spend"]:
                    discount = (subtotal * p["discount_percent"]) / 100.0
                    applied_coupon = p
                break

    delivery_fee = RESTAURANT.get("delivery_fee", 2.50) if items_list else 0.0
    total = max(0.0, subtotal - discount + delivery_fee)

    return {
        "items": items_list,
        "item_count": sum(item["quantity"] for item in items_list),
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "applied_coupon": applied_coupon,
        "delivery_fee": round(delivery_fee, 2),
        "total": round(total, 2),
        "currency": RESTAURANT.get("currency", "£")
    }

def clear_cart(session_id: str) -> Dict[str, str]:
    """Clear all items in the customer's cart."""
    if session_id in CARTS:
        CARTS[session_id] = {"items": {}, "coupon": None}
    return {"message": "Cart cleared successfully."}

def book_table(
    customer_name: str,
    party_size: int,
    date: str,
    time_slot: str,
    special_request: str = ""
) -> Dict[str, Any]:
    """
    Reserve a table at the restaurant.
    """
    res_id = f"RES-{random.randint(1000, 9999)}"
    reservation = {
        "reservation_id": res_id,
        "customer_name": customer_name or "Guest",
        "party_size": int(party_size) if party_size else 2,
        "date": date or "Today",
        "time": time_slot or "7:00 PM",
        "special_request": special_request or "None",
        "status": "Confirmed ✅",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    RESERVATIONS[res_id] = reservation
    return reservation

def place_order(
    session_id: str,
    customer_name: str,
    phone: str,
    address: str,
    payment_method: str = "Cash on Delivery"
) -> Dict[str, Any]:
    """
    Finalize and place the current cart as a real-time order.
    """
    cart = get_cart_summary(session_id)
    if not cart["items"]:
        return {"error": "Your cart is currently empty! Please add some dishes first."}

    order_id = f"ORD-{random.randint(1000, 9999)}"
    now = datetime.datetime.now()

    order = {
        "order_id": order_id,
        "customer_name": customer_name or "Valued Guest",
        "phone": phone or "N/A",
        "address": address or "Dine-in / Pickup",
        "payment_method": payment_method or "Cash on Delivery",
        "items": cart["items"],
        "subtotal": cart["subtotal"],
        "discount": cart["discount"],
        "delivery_fee": cart["delivery_fee"],
        "total": cart["total"],
        "currency": cart["currency"],
        "status": "Order Confirmed",
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "created_timestamp": now.timestamp(),
        "estimated_delivery": (now + datetime.timedelta(minutes=35)).strftime("%I:%M %p")
    }

    ORDERS[order_id] = order
    clear_cart(session_id)
    return order

def track_order(order_id: str) -> Dict[str, Any]:
    """
    Get live tracking status for an existing order.
    """
    clean_id = order_id.upper().strip()
    # Also support searching by number without prefix
    if not clean_id.startswith("ORD-") and clean_id.isdigit():
        clean_id = f"ORD-{clean_id}"

    order = ORDERS.get(clean_id)
    if not order:
        return {"error": f"Order #{clean_id} not found. Please check your order ID."}

    # Simulate realistic timeline based on elapsed minutes
    elapsed_seconds = datetime.datetime.now().timestamp() - order.get("created_timestamp", 0)
    elapsed_mins = elapsed_seconds / 60.0

    if elapsed_mins < 2:
        status = "Order Confirmed"
        step = 1
        description = "Kitchen has received your order and is reviewing it."
    elif elapsed_mins < 8:
        status = "Cooking in Kitchen 👨‍🍳"
        step = 2
        description = "Our chef is preparing your fresh meal with love."
    elif elapsed_mins < 18:
        status = "Out for Delivery 🛵"
        step = 3
        description = "Rider is on the way to your address."
    else:
        status = "Delivered 🎉"
        step = 4
        description = "Order has been delivered. Enjoy your meal!"

    order["status"] = status
    order["current_step"] = step
    order["status_description"] = description
    return order

# --- Agent System (Gemini LLM with Native Tool Calling & Smart Local Fallback) ---

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Define tool schemas for Gemini function calling
AGENT_TOOLS = [
    search_menu,
    get_restaurant_info,
    get_promotions,
    book_table,
    manage_cart_add,
    get_cart_summary,
    place_order,
    track_order
]

def run_gemini_agent(user_message: str, session_id: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run Google Gemini with native function calling."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        system_instruction = f"""
You are the AI Concierge for '{RESTAURANT.get('name')}', a premium restaurant.
Your tone is warm, polite, enthusiastic, and culinary-knowledgeable.
Address: {RESTAURANT.get('address')}
Opening Hours: {RESTAURANT.get('opening_hours')}
Phone: {RESTAURANT.get('phone')}

Capabilities:
1. Recommend dishes, search the menu, explain ingredients, dietary options (veg, vegan, gluten-free), and prices.
2. Book table reservations when requested using `book_table`.
3. Add food items to the customer's cart using `manage_cart_add` when they want to order.
4. Show promotions and discounts using `get_promotions`.
5. Check order status using `track_order`.
6. Provide accurate delivery details and hours.

Always use your tools when specific data, ordering, or booking actions are requested.
Current session ID: {session_id}
"""
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=AGENT_TOOLS,
            system_instruction=system_instruction
        )

        chat = model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(user_message)

        text_response = response.text or "How else can I assist you with your dining experience today?"

        # Extract any tool invocation data to augment UI cards
        ui_card = None
        # Check if we should attach dish cards or reservation cards
        if "order" in user_message.lower() or "menu" in user_message.lower() or "special" in user_message.lower() or "dishes" in user_message.lower():
            dishes = search_menu(query=user_message.split()[-1] if len(user_message.split()) > 1 else "")
            if not dishes:
                dishes = MENU_ITEMS[:4]
            ui_card = {"type": "dish_list", "items": dishes[:4]}

        return {
            "response": text_response,
            "ui_card": ui_card,
            "engine": "gemini-llm"
        }
    except Exception as e:
        print(f"[Gemini Agent Error: {e}] Falling back to intelligent local engine...")
        return run_local_agent(user_message, session_id)


def run_local_agent(user_message: str, session_id: str) -> Dict[str, Any]:
    """
    Intelligent built-in fallback agent that performs tool dispatching,
    natural language understanding, dietary filtering, reservation, and order flows.
    """
    msg = user_message.lower().strip()

    # 1. Table Reservation
    if any(k in msg for k in ["book", "reserve", "reservation", "table"]):
        # Extract party size if mentioned
        party_size = 2
        match_party = re.search(r'(\d+)\s*(?:people|person|guest|pax|seats?)', msg)
        if match_party:
            party_size = int(match_party.group(1))

        # Extract time if mentioned
        time_slot = "7:30 PM"
        match_time = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))', msg, re.IGNORECASE)
        if match_time:
            time_slot = match_time.group(1).upper()

        booking = book_table(
            customer_name="Guest",
            party_size=party_size,
            date="Tonight",
            time_slot=time_slot,
            special_request="Standard Table"
        )
        return {
            "response": f"🎉 Splendid! I've reserved a table for **{booking['party_size']} guests** at **{booking['time']}** under confirmation code **#{booking['reservation_id']}**. We look forward to hosting you!",
            "ui_card": {"type": "reservation_card", "data": booking},
            "engine": "local-agent"
        }

    # 2. Track Order
    if any(k in msg for k in ["track", "status of order", "where is my order", "order #", "ord-"]):
        match_id = re.search(r'(?:ord-)?(\d{4})', msg, re.IGNORECASE)
        if match_id:
            order_id = f"ORD-{match_id.group(1)}"
            tracking = track_order(order_id)
            if "error" not in tracking:
                return {
                    "response": f"📦 Here is the live status for **Order #{tracking['order_id']}**:",
                    "ui_card": {"type": "order_tracking", "data": tracking},
                    "engine": "local-agent"
                }
            else:
                return {"response": tracking["error"], "engine": "local-agent"}
        else:
            # Check most recent order
            if ORDERS:
                last_order = list(ORDERS.values())[-1]
                tracking = track_order(last_order["order_id"])
                return {
                    "response": f"📦 Found your recent **Order #{tracking['order_id']}**! Here is the live status:",
                    "ui_card": {"type": "order_tracking", "data": tracking},
                    "engine": "local-agent"
                }
            return {
                "response": "Please provide your Order ID (e.g. `ORD-1234`) so I can track it for you in real-time!",
                "engine": "local-agent"
            }

    # 3. Add to Cart / Ordering
    if any(k in msg for k in ["add to cart", "order", "i want", "buy", "get me"]):
        matched_items = []
        for item in MENU_ITEMS:
            if item["name"].lower() in msg or any(t in msg for t in item["tags"] if len(t) > 3):
                matched_items.append(item)

        if matched_items:
            chosen = matched_items[0]
            manage_cart_add(session_id, chosen["id"], quantity=1)
            cart = get_cart_summary(session_id)
            return {
                "response": f"🛒 Added **{chosen['name']}** ({RESTAURANT['currency']}{chosen['price']:.2f}) to your order! Your cart now has {cart['item_count']} item(s).",
                "ui_card": {"type": "cart_update", "cart": cart, "added_item": chosen},
                "engine": "local-agent"
            }

    # 4. View Cart
    if any(k in msg for k in ["view cart", "show cart", "my cart", "checkout", "bill", "my order"]):
        cart = get_cart_summary(session_id)
        if not cart["items"]:
            return {
                "response": "🛒 Your cart is currently empty. Would you like to explore our Chef's Specials or Starters?",
                "ui_card": {"type": "dish_list", "items": MENU_ITEMS[:3]},
                "engine": "local-agent"
            }
        return {
            "response": f"🛒 Here is your current order summary ({cart['item_count']} items) totaling **{cart['currency']}{cart['total']:.2f}**:",
            "ui_card": {"type": "cart_summary", "cart": cart},
            "engine": "local-agent"
        }

    # 5. Offers / Discounts / Promo Codes
    if any(k in msg for k in ["offer", "discount", "coupon", "promo", "deal"]):
        promos = get_promotions()
        promo_text = "🎉 **Active Special Offers:**\n" + "\n".join(
            f"• Use code **{p['code']}**: {p['description']}" for p in promos
        )
        return {
            "response": promo_text,
            "ui_card": {"type": "promotions", "promotions": promos},
            "engine": "local-agent"
        }

    # 6. Dietary & Menu Filtering (Veg, Vegan, Budget, Specific dishes)
    is_veg = "veg" in msg and "non-veg" not in msg and "non veg" not in msg
    is_vegan = "vegan" in msg
    price_match = re.search(r'under\s*£?\$?(\d+)', msg)
    max_price = float(price_match.group(1)) if price_match else None

    # Check for specific categories or items
    matched_category = next((c for c in CATEGORIES if c.lower() in msg), "")
    
    # Keyword search
    keywords = [w for w in msg.split() if w not in ["show", "me", "the", "what", "is", "have", "you", "got", "dishes", "food", "menu", "options", "do"]]
    query = " ".join(keywords)

    filtered_items = search_menu(
        query=query if not matched_category else "",
        category=matched_category,
        max_price=max_price,
        veg_only=is_veg,
        vegan_only=is_vegan
    )

    if filtered_items:
        prefix = "Here are our recommendations matching your taste:"
        if is_vegan: prefix = "🌱 Here are our fresh 100% plant-based Vegan options:"
        elif is_veg: prefix = "🧀 Here are our mouth-watering Vegetarian selections:"
        elif max_price: prefix = f"💰 Delicious dishes under {RESTAURANT['currency']}{max_price:.2f}:"

        return {
            "response": prefix,
            "ui_card": {"type": "dish_list", "items": filtered_items[:4]},
            "engine": "local-agent"
        }

    # 7. Today's Specials / Chef Specials
    if any(k in msg for k in ["special", "chef", "popular", "recommend", "best"]):
        specials = [item for item in MENU_ITEMS if item.get("category") == "Chef's Specials"]
        return {
            "response": "⭐ Here are our **Chef's Signature Specials** handcrafted for an unforgettable meal:",
            "ui_card": {"type": "dish_list", "items": specials},
            "engine": "local-agent"
        }

    # 8. Hours, Location, Contact
    if any(k in msg for k in ["hour", "time", "open", "close", "timing"]):
        return {
            "response": f"🕒 We are delighted to serve you daily from **{RESTAURANT['opening_hours']}**.",
            "engine": "local-agent"
        }
    if any(k in msg for k in ["location", "address", "where", "map", "directions"]):
        return {
            "response": f"📍 We are located at **{RESTAURANT['address']}**.\n📞 Call us: **{RESTAURANT['phone']}**",
            "engine": "local-agent"
        }
    if any(k in msg for k in ["delivery", "swiggy", "zomato", "takeaway", "home delivery"]):
        return {
            "response": f"🚗 Yes! We offer fast home delivery in **{RESTAURANT['delivery_time']}** with a standard delivery fee of {RESTAURANT['currency']}{RESTAURANT['delivery_fee']:.2f}. You can order directly here in the chat!",
            "engine": "local-agent"
        }

    # 9. Greetings & Farewell
    if any(k in msg for k in ["hi", "hello", "hey", "good morning", "good evening", "greetings"]):
        return {
            "response": f"Hello and welcome to **{RESTAURANT['name']}**! 🍽️\nI am your AI dining assistant. How may I assist you today? You can explore our menu, book a table, or track an existing order!",
            "ui_card": {"type": "dish_list", "items": MENU_ITEMS[:3]},
            "engine": "local-agent"
        }
    if any(k in msg for k in ["bye", "goodbye", "see you", "thank"]):
        return {
            "response": "It was our pleasure assisting you! Have a wonderful day and we hope to serve you soon. 😊",
            "engine": "local-agent"
        }

    # Default fallback
    sample_dishes = random.sample(MENU_ITEMS, min(3, len(MENU_ITEMS)))
    return {
        "response": "I can help you explore our menu, recommend dishes based on dietary preferences, book a table reservation, or place & track an order. What would you like to do?",
        "ui_card": {"type": "dish_list", "items": sample_dishes},
        "engine": "local-agent"
    }


def handle_chat(message: str, session_id: str = "guest_session") -> Dict[str, Any]:
    """Primary chat entrypoint that automatically chooses Gemini LLM or Local Agent."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
        return run_gemini_agent(message, session_id)
    else:
        return run_local_agent(message, session_id)
