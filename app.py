import os
import json
from flask import Flask, request, jsonify, render_template_string, session
import restaurant_agent

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "spicensavor-secret-token-key-2026")

# --- ✨ Luxury Next-Gen AI Dining Concierge (Full Aesthetic Redesign) ---
CHAT_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Spice & Savor Bistro — Next-Gen AI Concierge</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-dark: #090d16;
    --panel-bg: rgba(15, 23, 42, 0.75);
    --panel-border: rgba(255, 255, 255, 0.08);
    --card-bg: rgba(30, 41, 59, 0.6);
    --card-border: rgba(255, 255, 255, 0.07);
    --gold: #f59e0b;
    --gold-glow: rgba(245, 158, 11, 0.28);
    --gold-light: #fbbf24;
    --emerald: #10b981;
    --emerald-glow: rgba(16, 185, 129, 0.25);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --user-gradient: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    --bot-gradient: rgba(26, 35, 53, 0.85);
    --radius-xl: 20px;
    --radius-lg: 14px;
    --radius-md: 10px;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    -webkit-tap-highlight-color: transparent;
}

body {
    background-color: var(--bg-dark);
    background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(245, 158, 11, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
    background-attachment: fixed;
    color: var(--text-primary);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 18px;
    overflow-x: hidden;
}

/* Ambient Floating Lights */
.ambient-sphere {
    position: fixed;
    border-radius: 50%;
    filter: blur(140px);
    pointer-events: none;
    z-index: 0;
    opacity: 0.35;
}
.sphere-1 { width: 500px; height: 500px; top: -100px; left: -100px; background: #2563eb; }
.sphere-2 { width: 450px; height: 450px; bottom: -80px; right: -80px; background: #d97706; }

/* Main Master Grid Container */
.master-layout {
    position: relative;
    z-index: 10;
    width: 100%;
    max-width: 1400px;
    height: 94vh;
    max-height: 920px;
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 20px;
}

/* Glass Panel Utility */
.glass-panel {
    background: var(--panel-bg);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid var(--panel-border);
    border-radius: var(--radius-xl);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.04);
    overflow: hidden;
}

/* ==================== LEFT SIDEBAR: MENU & SHOWCASE ==================== */
.sidebar-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.restaurant-hero {
    padding: 24px;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.4) 100%);
    border-bottom: 1px solid var(--panel-border);
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: var(--gold-light);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
}

.restaurant-title {
    font-family: 'Outfit', sans-serif;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 60%, var(--gold-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.restaurant-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
    font-size: 12.5px;
    color: var(--text-secondary);
}
.meta-pill {
    display: flex;
    align-items: center;
    gap: 5px;
}
.live-dot {
    width: 7px;
    height: 7px;
    background: var(--emerald);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--emerald);
    animation: pulseGlow 2s infinite;
}
@keyframes pulseGlow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* Sidebar Fast Action Shortcuts */
.sidebar-shortcuts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    padding: 16px 24px;
    border-bottom: 1px solid var(--panel-border);
}
.shortcut-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--card-border);
    padding: 12px 14px;
    border-radius: var(--radius-lg);
    cursor: pointer;
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    gap: 10px;
}
.shortcut-card:hover {
    background: rgba(245, 158, 11, 0.12);
    border-color: var(--gold);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}
.shortcut-icon {
    font-size: 20px;
}
.shortcut-text strong {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
}
.shortcut-text span {
    font-size: 11px;
    color: var(--text-muted);
}

/* Live Menu Browser Tabs in Sidebar */
.menu-browser {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 16px 20px;
}
.browser-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.browser-header h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.2px;
}
.category-pills {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 8px;
    margin-bottom: 12px;
    scrollbar-width: none;
}
.category-pills::-webkit-scrollbar { display: none; }
.cat-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--card-border);
    color: var(--text-secondary);
    font-size: 11.5px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
    white-space: nowrap;
    cursor: pointer;
    transition: all 0.2s;
}
.cat-btn:hover, .cat-btn.active {
    background: var(--gold);
    color: #000;
    border-color: var(--gold);
}

.sidebar-dish-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-right: 4px;
}
.sidebar-dish-list::-webkit-scrollbar { width: 4px; }
.sidebar-dish-list::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }

.side-dish-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 10px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s;
    cursor: pointer;
}
.side-dish-item:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(245, 158, 11, 0.4);
    transform: translateX(3px);
}
.side-dish-left {
    display: flex;
    align-items: center;
    gap: 10px;
    overflow: hidden;
}
.side-dish-emoji {
    font-size: 24px;
    flex-shrink: 0;
}
.side-dish-name {
    font-size: 13px;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.side-dish-sub {
    font-size: 11px;
    color: var(--text-muted);
}
.side-dish-action {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}
.side-dish-price {
    font-size: 13px;
    font-weight: 700;
    color: var(--gold-light);
}
.side-add-btn {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid var(--gold);
    color: var(--gold-light);
    width: 26px;
    height: 26px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
}
.side-add-btn:hover {
    background: var(--gold);
    color: #000;
}


/* ==================== RIGHT PANEL: CHAT CONCIERGE ==================== */
.chat-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
}

/* Chat Header */
.chat-header {
    padding: 18px 24px;
    background: rgba(15, 23, 42, 0.85);
    border-bottom: 1px solid var(--panel-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chat-concierge-info {
    display: flex;
    align-items: center;
    gap: 12px;
}
.concierge-avatar {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35);
}
.concierge-title h2 {
    font-family: 'Outfit', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #fff;
}
.concierge-title p {
    font-size: 12px;
    color: var(--emerald);
    display: flex;
    align-items: center;
    gap: 5px;
}

.chat-controls {
    display: flex;
    align-items: center;
    gap: 10px;
}
.control-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--card-border);
    color: var(--text-primary);
    padding: 8px 14px;
    border-radius: var(--radius-md);
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
}
.control-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-1px);
}
.cart-pill {
    background: var(--gold);
    color: #000;
    font-weight: 800;
    font-size: 11px;
    padding: 2px 7px;
    border-radius: 20px;
}

/* Quick Prompt Horizontal Scrollbar */
.prompts-ribbon {
    display: flex;
    gap: 8px;
    padding: 12px 24px;
    background: rgba(10, 15, 28, 0.5);
    border-bottom: 1px solid var(--panel-border);
    overflow-x: auto;
    scrollbar-width: none;
}
.prompts-ribbon::-webkit-scrollbar { display: none; }
.prompt-chip {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.07);
    color: #cbd5e1;
    font-size: 12px;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 20px;
    white-space: nowrap;
    cursor: pointer;
    transition: all 0.2s;
}
.prompt-chip:hover {
    background: rgba(245, 158, 11, 0.15);
    border-color: var(--gold);
    color: var(--gold-light);
    transform: translateY(-1px);
}

/* Messages Feed */
.messages-feed {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 18px;
}
.messages-feed::-webkit-scrollbar { width: 5px; }
.messages-feed::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 4px; }

.message-unit {
    display: flex;
    gap: 12px;
    max-width: 82%;
    animation: fadeInSlide 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes fadeInSlide {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.message-unit.user {
    align-self: flex-end;
    flex-direction: row-reverse;
}
.message-unit.bot {
    align-self: flex-start;
}

.msg-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    flex-shrink: 0;
}
.bot-icon { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); }
.user-icon { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }

.bubble {
    padding: 15px 18px;
    border-radius: 18px;
    font-size: 14.5px;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
.user .bubble {
    background: var(--user-gradient);
    border-bottom-right-radius: 4px;
    color: #fff;
}
.bot .bubble {
    background: var(--bot-gradient);
    border-bottom-left-radius: 4px;
    border: 1px solid var(--card-border);
    color: #e2e8f0;
}

.bubble strong { color: var(--gold-light); }
.user .bubble strong { color: #fff; }

/* Dynamic Rich Action Cards inside Chat */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    margin-top: 14px;
    width: 100%;
}
.interactive-dish {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.25s ease;
}
.interactive-dish:hover {
    border-color: var(--gold);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.dish-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}
.dish-icon-large { font-size: 32px; }
.diet-tag {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 6px;
    text-transform: uppercase;
}
.tag-veg { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.tag-nonveg { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

.dish-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
}
.dish-summary {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.45;
    margin-bottom: 12px;
}
.dish-bottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 10px;
    margin-top: auto;
}
.dish-cost {
    font-size: 16px;
    font-weight: 800;
    color: var(--gold-light);
}
.order-add-btn {
    background: var(--gold);
    color: #000;
    font-weight: 700;
    font-size: 12px;
    padding: 6px 14px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
}
.order-add-btn:hover {
    background: #d97706;
    transform: scale(1.04);
}

/* Gold Reservation Ticket */
.res-ticket {
    margin-top: 14px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px dashed var(--gold);
    border-radius: var(--radius-lg);
    padding: 16px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.res-ticket-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 10px;
    margin-bottom: 14px;
}
.res-ticket-head strong { font-size: 15px; color: #fff; }
.res-badge {
    background: rgba(245, 158, 11, 0.18);
    color: var(--gold-light);
    font-size: 12px;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 6px;
}
.res-matrix {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    font-size: 13px;
}
.res-matrix div span { display: block; font-size: 11px; color: var(--text-muted); }
.res-matrix div strong { font-size: 14px; color: #fff; }

/* Real-Time Order Stepper */
.order-progress-card {
    margin-top: 14px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 18px;
}
.stepper-track {
    display: flex;
    justify-content: space-between;
    position: relative;
    margin: 22px 0 12px 0;
}
.stepper-track::before {
    content: '';
    position: absolute;
    top: 14px;
    left: 10%;
    right: 10%;
    height: 3px;
    background: rgba(255, 255, 255, 0.1);
    z-index: 1;
}
.step-node {
    position: relative;
    z-index: 2;
    text-align: center;
    flex: 1;
}
.node-circle {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #1e293b;
    border: 2px solid rgba(255, 255, 255, 0.2);
    color: #94a3b8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    margin: 0 auto 6px;
}
.step-node.active .node-circle {
    background: var(--gold);
    border-color: var(--gold);
    color: #000;
    font-weight: 800;
    box-shadow: 0 0 14px var(--gold-glow);
}
.step-node.completed .node-circle {
    background: var(--emerald);
    border-color: var(--emerald);
    color: #fff;
}
.node-text { font-size: 11px; color: var(--text-muted); }
.step-node.active .node-text { color: var(--gold-light); font-weight: 700; }

/* Typing Visual */
.typing-box {
    display: flex;
    gap: 5px;
    padding: 14px 20px;
    background: var(--bot-gradient);
    border-radius: 18px;
    border: 1px solid var(--card-border);
    width: fit-content;
}
.t-dot {
    width: 6px;
    height: 6px;
    background: #94a3b8;
    border-radius: 50%;
    animation: tBounce 1.4s infinite ease-in-out both;
}
.t-dot:nth-child(1) { animation-delay: -0.32s; }
.t-dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes tBounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* Chat Input Bar */
.chat-composer {
    padding: 18px 24px;
    background: rgba(15, 23, 42, 0.95);
    border-top: 1px solid var(--panel-border);
}
.composer-form {
    display: flex;
    gap: 12px;
    align-items: center;
}
.input-holder {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
}
#chat-input {
    width: 100%;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #fff;
    padding: 15px 48px 15px 20px;
    border-radius: var(--radius-lg);
    font-size: 14.5px;
    outline: none;
    transition: all 0.2s;
}
#chat-input:focus {
    border-color: var(--gold);
    background: rgba(30, 41, 59, 0.95);
    box-shadow: 0 0 0 3px var(--gold-glow);
}
.speech-btn {
    position: absolute;
    right: 12px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 19px;
    cursor: pointer;
    padding: 6px;
    border-radius: 8px;
    transition: all 0.2s;
}
.speech-btn:hover { color: var(--gold-light); }
.speech-btn.active-mic {
    color: #ef4444;
    animation: pulseGlow 1s infinite;
}

.send-trigger {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: #000;
    border: none;
    width: 50px;
    height: 50px;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    cursor: pointer;
    transition: all 0.25s;
    font-weight: 800;
    flex-shrink: 0;
    box-shadow: 0 4px 16px var(--gold-glow);
}
.send-trigger:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.45);
}

/* ==================== CART DRAWER ==================== */
.drawer-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(6px);
    z-index: 100;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
}
.drawer-backdrop.active { opacity: 1; pointer-events: auto; }

.cart-tray {
    position: fixed;
    top: 0;
    right: -450px;
    width: 400px;
    max-width: 90%;
    height: 100%;
    background: #0d1322;
    border-left: 1px solid var(--panel-border);
    z-index: 101;
    transition: right 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    flex-direction: column;
    box-shadow: -15px 0 50px rgba(0, 0, 0, 0.7);
}
.cart-tray.active { right: 0; }

.tray-header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--panel-border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.tray-header h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 19px;
    font-weight: 700;
    color: #fff;
}
.tray-close {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 20px;
    cursor: pointer;
}

.tray-items {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.tray-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 12px;
}
.tray-row-info {
    display: flex;
    align-items: center;
    gap: 10px;
}
.tray-row-info span { font-size: 24px; }
.tray-row-name { font-size: 13.5px; font-weight: 600; color: #fff; }
.tray-row-price { font-size: 12px; color: var(--gold-light); }

.counter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.8);
    border-radius: 8px;
    padding: 2px 6px;
}
.counter-btn {
    background: transparent;
    border: none;
    color: #fff;
    font-size: 15px;
    cursor: pointer;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.tray-footer {
    padding: 20px 24px;
    background: rgba(15, 23, 42, 0.95);
    border-top: 1px solid var(--panel-border);
}
.calc-line {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.calc-line.grand-total {
    font-size: 17px;
    font-weight: 800;
    color: #fff;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}
.voucher-form {
    display: flex;
    gap: 8px;
    margin: 12px 0;
}
.voucher-input {
    flex: 1;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid var(--card-border);
    color: #fff;
    padding: 9px 12px;
    border-radius: 8px;
    font-size: 12px;
    text-transform: uppercase;
}
.voucher-btn {
    background: rgba(255, 255, 255, 0.1);
    border: none;
    color: #fff;
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
}
.checkout-action {
    width: 100%;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #fff;
    border: none;
    padding: 13px;
    border-radius: var(--radius-md);
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    margin-top: 10px;
    transition: all 0.2s;
}
.checkout-action:hover {
    box-shadow: 0 4px 20px var(--emerald-glow);
    transform: translateY(-1px);
}

/* ==================== MODALS ==================== */
.modal-cover {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.75);
    backdrop-filter: blur(10px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 200;
    padding: 16px;
}
.modal-cover.active { display: flex; }
.modal-window {
    background: #0f172a;
    border: 1px solid var(--panel-border);
    border-radius: var(--radius-xl);
    width: 100%;
    max-width: 440px;
    padding: 26px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.8);
    animation: popIn 0.25s ease;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.94); }
    to { opacity: 1; transform: scale(1); }
}
.modal-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.modal-top h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #fff;
}
.field-item { margin-bottom: 14px; }
.field-item label {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 6px;
    font-weight: 500;
}
.field-control {
    width: 100%;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid var(--card-border);
    color: #fff;
    padding: 11px 14px;
    border-radius: var(--radius-md);
    font-size: 14px;
    outline: none;
}
.field-control:focus { border-color: var(--gold); }
.submit-btn-full {
    width: 100%;
    background: var(--gold);
    color: #000;
    font-weight: 800;
    border: none;
    padding: 13px;
    border-radius: var(--radius-md);
    font-size: 14.5px;
    cursor: pointer;
    margin-top: 8px;
    transition: all 0.2s;
}
.submit-btn-full:hover {
    background: #d97706;
}

/* ==================== RESPONSIVE DESIGN ==================== */
@media (max-width: 960px) {
    .master-layout {
        grid-template-columns: 1fr;
        height: 100vh;
        max-height: 100vh;
    }
    .sidebar-panel { display: none; }
    body { padding: 0; }
    .glass-panel { border-radius: 0; border: none; }
}
</style>
</head>
<body>

<div class="ambient-sphere sphere-1"></div>
<div class="ambient-sphere sphere-2"></div>

<div class="master-layout">
    <!-- ==================== LEFT SIDEBAR: MENU & SHOWCASE ==================== -->
    <aside class="glass-panel sidebar-panel">
        <!-- Restaurant Hero Branding -->
        <div class="restaurant-hero">
            <div class="brand-badge">⭐ MICHELIN INSPIRED &bull; LONDON</div>
            <h1 class="restaurant-title">Spice & Savor Bistro</h1>
            <div class="restaurant-meta">
                <div class="meta-pill">
                    <span class="live-dot"></span>
                    <span style="color: #fff; font-weight: 600;">Open Today</span>
                </div>
                <span>&bull;</span>
                <span>11:00 AM - 11:00 PM</span>
                <span>&bull;</span>
                <span>⭐ 4.9 (1.2k)</span>
            </div>
        </div>

        <!-- Quick Action Cards -->
        <div class="sidebar-shortcuts">
            <div class="shortcut-card" onclick="openBookingModal()">
                <span class="shortcut-icon">📅</span>
                <div class="shortcut-text">
                    <strong>Book Table</strong>
                    <span>Instant Seating</span>
                </div>
            </div>
            <div class="shortcut-card" onclick="sendPrompt('What active discount coupons do you have?')">
                <span class="shortcut-icon">🎁</span>
                <div class="shortcut-text">
                    <strong>Offers</strong>
                    <span>Save up to 20%</span>
                </div>
            </div>
        </div>

        <!-- Interactive Live Menu Browser -->
        <div class="menu-browser">
            <div class="browser-header">
                <h3>🍽️ Popular Signatures</h3>
                <span style="font-size: 11px; color: var(--text-muted);">Click to Add</span>
            </div>

            <div class="category-pills" id="category-pills">
                <button class="cat-btn active" onclick="filterCategory('All')">All</button>
                <button class="cat-btn" onclick="filterCategory('Chef\'s Specials')">Specials</button>
                <button class="cat-btn" onclick="filterCategory('Starters')">Starters</button>
                <button class="cat-btn" onclick="filterCategory('North Indian')">North Indian</button>
                <button class="cat-btn" onclick="filterCategory('Italian & Continental')">Italian</button>
                <button class="cat-btn" onclick="filterCategory('Desserts')">Desserts</button>
            </div>

            <div class="sidebar-dish-list" id="sidebar-dishes">
                <!-- Dynamically loaded dishes -->
            </div>
        </div>
    </aside>

    <!-- ==================== RIGHT PANEL: AI CONCIERGE CHAT ==================== -->
    <main class="glass-panel chat-panel">
        <!-- Chat Header -->
        <header class="chat-header">
            <div class="chat-concierge-info">
                <div class="concierge-avatar">🤖</div>
                <div class="concierge-title">
                    <h2>Dining Concierge AI</h2>
                    <p><span class="live-dot"></span> Powered by Google Gemini</p>
                </div>
            </div>
            <div class="chat-controls">
                <button class="control-btn" id="tts-btn" onclick="toggleSpeechAudio()">
                    <span id="tts-state-icon">🔇</span> Voice Off
                </button>
                <button class="control-btn" onclick="toggleCartDrawer()">
                    🛒 Cart <span class="cart-pill" id="cart-pill-count">0</span>
                </button>
            </div>
        </header>

        <!-- Quick Prompt Chips -->
        <div class="prompts-ribbon">
            <button class="prompt-chip" onclick="sendPrompt('Show me today\'s Chef Specials')">⭐ Chef's Specials</button>
            <button class="prompt-chip" onclick="sendPrompt('Do you have vegan dishes?')">🌱 Vegan Selections</button>
            <button class="prompt-chip" onclick="sendPrompt('Show vegetarian dishes under £12')">💰 Veg Under £12</button>
            <button class="prompt-chip" onclick="sendPrompt('I want to book a table for 4 tonight at 8 PM under Prabh')">📅 Book Table for 4</button>
            <button class="prompt-chip" onclick="sendPrompt('What active discount coupons do you have?')">🎁 Active Offers</button>
            <button class="prompt-chip" onclick="sendPrompt('Track order')">📦 Track Order</button>
            <button class="prompt-chip" onclick="sendPrompt('Where are you located and what are your opening hours?')">📍 Location & Hours</button>
        </div>

        <!-- Messages Feed -->
        <div class="messages-feed" id="chat-messages">
            <div class="message-unit bot">
                <div class="msg-icon bot-icon">🤖</div>
                <div class="bubble">
                    Hello and welcome to <strong>Spice & Savor Bistro</strong>! ✨<br>
                    I am your personal AI dining concierge. I can recommend gourmet pairings, check dietary requirements, reserve tables, and manage your food order in real time.
                    <p style="margin-top: 8px;">How may I delight your palate today?</p>
                </div>
            </div>
        </div>

        <!-- Chat Composer -->
        <div class="chat-composer">
            <form class="composer-form" onsubmit="submitUserPrompt(event)">
                <div class="input-holder">
                    <input type="text" id="chat-input" placeholder="Ask anything: 'Recommend spicy starters', 'Book table for 2', 'Track order'..." autocomplete="off" />
                    <button type="button" class="speech-btn" id="voice-rec-btn" onclick="triggerSpeechInput()" title="Voice Dictation">🎙️</button>
                </div>
                <button type="submit" class="send-trigger" title="Send Message">➤</button>
            </form>
        </div>
    </main>
</div>

<!-- ==================== CART DRAWER ==================== -->
<div class="drawer-backdrop" id="cart-backdrop" onclick="toggleCartDrawer()"></div>
<div class="cart-tray" id="cart-tray">
    <div class="tray-header">
        <h3>🛒 Your Dining Cart</h3>
        <button class="tray-close" onclick="toggleCartDrawer()">✕</button>
    </div>
    <div class="tray-items" id="tray-items-container">
        <p style="color: var(--text-muted); font-size: 13px; text-align: center; margin-top: 50px;">Your cart is currently empty. Explore our signatures!</p>
    </div>
    <div class="tray-footer" id="tray-footer-box" style="display: none;">
        <div class="voucher-form">
            <input type="text" class="voucher-input" id="voucher-code" placeholder="Promo code (SAVE20)" />
            <button class="voucher-btn" onclick="redeemCoupon()">Apply</button>
        </div>
        <div class="calc-line"><span>Subtotal:</span><strong id="tray-subtotal">£0.00</strong></div>
        <div class="calc-line"><span>Discount:</span><strong id="tray-discount" style="color: #10b981;">-£0.00</strong></div>
        <div class="calc-line"><span>Delivery Fee:</span><strong id="tray-delivery">£2.50</strong></div>
        <div class="calc-line grand-total"><span>Total:</span><strong id="tray-grand-total" style="color: #fbbf24;">£0.00</strong></div>
        <button class="checkout-action" onclick="openCheckoutDialog()">Proceed to Checkout 🚀</button>
    </div>
</div>

<!-- ==================== CHECKOUT MODAL ==================== -->
<div class="modal-cover" id="checkout-dialog">
    <div class="modal-window">
        <div class="modal-top">
            <h3>🛍️ Complete Order</h3>
            <button class="tray-close" onclick="closeCheckoutDialog()">✕</button>
        </div>
        <form onsubmit="finalizeOrder(event)">
            <div class="field-item">
                <label>Your Full Name</label>
                <input type="text" class="field-control" id="order-name" placeholder="John Doe" required />
            </div>
            <div class="field-item">
                <label>Contact Phone</label>
                <input type="tel" class="field-control" id="order-phone" placeholder="+44 7911 123456" required />
            </div>
            <div class="field-item">
                <label>Delivery Address / Table #</label>
                <input type="text" class="field-control" id="order-address" placeholder="1000 Bridge Avenue, London" required />
            </div>
            <div class="field-item">
                <label>Payment Method</label>
                <select class="field-control" id="order-payment">
                    <option value="Cash on Delivery">💵 Cash on Delivery / Pay at Counter</option>
                    <option value="Credit / Debit Card">💳 Credit / Debit Card (Online)</option>
                    <option value="Apple / Google Pay">📱 Apple Pay / Google Pay / UPI</option>
                </select>
            </div>
            <button type="submit" class="submit-btn-full">Confirm & Place Order 🎉</button>
        </form>
    </div>
</div>

<!-- ==================== TABLE RESERVATION MODAL ==================== -->
<div class="modal-cover" id="booking-dialog">
    <div class="modal-window">
        <div class="modal-top">
            <h3>📅 Reserve a Table</h3>
            <button class="tray-close" onclick="closeBookingModal()">✕</button>
        </div>
        <form onsubmit="finalizeBooking(event)">
            <div class="field-item">
                <label>Guest Name</label>
                <input type="text" class="field-control" id="res-name" placeholder="John Doe" required />
            </div>
            <div class="field-item">
                <label>Number of Guests</label>
                <input type="number" min="1" max="20" class="field-control" id="res-guests" value="2" required />
            </div>
            <div class="field-item">
                <label>Date</label>
                <input type="date" class="field-control" id="res-date" required />
            </div>
            <div class="field-item">
                <label>Time Slot</label>
                <select class="field-control" id="res-time">
                    <option value="12:30 PM">12:30 PM (Lunch)</option>
                    <option value="1:30 PM">1:30 PM (Lunch)</option>
                    <option value="6:00 PM">6:00 PM (Dinner)</option>
                    <option value="7:30 PM" selected>7:30 PM (Dinner)</option>
                    <option value="8:30 PM">8:30 PM (Dinner)</option>
                    <option value="9:30 PM">9:30 PM (Late Dinner)</option>
                </select>
            </div>
            <div class="field-item">
                <label>Special Requests (Optional)</label>
                <input type="text" class="field-control" id="res-requests" placeholder="Window booth, Anniversary, etc." />
            </div>
            <button type="submit" class="submit-btn-full">Confirm Reservation ✨</button>
        </form>
    </div>
</div>

<script>
let fullMenu = [];
let speechVoiceEnabled = false;
let speechRecognizer = null;
const feedContainer = document.getElementById('chat-messages');

// Voice Recognition
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    speechRecognizer = new SpeechRec();
    speechRecognizer.continuous = false;
    speechRecognizer.interimResults = false;
    speechRecognizer.lang = 'en-US';

    speechRecognizer.onresult = (evt) => {
        const text = evt.results[0][0].transcript;
        document.getElementById('chat-input').value = text;
        submitUserPrompt(new Event('submit'));
    };

    speechRecognizer.onend = () => {
        document.getElementById('voice-rec-btn').classList.remove('active-mic');
    };
}

function triggerSpeechInput() {
    if (!speechRecognizer) {
        alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
        return;
    }
    const btn = document.getElementById('voice-rec-btn');
    if (btn.classList.contains('active-mic')) {
        speechRecognizer.stop();
        btn.classList.remove('active-mic');
    } else {
        speechRecognizer.start();
        btn.classList.add('active-mic');
    }
}

function toggleSpeechAudio() {
    speechVoiceEnabled = !speechVoiceEnabled;
    const btn = document.getElementById('tts-btn');
    const icon = document.getElementById('tts-state-icon');
    if (speechVoiceEnabled) {
        icon.textContent = "🔊";
        btn.innerHTML = `<span>🔊</span> Voice On`;
        vocalizeText("Voice audio enabled.");
    } else {
        icon.textContent = "🔇";
        btn.innerHTML = `<span>🔇</span> Voice Off`;
        if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
}

function vocalizeText(text) {
    if (!speechVoiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const clean = text.replace(/[*#_`]/g, '').replace(/\[.*?\]\(.*?\)/g, '');
    const utter = new SpeechSynthesisUtterance(clean);
    utter.rate = 1.05;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
}

function sendPrompt(text) {
    document.getElementById('chat-input').value = text;
    submitUserPrompt(new Event('submit'));
}

async function submitUserPrompt(e) {
    if (e && e.preventDefault) e.preventDefault();
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    pushUserMessage(msg);
    input.value = '';

    const typingEl = showTypingIndicator();

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        const data = await resp.json();
        typingEl.remove();

        pushBotMessage(data.response, data.ui_card);
        vocalizeText(data.response);

        syncCart();
    } catch (err) {
        typingEl.remove();
        pushBotMessage("⚠️ Connection error. Please try again.");
    }
}

function pushUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-unit user';
    row.innerHTML = `
        <div class="msg-icon user-icon">👤</div>
        <div class="bubble">${escapeHtml(text)}</div>
    `;
    feedContainer.appendChild(row);
    scrollFeed();
}

function pushBotMessage(text, uiCard = null) {
    const row = document.createElement('div');
    row.className = 'message-unit bot';

    let cardMarkup = '';
    if (uiCard) {
        if (uiCard.type === 'dish_list') cardMarkup = buildDishCards(uiCard.items);
        else if (uiCard.type === 'reservation_card') cardMarkup = buildReservationTicket(uiCard.data);
        else if (uiCard.type === 'order_tracking') cardMarkup = buildOrderProgress(uiCard.data);
        else if (uiCard.type === 'promotions') cardMarkup = buildPromotions(uiCard.promotions);
    }

    row.innerHTML = `
        <div class="msg-icon bot-icon">🤖</div>
        <div class="bubble">
            <div>${formatRichText(text)}</div>
            ${cardMarkup}
        </div>
    `;
    feedContainer.appendChild(row);
    scrollFeed();
}

function buildDishCards(items) {
    if (!items || !items.length) return '';
    return `
        <div class="cards-grid">
            ${items.map(item => `
                <div class="interactive-dish">
                    <div>
                        <div class="dish-top">
                            <span class="dish-icon-large">${item.emoji || '🥘'}</span>
                            <span class="diet-tag ${item.veg ? 'tag-veg' : 'tag-nonveg'}">
                                ${item.vegan ? '🌱 Vegan' : (item.veg ? '🧀 Veg' : '🍗 Non-Veg')}
                            </span>
                        </div>
                        <div class="dish-heading">${escapeHtml(item.name)}</div>
                        <div class="dish-summary">${escapeHtml(item.description)}</div>
                    </div>
                    <div class="dish-bottom">
                        <div class="dish-cost">£${item.price.toFixed(2)}</div>
                        <button class="order-add-btn" onclick="addItemToCart('${item.id}', '${escapeHtml(item.name)}')">
                            + Add
                        </button>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function buildReservationTicket(booking) {
    return `
        <div class="res-ticket">
            <div class="res-ticket-head">
                <strong>Table Reservation Confirmed</strong>
                <span class="res-badge">#${booking.reservation_id}</span>
            </div>
            <div class="res-matrix">
                <div><span>PARTY SIZE</span><strong>👥 ${booking.party_size} Guests</strong></div>
                <div><span>TIME SLOT</span><strong>⏰ ${booking.time}</strong></div>
                <div><span>DATE</span><strong>📅 ${booking.date}</strong></div>
                <div><span>STATUS</span><strong style="color: #10b981;">${booking.status}</strong></div>
            </div>
        </div>
    `;
}

function buildOrderProgress(order) {
    const step = order.current_step || 1;
    return `
        <div class="order-progress-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <strong style="color: #fff; font-size: 15px;">Order #${order.order_id}</strong>
                <span style="font-size: 12px; color: #fbbf24; font-weight: 700;">Est: ${order.estimated_delivery}</span>
            </div>
            <div style="font-size: 13px; color: #94a3b8; margin-bottom: 10px;">${order.status_description}</div>
            <div class="stepper-track">
                <div class="step-node ${step >= 1 ? (step > 1 ? 'completed' : 'active') : ''}">
                    <div class="node-circle">1</div>
                    <div class="node-text">Confirmed</div>
                </div>
                <div class="step-node ${step >= 2 ? (step > 2 ? 'completed' : 'active') : ''}">
                    <div class="node-circle">2</div>
                    <div class="node-text">Cooking 👨‍🍳</div>
                </div>
                <div class="step-node ${step >= 3 ? (step > 3 ? 'completed' : 'active') : ''}">
                    <div class="node-circle">3</div>
                    <div class="node-text">On Way 🛵</div>
                </div>
                <div class="step-node ${step >= 4 ? 'completed' : ''}">
                    <div class="node-circle">4</div>
                    <div class="node-text">Delivered 🎉</div>
                </div>
            </div>
        </div>
    `;
}

function buildPromotions(promos) {
    return `
        <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">
            ${promos.map(p => `
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px dashed var(--gold); padding: 10px 14px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #fbbf24; font-size: 14px;">${p.code}</strong>
                        <div style="font-size: 12px; color: var(--text-secondary);">${p.description}</div>
                    </div>
                    <button class="order-add-btn" style="background: #fbbf24; color: #000;" onclick="copyCouponCode('${p.code}')">Copy</button>
                </div>
            `).join('')}
        </div>
    `;
}

function copyCouponCode(code) {
    navigator.clipboard.writeText(code);
    alert(`Copied voucher code ${code}! You can apply it in your Cart.`);
}

/* Cart Management */
async function addItemToCart(itemId, name) {
    try {
        const resp = await fetch('/api/cart/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_id: itemId, quantity: 1 })
        });
        const cart = await resp.json();
        renderCartUI(cart);
        toggleCartDrawer(true);
    } catch (e) { console.error(e); }
}

async function alterItemQty(itemId, delta) {
    try {
        const resp = await fetch('/api/cart/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_id: itemId, quantity: delta })
        });
        const cart = await resp.json();
        renderCartUI(cart);
    } catch (e) { console.error(e); }
}

async function redeemCoupon() {
    const code = document.getElementById('voucher-code').value.trim();
    if (!code) return;
    try {
        const resp = await fetch('/api/cart/coupon', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coupon: code })
        });
        const cart = await resp.json();
        renderCartUI(cart);
    } catch (e) { console.error(e); }
}

async function syncCart() {
    try {
        const resp = await fetch('/api/cart');
        const cart = await resp.json();
        renderCartUI(cart);
    } catch (e) { console.error(e); }
}

function renderCartUI(cart) {
    document.getElementById('cart-pill-count').textContent = cart.item_count || 0;
    const itemsEl = document.getElementById('tray-items-container');
    const footerEl = document.getElementById('tray-footer-box');

    if (!cart.items || cart.items.length === 0) {
        itemsEl.innerHTML = `<p style="color: var(--text-muted); font-size: 13px; text-align: center; margin-top: 50px;">Your cart is currently empty. Explore our signatures!</p>`;
        footerEl.style.display = 'none';
        return;
    }

    footerEl.style.display = 'block';
    itemsEl.innerHTML = cart.items.map(item => `
        <div class="tray-row">
            <div class="tray-row-info">
                <span>${item.emoji || '🍽️'}</span>
                <div>
                    <div class="tray-row-name">${escapeHtml(item.name)}</div>
                    <div class="tray-row-price">£${item.price.toFixed(2)}</div>
                </div>
            </div>
            <div class="counter-group">
                <button class="counter-btn" onclick="alterItemQty('${item.id}', -1)">-</button>
                <span style="font-size: 13px; font-weight: 700; min-width: 16px; text-align: center;">${item.quantity}</span>
                <button class="counter-btn" onclick="alterItemQty('${item.id}', 1)">+</button>
            </div>
        </div>
    `).join('');

    document.getElementById('tray-subtotal').textContent = `£${cart.subtotal.toFixed(2)}`;
    document.getElementById('tray-discount').textContent = `-£${cart.discount.toFixed(2)}`;
    document.getElementById('tray-delivery').textContent = `£${cart.delivery_fee.toFixed(2)}`;
    document.getElementById('tray-grand-total').textContent = `£${cart.total.toFixed(2)}`;
}

function toggleCartDrawer(forceOpen = false) {
    const tray = document.getElementById('cart-tray');
    const backdrop = document.getElementById('cart-backdrop');
    if (forceOpen || !tray.classList.contains('active')) {
        tray.classList.add('active');
        backdrop.classList.add('active');
    } else {
        tray.classList.remove('active');
        backdrop.classList.remove('active');
    }
}

/* Modals */
function openCheckoutDialog() {
    toggleCartDrawer();
    document.getElementById('checkout-dialog').classList.add('active');
}
function closeCheckoutDialog() {
    document.getElementById('checkout-dialog').classList.remove('active');
}

async function finalizeOrder(e) {
    e.preventDefault();
    const name = document.getElementById('order-name').value;
    const phone = document.getElementById('order-phone').value;
    const address = document.getElementById('order-address').value;
    const payment = document.getElementById('order-payment').value;

    try {
        const resp = await fetch('/api/order/place', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, phone, address, payment })
        });
        const order = await resp.json();
        closeCheckoutDialog();
        syncCart();

        pushBotMessage(`🎉 Thank you **${order.customer_name}**! Your order **#${order.order_id}** totaling **£${order.total.toFixed(2)}** is confirmed. Estimated delivery by **${order.estimated_delivery}**.`, {
            type: 'order_tracking',
            data: order
        });
    } catch (e) {
        alert("Failed to place order.");
    }
}

function openBookingModal() {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('res-date').value = today;
    document.getElementById('booking-dialog').classList.add('active');
}
function closeBookingModal() {
    document.getElementById('booking-dialog').classList.remove('active');
}

async function finalizeBooking(e) {
    e.preventDefault();
    const name = document.getElementById('res-name').value;
    const guests = document.getElementById('res-guests').value;
    const date = document.getElementById('res-date').value;
    const time = document.getElementById('res-time').value;
    const requests = document.getElementById('res-requests').value;

    try {
        const resp = await fetch('/api/book', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, guests, date, time, requests })
        });
        const booking = await resp.json();
        closeBookingModal();

        pushBotMessage(`🎉 Wonderful! Table reservation confirmed for **${booking.party_size} guests** on **${booking.date}** at **${booking.time}**. Your booking ID is **#${booking.reservation_id}**.`, {
            type: 'reservation_card',
            data: booking
        });
    } catch (e) {
        alert("Failed to reserve table.");
    }
}

/* Sidebar Menu Loading */
async function loadFullMenu() {
    try {
        const resp = await fetch('/api/menu');
        const data = await resp.json();
        fullMenu = data.menu || [];
        renderSidebarDishes(fullMenu);
    } catch (e) { console.error(e); }
}

function filterCategory(cat) {
    document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
    if (event && event.target) event.target.classList.add('active');

    if (cat === 'All') {
        renderSidebarDishes(fullMenu);
    } else {
        const filtered = fullMenu.filter(item => item.category.toLowerCase().includes(cat.toLowerCase()));
        renderSidebarDishes(filtered);
    }
}

function renderSidebarDishes(items) {
    const listEl = document.getElementById('sidebar-dishes');
    if (!items || !items.length) {
        listEl.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px;">No dishes found.</div>';
        return;
    }
    listEl.innerHTML = items.map(item => `
        <div class="side-dish-item" onclick="sendPrompt('Tell me more about ${escapeHtml(item.name)}')">
            <div class="side-dish-left">
                <span class="side-dish-emoji">${item.emoji || '🍲'}</span>
                <div>
                    <div class="side-dish-name">${escapeHtml(item.name)}</div>
                    <div class="side-dish-sub">${item.vegan ? '🌱 Vegan' : (item.veg ? '🧀 Veg' : '🍗 Non-Veg')} &bull; ${item.calories} kcal</div>
                </div>
            </div>
            <div class="side-dish-action">
                <span class="side-dish-price">£${item.price.toFixed(2)}</span>
                <button class="side-add-btn" onclick="event.stopPropagation(); addItemToCart('${item.id}', '${escapeHtml(item.name)}')">+</button>
            </div>
        </div>
    `).join('');
}

/* UI Helpers */
function showTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'message-unit bot';
    row.innerHTML = `
        <div class="msg-icon bot-icon">🤖</div>
        <div class="typing-box">
            <div class="t-dot"></div>
            <div class="t-dot"></div>
            <div class="t-dot"></div>
        </div>
    `;
    feedContainer.appendChild(row);
    scrollFeed();
    return row;
}

function scrollFeed() {
    feedContainer.scrollTop = feedContainer.scrollHeight;
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function formatRichText(txt) {
    return escapeHtml(txt)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; color: #fbbf24;">$1</code>')
        .replace(/\n/g, '<br>');
}

window.onload = () => {
    loadFullMenu();
    syncCart();
};
</script>
</body>
</html>
"""

# --- 🚀 REST API Endpoints ---

@app.route("/")
def index():
    return render_template_string(CHAT_HTML)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    message = data.get("message", "").strip()
    session_id = session.get("sid", "guest_session")
    if not message:
        return jsonify({"response": "Please enter a message."})

    result = restaurant_agent.handle_chat(message, session_id=session_id)
    return jsonify(result)

@app.route("/api/menu", methods=["GET"])
def api_menu():
    category = request.args.get("category", "")
    query = request.args.get("q", "")
    items = restaurant_agent.search_menu(query=query, category=category)
    return jsonify({"menu": items, "categories": restaurant_agent.CATEGORIES})

@app.route("/api/cart", methods=["GET"])
def api_get_cart():
    session_id = session.get("sid", "guest_session")
    return jsonify(restaurant_agent.get_cart_summary(session_id))

@app.route("/api/cart/update", methods=["POST"])
def api_cart_update():
    data = request.get_json(force=True) or {}
    item_id = data.get("item_id")
    quantity = int(data.get("quantity", 1))
    session_id = session.get("sid", "guest_session")

    restaurant_agent.manage_cart_add(session_id, item_id, quantity)
    return jsonify(restaurant_agent.get_cart_summary(session_id))

@app.route("/api/cart/coupon", methods=["POST"])
def api_cart_coupon():
    data = request.get_json(force=True) or {}
    coupon = data.get("coupon", "")
    session_id = session.get("sid", "guest_session")
    return jsonify(restaurant_agent.get_cart_summary(session_id, coupon_code=coupon))

@app.route("/api/book", methods=["POST"])
def api_book_table():
    data = request.get_json(force=True) or {}
    name = data.get("name", "Guest")
    guests = int(data.get("guests", 2))
    date = data.get("date", "Today")
    time_slot = data.get("time", "7:30 PM")
    requests = data.get("requests", "")

    booking = restaurant_agent.book_table(name, guests, date, time_slot, requests)
    return jsonify(booking)

@app.route("/api/order/place", methods=["POST"])
def api_place_order():
    data = request.get_json(force=True) or {}
    name = data.get("name", "Guest")
    phone = data.get("phone", "")
    address = data.get("address", "")
    payment = data.get("payment", "Cash on Delivery")
    session_id = session.get("sid", "guest_session")

    order = restaurant_agent.place_order(session_id, name, phone, address, payment_method=payment)
    return jsonify(order)

@app.route("/api/order/track/<order_id>", methods=["GET"])
def api_track_order(order_id):
    return jsonify(restaurant_agent.track_order(order_id))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
