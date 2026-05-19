// ============================================
// 1. Configuration & State
// ============================================
const messageHistory = [];

// ============================================
// 2. DOM References
// ============================================
const form = document.getElementById("chatForm");
const input = document.getElementById("queryInput");
const submitButton = document.getElementById("submitButton");
const conversation = document.getElementById("conversation");
const statusBadge = document.getElementById("statusBadge");
const chips = document.querySelectorAll(".prompt-chip");
const themeToggle = document.getElementById("themeToggle");
const clearChatBtn = document.getElementById("clearChat");

// ============================================
// 3. Theme Management
// ============================================
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    themeToggle.textContent = theme === "dark" ? "☀️" : "\u{1F319}";
    localStorage.setItem("theme", theme);
}

// Sync toggle button state with the theme set by inline flash-prevention script
(function initTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    themeToggle.textContent = current === "dark" ? "☀️" : "\u{1F319}";
})();

themeToggle.addEventListener("click", function () {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(current === "dark" ? "light" : "dark");
});

// Listen for OS-level preference changes (only applies when user hasn't set a preference)
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
    if (!localStorage.getItem("theme")) {
        applyTheme(e.matches ? "dark" : "light");
    }
});

// ============================================
// 4. Markdown Configuration
// ============================================
if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
}

// ============================================
// 5. Utility Functions
// ============================================
function setStatus(text, mode) {
    statusBadge.textContent = text;
    statusBadge.className = "status-badge";
    if (mode) statusBadge.classList.add(mode);
}

function scrollToBottom() {
    conversation.scrollTop = conversation.scrollHeight;
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

// ============================================
// 6. Toast Notifications
// ============================================
function showToast(message, type) {
    var container = document.getElementById("toastContainer");
    var toast = document.createElement("div");
    toast.className = "toast" + (type ? " toast-" + type : "");
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(function () {
        toast.classList.add("toast-leave");
        setTimeout(function () { toast.remove(); }, 250);
    }, 3000);
}

// ============================================
// 7. Message Rendering
// ============================================
function addMessage(role, content, sources) {
    sources = sources || [];

    var messageDiv = document.createElement("div");
    messageDiv.className = "message " + role + "-message";

    // Avatar
    var avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "\u{1F464}" : "\u{1F373}";
    messageDiv.appendChild(avatar);

    // Content bubble
    var contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    if (role === "assistant" && typeof marked !== "undefined") {
        contentDiv.innerHTML = marked.parse(content);
    } else {
        contentDiv.textContent = content;
    }
    messageDiv.appendChild(contentDiv);

    // Sources (assistant messages only)
    if (role === "assistant" && sources.length > 0) {
        var sourcesDiv = document.createElement("div");
        sourcesDiv.className = "message-sources";

        var label = document.createElement("span");
        label.className = "message-sources-label";
        label.textContent = "参考来源";
        sourcesDiv.appendChild(label);

        sources.forEach(function (s) {
            var badge = document.createElement("span");
            badge.className = "source-badge";
            badge.textContent = s;
            sourcesDiv.appendChild(badge);
        });

        messageDiv.appendChild(sourcesDiv);
    }

    conversation.appendChild(messageDiv);
    messageHistory.push({ role: role, content: content, sources: sources });
    scrollToBottom();

    return messageDiv;
}

// ============================================
// 8. Typing Indicator
// ============================================
function showTypingIndicator() {
    hideTypingIndicator();

    var wrapper = document.createElement("div");
    wrapper.className = "message assistant-message";
    wrapper.id = "typingIndicator";

    var avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "\u{1F373}";
    wrapper.appendChild(avatar);

    var content = document.createElement("div");
    content.className = "message-content typing-indicator";

    for (var i = 0; i < 3; i++) {
        var dot = document.createElement("span");
        dot.className = "typing-dot";
        content.appendChild(dot);
    }

    wrapper.appendChild(content);
    conversation.appendChild(wrapper);
    scrollToBottom();
}

function hideTypingIndicator() {
    var el = document.getElementById("typingIndicator");
    if (el) el.remove();
}

// ============================================
// 9. Form Submission
// ============================================
form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Guard against double-submit (e.g. chip click during a pending request)
    if (submitButton.disabled) return;

    var query = input.value.trim();
    if (!query) {
        setStatus("请输入问题", "error");
        input.focus();
        return;
    }

    submitButton.disabled = true;
    setStatus("正在生成", "loading");

    // Add user message and clear the input
    addMessage("user", query);
    input.value = "";
    input.style.height = "auto";
    showTypingIndicator();

    try {
        var response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query }),
        });

        var data = await response.json();
        hideTypingIndicator();

        if (!response.ok) {
            throw new Error(data.detail || "服务端返回了错误");
        }

        addMessage("assistant", data.answer || "未返回回答内容", data.sources || []);
        setStatus("回答完成", "success");
    } catch (error) {
        hideTypingIndicator();
        var errMsg = error.message || "请求失败，请稍后重试";
        addMessage("assistant", "请求失败：" + errMsg);
        setStatus("出现错误", "error");
        showToast(errMsg, "error");
    } finally {
        submitButton.disabled = false;
    }
});

// ============================================
// 10. Keyboard Shortcuts
// ============================================
input.addEventListener("keydown", function (e) {
    // Enter submits (Shift+Enter inserts newline; isComposing guards IME)
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        form.requestSubmit();
    }
});

// ============================================
// 11. Auto-resize Textarea
// ============================================
input.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 150) + "px";
});

// ============================================
// 12. Preset Chips
// ============================================
chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
        input.value = this.dataset.prompt || "";
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 150) + "px";
        form.requestSubmit();
    });
});

// ============================================
// 13. Clear Conversation
// ============================================
clearChatBtn.addEventListener("click", function () {
    var messages = conversation.querySelectorAll(".message");
    // Keep the welcome message (first child), remove the rest
    for (var i = 1; i < messages.length; i++) {
        messages[i].remove();
    }
    messageHistory.length = 0;
    setStatus("就绪", "");
    showToast("对话已清除", "success");
});
