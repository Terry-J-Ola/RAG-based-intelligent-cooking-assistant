const form = document.getElementById("chatForm");
const input = document.getElementById("queryInput");
const submitButton = document.getElementById("submitButton");
const resultCard = document.getElementById("resultCard");
const statusBadge = document.getElementById("statusBadge");
const chips = document.querySelectorAll(".prompt-chip");

chips.forEach((chip) => {
    chip.addEventListener("click", () => {
        input.value = chip.dataset.prompt || "";
        input.focus();
    });
});

function setStatus(text, mode = "") {
    statusBadge.textContent = text;
    statusBadge.className = "status-badge";
    if (mode) {
        statusBadge.classList.add(mode);
    }
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#39;");
}

function renderResult(answer, sources = []) {
    const sourceItems = sources.length
        ? `<p class="sources-title">参考来源</p><ul class="sources-list">${sources
              .map((item) => `<li>${escapeHtml(item)}</li>`)
              .join("")}</ul>`
        : "";

    resultCard.classList.remove("is-empty");
    resultCard.innerHTML = `
        <p class="answer-title">回答</p>
        <p class="answer-body">${escapeHtml(answer)}</p>
        ${sourceItems}
    `;
}

function renderError(message) {
    resultCard.classList.remove("is-empty");
    resultCard.innerHTML = `
        <p class="answer-title">请求失败</p>
        <p class="answer-body">${escapeHtml(message)}</p>
    `;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const query = input.value.trim();
    if (!query) {
        setStatus("请输入问题", "error");
        input.focus();
        return;
    }

    submitButton.disabled = true;
    setStatus("正在生成", "loading");
    renderResult("正在检索知识库并生成回答，请稍候。");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ query }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "服务端返回了错误");
        }

        renderResult(data.answer || "未返回回答内容", data.sources || []);
        setStatus("回答完成", "success");
    } catch (error) {
        renderError(error.message || "请求失败，请稍后重试");
        setStatus("出现错误", "error");
    } finally {
        submitButton.disabled = false;
    }
});
