const form = document.querySelector("#question-form");
const question = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const loadingMessage = document.querySelector("#loading-message");
const result = document.querySelector("#result");
const answer = document.querySelector("#answer");
const notice = document.querySelector("#notice");
const sourcesSection = document.querySelector("#sources-section");
const sourcesList = document.querySelector("#sources-list");
const errorMessage = document.querySelector("#error-message");

const insufficientMessage = "Mevcut dok\u00fcmanlarda bu soruyu yan\u0131tlamak i\u00e7in yeterli bilgi bulunamad\u0131.";
const requestErrorMessage = "\u015eu anda yan\u0131t olu\u015fturulamad\u0131. L\u00fctfen tekrar deneyin.";

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) {
    showError("L\u00fctfen bir soru yaz\u0131n.");
    question.focus();
    return;
  }

  setLoading(true);
  clearDisplay();
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text }),
    });
    if (!response.ok) {
      throw new Error("İstek tamamlanamadı.");
    }

    const data = await response.json();
    showResult(data);
  } catch {
    showError(requestErrorMessage);
  } finally {
    setLoading(false);
  }
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function setLoading(isLoading) {
  askButton.disabled = isLoading;
  loadingMessage.hidden = !isLoading;
}

function clearDisplay() {
  result.hidden = true;
  errorMessage.hidden = true;
  notice.hidden = true;
  answer.textContent = "";
  sourcesList.replaceChildren();
  sourcesSection.hidden = true;
}

function showResult(data) {
  if (data.status === "insufficient_evidence") {
    answer.textContent = insufficientMessage;
  } else {
    answer.textContent = data.answer;
  }

  if (data.status === "conflict") {
    notice.textContent = "Kaynaklarda birbiriyle uyu\u015fmayan bilgiler bulundu.";
    notice.hidden = false;
  }

  renderSources(data.sources || []);
  result.hidden = false;
}

function renderSources(sources) {
  const seen = new Set();
  for (const source of sources) {
    const key = `${source.source_name}:${source.page_start}:${source.page_end}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const item = document.createElement("li");
    item.textContent = source.source_name;
    const page = document.createElement("span");
    page.className = "source-page";
    page.textContent = source.page_start === source.page_end
      ? `Sayfa ${source.page_start}`
      : `Sayfa ${source.page_start}–${source.page_end}`;
    item.append(page);
    sourcesList.append(item);
  }
  sourcesSection.hidden = sourcesList.childElementCount === 0;
}

function showError(message) {
  clearDisplay();
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}
